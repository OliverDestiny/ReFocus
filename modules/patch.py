# This file is part of ReFocus.
# Original work Copyright (c) 2023 lllyasviel (Fooocus) & 2024 ehristoforu (DeFooocus).
# Modified and distributed under the terms of the GNU General Public License v3.0.

"""Injection layer between ReFocus and the vendored ComfyUI core.

Keep this file minimal: it is the whole coupling surface to upstream, and only abilities
upstream lacks belong here. See README_DEV, "Vendored core (`comfy/`)".
"""

import os
import torch
import time
import math
import warnings
import safetensors.torch

import comfy.model_base
import comfy.cldm.cldm
import comfy.model_management
import comfy.model_patcher

import modules.anisotropic as anisotropic


class PatchSettings:
    def __init__(self,
                 sharpness=2.0,
                 adm_scaler_end=0.3,
                 positive_adm_scale=1.5,
                 negative_adm_scale=0.8,
                 controlnet_softness=0.25,
                 adaptive_cfg=7.0):
        self.sharpness = sharpness
        self.adm_scaler_end = adm_scaler_end
        self.positive_adm_scale = positive_adm_scale
        self.negative_adm_scale = negative_adm_scale
        self.controlnet_softness = controlnet_softness
        self.adaptive_cfg = adaptive_cfg
        # The VAE refiner swap records eps during sampling; None disables recording.
        self.eps_record = None


patch_settings = {}


def settings():
    """Return this process's PatchSettings, written by async_worker at each task start."""
    return patch_settings[os.getpid()]


# ---------------------------------------------------------------------------
# ADM guidance (no upstream equivalent, so it must stay)
# ---------------------------------------------------------------------------

def round_to_64(x):
    h = float(x)
    h = h / 64.0
    h = round(h)
    h = int(h)
    h = h * 64
    return h


def timed_adm(y, timesteps):
    """Fade ADM out over late steps per adm_scaler_end; upstream lacks this, so apply it to both y paths."""
    if isinstance(y, torch.Tensor) and int(y.dim()) == 2 and int(y.shape[1]) == 5632:
        y_mask = (timesteps > 999.0 * (1.0 - float(settings().adm_scaler_end))).to(y)[..., None]
        y_with_adm = y[..., :2816].clone()
        y_without_adm = y[..., 2816:].clone()
        return y_with_adm * y_mask + y_without_adm * (1.0 - y_mask)
    return y


def sdxl_encode_adm_patched(self, **kwargs):
    clip_pooled = comfy.model_base.sdxl_pooled(kwargs, self.noise_augmentor)
    width = kwargs.get("width", 1024)
    height = kwargs.get("height", 1024)
    target_width = width
    target_height = height

    if kwargs.get("prompt_type", "") == "negative":
        width = float(width) * settings().negative_adm_scale
        height = float(height) * settings().negative_adm_scale
    elif kwargs.get("prompt_type", "") == "positive":
        width = float(width) * settings().positive_adm_scale
        height = float(height) * settings().positive_adm_scale

    def embedder(number_list):
        h = self.embedder(torch.tensor(number_list, dtype=torch.float32))
        h = torch.flatten(h).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        return h

    width, height = int(width), int(height)
    target_width, target_height = round_to_64(target_width), round_to_64(target_height)

    adm_emphasized = embedder([height, width, 0, 0, target_height, target_width])
    adm_consistent = embedder([target_height, target_width, 0, 0, target_height, target_width])

    clip_pooled = clip_pooled.to(adm_emphasized)
    final_adm = torch.cat((clip_pooled, adm_emphasized, clip_pooled, adm_consistent), dim=1)

    return final_adm


# ---------------------------------------------------------------------------
# CFG hook: sharpness + adaptive CFG (TSNR mimicry) + eps_record
# Replaces the old whole-function swap of comfy.samplers.sampling_function; attached through
# model_options["sampler_cfg_function"], the same extension point upstream nodes use.
# ---------------------------------------------------------------------------

def diffusion_progress(model, sigma):
    """Convert sigma to 0..1 diffusion progress via the model's own model_sampling.timestep.

    Only LCM swaps model_sampling, and LCM forces sharpness 0 / cfg 1, so the stale value is unused there.
    """
    t = model.model_sampling.timestep(sigma)
    return float(1.0 - (t.flatten()[0].item() / 999.0))


def compute_cfg(uncond, cond, cfg_scale, t):
    mimic_cfg = float(settings().adaptive_cfg)
    real_cfg = float(cfg_scale)

    real_eps = uncond + real_cfg * (cond - uncond)

    if cfg_scale > settings().adaptive_cfg:
        mimicked_eps = uncond + mimic_cfg * (cond - uncond)
        return real_eps * t + mimicked_eps * (1 - t)
    else:
        return real_eps


def custom_sampler_cfg_function(args):
    """comfy sampler_cfg_function: cond/uncond arrive as eps and the return value must be eps too.

    At cfg ~= 1 upstream skips the uncond branch, so args["uncond"] is x and cannot be used here.
    """
    x = args["input"]
    sigma = args["sigma"]
    cond_scale = args["cond_scale"]
    cond_eps = args["cond"]
    uncond_eps = args["uncond"]
    cond_denoised = args["cond_denoised"]

    progress = diffusion_progress(args["model"], sigma)

    if math.isclose(cond_scale, 1.0) and not args["model_options"].get("disable_cfg1_optimization", False):
        final_eps = cond_eps
    else:
        alpha = 0.001 * settings().sharpness * progress
        cond_eps_degraded = anisotropic.adaptive_anisotropic_filter(x=cond_eps, g=cond_denoised)
        cond_eps_weighted = cond_eps_degraded * alpha + cond_eps * (1.0 - alpha)
        final_eps = compute_cfg(uncond=uncond_eps, cond=cond_eps_weighted, cfg_scale=cond_scale, t=progress)

    if settings().eps_record is not None:
        settings().eps_record = (final_eps / sigma).cpu()

    return final_eps


def attach_model_hooks(model):
    """Attach the CFG hook and the UNet wrapper to one model; call it for base and refiner both.

    The wrapper re-applies timed_adm, compressing the 5632-wide double ADM back to 2816 before label_emb.
    """
    global _active_model_sampling
    _active_model_sampling = model.get_model_object("model_sampling")

    model.set_model_sampler_cfg_function(custom_sampler_cfg_function)

    def unet_wrapper(apply_model, args):
        c = args["c"]

        # args["timestep"] is sigma, while timed_adm and current_step need a 0..999 discrete timestep.
        # Read model_sampling through get_model_object so the LCM-replaced one is picked up.
        t = model.get_model_object("model_sampling").timestep(args["timestep"])

        # current_step is the 0..1 diffusion progress; extras/ip_adapter.py:210 reads it to decide
        # whether IP-Adapter applies at this step (against cn_stop). It must stay a tensor.
        model.model.diffusion_model.current_step = 1.0 - t / 999.0

        if c.get("y") is not None:
            c["y"] = timed_adm(c["y"], t)
        return apply_model(args["input"], args["timestep"], **c)

    model.set_model_unet_function_wrapper(unet_wrapper)
    return model


# ---------------------------------------------------------------------------
# ControlNet: thin wrapper, not a rewrite of the whole forward
# Upstream forward already handles union ControlNets and the {"middle", "output"} dict return;
# this only adds timed_adm at entry and softness scaling at exit, so upstream changes carry over.
# ---------------------------------------------------------------------------

_cldm_forward_origin = None

# ControlNet.forward receives sigma and no model handle, so stash model_sampling at hook time
# for the conversion here. Only the last one is kept if base and refiner are both mounted.
_active_model_sampling = None


def patched_cldm_forward(self, x, hint, timesteps, context, y=None, **kwargs):
    if _active_model_sampling is not None and y is not None:
        timesteps = _active_model_sampling.timestep(timesteps)
    y = timed_adm(y, timesteps)

    out = _cldm_forward_origin(self, x, hint, timesteps, context, y=y, **kwargs)

    softness = settings().controlnet_softness
    if softness > 0 and isinstance(out, dict):
        outputs = out.get("output", [])
        for i in range(min(10, len(outputs))):
            k = 1.0 - float(i) / 9.0
            outputs[i] = outputs[i] * (1.0 - softness * k)

    return out


# ---------------------------------------------------------------------------
# Model-management timing and corrupted-weight recovery (independent of core behaviour)
# ---------------------------------------------------------------------------

def patched_load_models_gpu(*args, **kwargs):
    execution_start_time = time.perf_counter()
    y = comfy.model_management.load_models_gpu_origin(*args, **kwargs)
    moving_time = time.perf_counter() - execution_start_time
    if moving_time > 0.1:
        print(f'[ReFocus Model Management] Moving model(s) has taken {moving_time:.2f} seconds')
    return y


def build_loaded(module, loader_name):
    original_loader_name = loader_name + '_origin'

    if not hasattr(module, original_loader_name):
        setattr(module, original_loader_name, getattr(module, loader_name))

    original_loader = getattr(module, original_loader_name)

    def loader(*args, **kwargs):
        result = None
        try:
            result = original_loader(*args, **kwargs)
        except Exception as e:
            result = None
            exp = str(e) + '\n'
            for path in list(args) + list(kwargs.values()):
                if isinstance(path, str):
                    if os.path.exists(path):
                        exp += f'File corrupted: {path} \n'
                        corrupted_backup_file = path + '.corrupted'
                        if os.path.exists(corrupted_backup_file):
                            os.remove(corrupted_backup_file)
                        os.replace(path, corrupted_backup_file)
                        if os.path.exists(path):
                            os.remove(path)
                        exp += f'ReFocus has tried to move the corrupted file to {corrupted_backup_file} \n'
                        exp += f'You may try again now and ReFocus will download models again. \n'
            raise ValueError(exp)
        return result

    setattr(module, loader_name, loader)
    return


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def _install(target, name, replacement):
    """Replace an attribute and assert it took effect, so a changed vendored core fails at startup."""
    setattr(target, name, replacement)
    if getattr(target, name) is not replacement:
        raise RuntimeError(
            f'[ReFocus] Patch did not take effect: {getattr(target, "__name__", target)}.{name}. '
            f'The vendored core may have changed; re-check modules/patch.py against upstream.'
        )


def patch_all():
    global _cldm_forward_origin

    if comfy.model_management.directml_enabled:
        comfy.model_management.lowvram_available = True
        comfy.model_management.OOM_EXCEPTION = Exception

    if not hasattr(comfy.model_management, 'load_models_gpu_origin'):
        comfy.model_management.load_models_gpu_origin = comfy.model_management.load_models_gpu
    _install(comfy.model_management, 'load_models_gpu', patched_load_models_gpu)

    # Capture the original forward once; repeated patch_all() calls must not save the patched one.
    if _cldm_forward_origin is None:
        _cldm_forward_origin = comfy.cldm.cldm.ControlNet.forward
    _install(comfy.cldm.cldm.ControlNet, 'forward', patched_cldm_forward)

    _install(comfy.model_base.SDXL, 'encode_adm', sdxl_encode_adm_patched)

    warnings.filterwarnings(action='ignore', module='torchsde')

    build_loaded(safetensors.torch, 'load_file')
    build_loaded(torch, 'load')
    for module, name in ((safetensors.torch, 'load_file'), (torch, 'load')):
        if not hasattr(module, name + '_origin'):
            raise RuntimeError(f'[ReFocus] Loader wrapper did not take effect: {name}.')

    return
