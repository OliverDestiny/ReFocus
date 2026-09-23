# This file is part of ReFocus.
# Original work Copyright (c) 2023 lllyasviel (Fooocus) & 2024 ehristoforu (DeFooocus).
# Modified and distributed under the terms of the GNU General Public License v3.0.

"""Cross-model conditioning helpers plus registration of the extra schedulers the UI exposes.

`clip_separate*` cut text conditioning for the target model (SDXLRefiner: last 1280 dims; SDXL:
as-is; SD1.5: first 768 dims with clip_l's final_layer_norm rerun). `sample_hacked` and
`calculate_sigmas_scheduler_hacked` are gone; refiner swapping uses process_diffusion's two passes.
"""

import torch

import comfy.model_base
import comfy.samplers

import modules.comfy_ops as comfy_ops


@torch.no_grad()
@torch.inference_mode()
def clip_separate_inner(c, p, target_model=None, target_clip=None):
    if target_model is None or isinstance(target_model, comfy.model_base.SDXLRefiner):
        c = c[..., -1280:].clone()
    elif isinstance(target_model, comfy.model_base.SDXL):
        c = c.clone()
    else:
        p = None
        c = c[..., :768].clone()

        final_layer_norm = target_clip.cond_stage_model.clip_l.transformer.text_model.final_layer_norm

        final_layer_norm_origin_device = final_layer_norm.weight.device
        final_layer_norm_origin_dtype = final_layer_norm.weight.dtype

        c_origin_device = c.device
        c_origin_dtype = c.dtype

        final_layer_norm.to(device='cpu', dtype=torch.float32)
        c = c.to(device='cpu', dtype=torch.float32)

        c = torch.chunk(c, int(c.size(1)) // 77, 1)
        c = [final_layer_norm(ci) for ci in c]
        c = torch.cat(c, dim=1)

        final_layer_norm.to(device=final_layer_norm_origin_device, dtype=final_layer_norm_origin_dtype)
        c = c.to(device=c_origin_device, dtype=c_origin_dtype)
    return c, p


@torch.no_grad()
@torch.inference_mode()
def clip_separate(cond, target_model=None, target_clip=None):
    results = []

    for c, px in cond:
        p = px.get('pooled_output', None)
        c, p = clip_separate_inner(c, p, target_model=target_model, target_clip=target_clip)
        p = {} if p is None else {'pooled_output': p.clone()}
        results.append([c, p])

    return results


def register_extra_schedulers():
    """Register 'turbo', which the UI exposes but the core's SCHEDULER_HANDLERS does not have.
    An unknown name makes KSampler.__init__ silently fall back to SCHEDULERS[0] (simple), not an error.
    """
    if 'turbo' not in comfy.samplers.SCHEDULER_HANDLERS:
        comfy.samplers.SCHEDULER_HANDLERS['turbo'] = comfy.samplers.SchedulerHandler(comfy_ops.turbo_sigmas)
        comfy.samplers.SCHEDULER_NAMES.append('turbo')
    return


register_extra_schedulers()
