import os
import einops
import torch
import numpy as np

import comfy.model_management
import comfy.model_detection
import comfy.model_patcher
import comfy.utils
import comfy.controlnet
import comfy.latent_formats
import comfy.sample

import modules.sample_hijack
import modules.comfy_ops as comfy_ops

from comfy.sd import load_checkpoint_guess_config
from modules.lora import load_lora_patches
from comfy.lora import model_lora_keys_unet, model_lora_keys_clip
from modules.config import path_embeddings


class StableDiffusionModel:
    def __init__(self, unet=None, vae=None, clip=None, clip_vision=None, filename=None):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision
        self.filename = filename
        self.unet_with_lora = unet
        self.clip_with_lora = clip
        self.visited_loras = ''

        self.lora_key_map_unet = {}
        self.lora_key_map_clip = {}

        if self.unet is not None:
            self.lora_key_map_unet = model_lora_keys_unet(self.unet.model, self.lora_key_map_unet)
            self.lora_key_map_unet.update({x: x for x in self.unet.model.state_dict().keys()})

        if self.clip is not None:
            self.lora_key_map_clip = model_lora_keys_clip(self.clip.cond_stage_model, self.lora_key_map_clip)
            self.lora_key_map_clip.update({x: x for x in self.clip.cond_stage_model.state_dict().keys()})

    @torch.no_grad()
    @torch.inference_mode()
    def refresh_loras(self, loras):
        assert isinstance(loras, list)

        if self.visited_loras == str(loras):
            return

        self.visited_loras = str(loras)

        if self.unet is None:
            return

        print(f'Request to load LoRAs {str(loras)} for model [{self.filename}].')

        loras_to_load = []

        for name, weight in loras:
            if name == 'None':
                continue

            if os.path.exists(name):
                lora_filename = name
            else:
                lora_filename = os.path.join(modules.config.path_loras, name)

            if not os.path.exists(lora_filename):
                print(f'Lora file not found: {lora_filename}')
                continue

            loras_to_load.append((lora_filename, weight))

        self.unet_with_lora = self.unet.clone() if self.unet is not None else None
        self.clip_with_lora = self.clip.clone() if self.clip is not None else None

        for lora_filename, weight in loras_to_load:
            lora_sd = comfy.utils.load_torch_file(lora_filename, safe_load=False)
            lora_unet = load_lora_patches(lora_sd, self.lora_key_map_unet)
            lora_clip = load_lora_patches(lora_sd, self.lora_key_map_clip)

            if len(lora_unet) == 0 and len(lora_clip) == 0:
                # 与当前模型完全不匹配（原实现用「未匹配 key 超过 12 个」判断，是启发式）
                print(f'LoRA [{lora_filename}] does not match model [{self.filename}], skipped.')
                continue

            if self.unet_with_lora is not None and len(lora_unet) > 0:
                loaded_keys = self.unet_with_lora.add_patches(lora_unet, weight)
                print(f'Loaded LoRA [{lora_filename}] for UNet [{self.filename}] '
                      f'with {len(loaded_keys)} keys at weight {weight}.')
                for item in lora_unet:
                    if item not in loaded_keys:
                        print("UNet LoRA key skipped: ", item)

            if self.clip_with_lora is not None and len(lora_clip) > 0:
                loaded_keys = self.clip_with_lora.add_patches(lora_clip, weight)
                print(f'Loaded LoRA [{lora_filename}] for CLIP [{self.filename}] '
                      f'with {len(loaded_keys)} keys at weight {weight}.')
                for item in lora_clip:
                    if item not in loaded_keys:
                        print("CLIP LoRA key skipped: ", item)


@torch.no_grad()
@torch.inference_mode()
def apply_freeu(model, b1, b2, s1, s2):
    return comfy_ops.freeu_patch(model, b1, b2, s1, s2)


@torch.no_grad()
@torch.inference_mode()
def apply_model_sampling_discrete(model, sampling='lcm', zsnr=False):
    return comfy_ops.model_sampling_discrete_patch(model, sampling, zsnr)


@torch.no_grad()
@torch.inference_mode()
def load_controlnet(ckpt_filename):
    return comfy.controlnet.load_controlnet(ckpt_filename)


@torch.no_grad()
@torch.inference_mode()
def apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent):
    return comfy_ops.controlnet_apply_advanced(positive, negative, control_net,
                                               image, strength, start_percent, end_percent)


@torch.no_grad()
@torch.inference_mode()
def load_model(ckpt_filename):
    unet, clip, vae, clip_vision = load_checkpoint_guess_config(ckpt_filename, embedding_directory=path_embeddings)
    return StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision, filename=ckpt_filename)


@torch.no_grad()
@torch.inference_mode()
def generate_empty_latent(width=1024, height=1024, batch_size=1):
    return comfy_ops.empty_latent(width, height, batch_size)


@torch.no_grad()
@torch.inference_mode()
def decode_vae(vae, latent_image, tiled=False):
    if tiled:
        return comfy_ops.vae_decode_tiled(vae, latent_image, tile_size=512)
    else:
        return comfy_ops.vae_decode(vae, latent_image)


@torch.no_grad()
@torch.inference_mode()
def encode_vae(vae, pixels, tiled=False):
    if tiled:
        return comfy_ops.vae_encode_tiled(vae, pixels, tile_size=512)
    else:
        return comfy_ops.vae_encode(vae, pixels)


@torch.no_grad()
@torch.inference_mode()
def encode_vae_inpaint(vae, pixels, mask):
    assert mask.ndim == 3 and pixels.ndim == 4
    assert mask.shape[-1] == pixels.shape[-2]
    assert mask.shape[-2] == pixels.shape[-3]

    w = mask.round()[..., None]
    pixels = pixels * (1 - w) + 0.5 * w

    latent = vae.encode(pixels)
    B, C, H, W = latent.shape

    latent_mask = mask[:, None, :, :]
    latent_mask = torch.nn.functional.interpolate(latent_mask, size=(H * 8, W * 8), mode="bilinear").round()
    latent_mask = torch.nn.functional.max_pool2d(latent_mask, (8, 8)).round().to(latent)

    return latent, latent_mask


class VAEApprox(torch.nn.Module):
    def __init__(self):
        super(VAEApprox, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, (7, 7))
        self.conv2 = torch.nn.Conv2d(8, 16, (5, 5))
        self.conv3 = torch.nn.Conv2d(16, 32, (3, 3))
        self.conv4 = torch.nn.Conv2d(32, 64, (3, 3))
        self.conv5 = torch.nn.Conv2d(64, 32, (3, 3))
        self.conv6 = torch.nn.Conv2d(32, 16, (3, 3))
        self.conv7 = torch.nn.Conv2d(16, 8, (3, 3))
        self.conv8 = torch.nn.Conv2d(8, 3, (3, 3))
        self.current_type = None

    def forward(self, x):
        extra = 11
        x = torch.nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        x = torch.nn.functional.pad(x, (extra, extra, extra, extra))
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
        return x


VAE_approx_models = {}


@torch.no_grad()
@torch.inference_mode()
def get_previewer(model):
    global VAE_approx_models

    from modules.config import path_vae_approx
    is_sdxl = isinstance(model.model.latent_format, comfy.latent_formats.SDXL)
    vae_approx_filename = os.path.join(path_vae_approx, 'xlvaeapp.pth' if is_sdxl else 'vaeapp_sd15.pth')

    if vae_approx_filename in VAE_approx_models:
        VAE_approx_model = VAE_approx_models[vae_approx_filename]
    else:
        sd = torch.load(vae_approx_filename, map_location='cpu')
        VAE_approx_model = VAEApprox()
        VAE_approx_model.load_state_dict(sd)
        del sd
        VAE_approx_model.eval()

        if comfy.model_management.should_use_fp16():
            VAE_approx_model.half()
            VAE_approx_model.current_type = torch.float16
        else:
            VAE_approx_model.float()
            VAE_approx_model.current_type = torch.float32

        VAE_approx_model.to(comfy.model_management.get_torch_device())
        VAE_approx_models[vae_approx_filename] = VAE_approx_model

    @torch.no_grad()
    @torch.inference_mode()
    def preview_function(x0, step, total_steps):
        with torch.no_grad():
            x_sample = x0.to(VAE_approx_model.current_type)
            x_sample = VAE_approx_model(x_sample) * 127.5 + 127.5
            x_sample = einops.rearrange(x_sample, 'b c h w -> b h w c')[0]
            x_sample = x_sample.cpu().numpy().clip(0, 255).astype(np.uint8)
            return x_sample

    return preview_function


@torch.no_grad()
@torch.inference_mode()
def ksampler(model, positive, negative, latent, seed=None, steps=30, cfg=7.0, sampler_name='dpmpp_2m_sde_gpu',
             scheduler='karras', denoise=1.0, disable_noise=False, start_step=None, last_step=None,
             force_full_denoise=False, callback_function=None,
             previewer_start=None, previewer_end=None, sigmas=None, noise_mean=None, disable_preview=False):
    # 注意：这里不再有 refiner / refiner_switch 参数。原来它们驱动 sample_hijack 的
    # 「采样中途换模型」，现代核心没有对应挂载点，已改为 process_diffusion 的原生两遍采样。

    if sigmas is not None:
        sigmas = sigmas.clone().to(comfy.model_management.get_torch_device())

    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    if isinstance(noise_mean, torch.Tensor):
        noise = noise + noise_mean - torch.mean(noise, dim=1, keepdim=True)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    previewer = get_previewer(model)

    if previewer_start is None:
        previewer_start = 0

    if previewer_end is None:
        previewer_end = steps

    warned_partial_load = False

    def callback(step, x0, x, total_steps):
        nonlocal warned_partial_load
        comfy.model_management.throw_exception_if_processing_interrupted()
        if step == 0 and not warned_partial_load:
            warned_partial_load = warn_if_partially_loaded(model) or warned_partial_load
        y = None
        if previewer is not None and not disable_preview:
            y = previewer(x0, previewer_start + step, previewer_end)
        if callback_function is not None:
            callback_function(previewer_start + step, x0, x, previewer_end, y)

    samples = comfy.sample.sample(model,
                                  noise, steps, cfg, sampler_name, scheduler,
                                  positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise,
                                  start_step=start_step,
                                  last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask,
                                  callback=callback,
                                  disable_pbar=False, seed=seed, sigmas=sigmas)

    out = latent.copy()
    out["samples"] = samples

    return out


def warn_if_partially_loaded(model, label=None):
    """模型被 comfy 部分加载时出声。

    显存不够时 `load_models_gpu` 不会报错，只会把一部分权重留在内存里、每步搬上来，
    于是同一个模型从 ~1.7 秒/步掉到 ~98 秒/步；而且部分加载走 per-op manual_cast，
    **数值也会变**。这种"只是变慢/悄悄变了"的失效必须显式说出来。

    只能在采样过程中调用：采样一结束 comfy 就把非持久模型卸掉了，
    那时 loaded_size() 恒为 0，会误报。
    """
    if label is None:
        label = type(model.model).__name__ if hasattr(model, 'model') else 'model'
    try:
        patcher = getattr(model, 'patcher', model)
        loaded, size = patcher.loaded_size(), patcher.model_size()
    except Exception:
        return False
    if size <= 0 or loaded >= size - 1:
        return False
    free = comfy.model_management.get_free_memory() / (1024 ** 3)
    print(f'[{label}] WARNING: only {loaded / 1024 ** 3:.2f}G of {size / 1024 ** 3:.2f}G of the model '
          f'is in VRAM ({free:.2f}G free), so every step streams weights from system memory and can '
          f'take a minute or more. Lower the resolution, close other GPU applications, or free memory.')
    return True


@torch.no_grad()
@torch.inference_mode()
def pytorch_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y
