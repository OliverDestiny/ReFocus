# This file is part of ReFocus.
# Original work Copyright (c) 2023 lllyasviel (Fooocus) & 2024 ehristoforu (DeFooocus).
# Modified and distributed under the terms of the GNU General Public License v3.0.

"""上游 ComfyUI 里没有随核心搬运进来、但 ReFocus 需要的那几个 op。

`comfy/` 是逐字节的上游拷贝，但上游把「节点实现」放在仓库根的 `nodes.py` 与
`comfy_extras/` 里，这两处都没搬（它们各自会拖进 comfy_api、folder_paths、
latent_preview 和一大批无关架构）。ReFocus 实际只用到其中少数几个 op，
按函数体移植到这里，而不是引入整套节点注册机制。

**换核心 tag 时，本文件是必须重新核对的清单**：上游这些实现曾多次改签名
（VAEEncodeTiled 现在必须传 overlap、set_cond_hint 多了 vae/extra_concat、
SDTurboScheduler 改走 get_model_object），逐条对照上游同名类即可。

注意：`vae_encode_tiled` / `vae_decode_tiled` **必须在 `torch.inference_mode()` 内调用**。
`comfy.utils.tiled_scale_multidim` 带 `@torch.inference_mode()` 装饰器，返回的是 inference
张量，而 `comfy/sd.py` 的 `encode_tiled_` 会对它做原地累加——在 inference_mode 之外会抛
"Inplace update to inference tensor outside InferenceMode"。ReFocus 的 `handler` 本身
就在 `@torch.inference_mode()` 下运行，所以应用路径上没有这个问题。

来源标注：
  FreeU_V2 / Fourier_filter      <- comfy_extras/nodes_freelunch.py
  LCM / ModelSamplingDiscrete*   <- comfy_extras/nodes_model_advanced.py
  SDTurboScheduler               <- comfy_extras/nodes_custom_sampler.py
  EmptyLatentImage 等 VAE/CN op  <- nodes.py
"""

import logging

import torch

import comfy.model_management
import comfy.model_sampling


# ---------------------------------------------------------------------------
# FreeU（来源：comfy_extras/nodes_freelunch.py）
# ---------------------------------------------------------------------------

def Fourier_filter(x, threshold, scale):
    # code originally taken from: https://github.com/ChenyangSi/FreeU (under MIT License)
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(x.dtype)


def freeu_patch(model, b1, b2, s1, s2):
    """FreeU_V2：按通道数缩放 skip 分支，并对 hsp 做傅里叶滤波。返回打了补丁的 model。"""
    model_channels = model.model.model_config.unet_config["model_channels"]
    scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
    on_cpu_devices = {}

    def output_block_patch(h, hsp, transformer_options):
        scale = scale_dict.get(int(h.shape[1]), None)
        if scale is not None:
            hidden_mean = h.mean(1).unsqueeze(1)
            B = hidden_mean.shape[0]
            hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

            h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2] * ((scale[0] - 1) * hidden_mean + 1)

            if hsp.device not in on_cpu_devices:
                try:
                    hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
                except Exception:
                    logging.warning("Device {} does not support the torch.fft functions used in the FreeU node, switching to CPU.".format(hsp.device))
                    on_cpu_devices[hsp.device] = True
                    hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)
            else:
                hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)

        return h, hsp

    m = model.clone()
    m.set_model_output_block_patch(output_block_patch)
    return m


# ---------------------------------------------------------------------------
# ModelSamplingDiscrete（来源：comfy_extras/nodes_model_advanced.py）
# LCM 与 Distilled 上游就定义在那个 extras 文件里，所以一并移植。
# ---------------------------------------------------------------------------

class LCM(comfy.model_sampling.EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        timestep = self.timestep(sigma).view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        x0 = model_input - model_output * sigma

        sigma_data = 0.5
        scaled_timestep = timestep * 10.0  # timestep_scaling

        c_skip = sigma_data ** 2 / (scaled_timestep ** 2 + sigma_data ** 2)
        c_out = scaled_timestep / (scaled_timestep ** 2 + sigma_data ** 2) ** 0.5

        return c_out * x0 + c_skip * model_input


class ModelSamplingDiscreteDistilled(comfy.model_sampling.ModelSamplingDiscrete):
    original_timesteps = 50

    def __init__(self, model_config=None, zsnr=None):
        super().__init__(model_config, zsnr=zsnr)

        self.skip_steps = self.num_timesteps // self.original_timesteps

        sigmas_valid = torch.zeros((self.original_timesteps), dtype=torch.float32)
        for x in range(self.original_timesteps):
            sigmas_valid[self.original_timesteps - 1 - x] = self.sigmas[self.num_timesteps - 1 - x * self.skip_steps]

        self.set_sigmas(sigmas_valid)

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return (dists.abs().argmin(dim=0).view(sigma.shape) * self.skip_steps + (self.skip_steps - 1)).to(sigma.device)

    def sigma(self, timestep):
        t = torch.clamp(((timestep.float().to(self.log_sigmas.device) - (self.skip_steps - 1)) / self.skip_steps).float(), min=0, max=(len(self.sigmas) - 1))
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp().to(timestep.device)


_SAMPLING_TYPES = {
    'eps': (comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.EPS),
    'v_prediction': (comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.V_PREDICTION),
    'lcm': (ModelSamplingDiscreteDistilled, LCM),
    'x0': (comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.X0),
    'img_to_img': (comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.IMG_TO_IMG),
    'img_to_img_flow': (comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.IMG_TO_IMG_FLOW),
}


def model_sampling_discrete_patch(model, sampling, zsnr=False):
    """替换模型的 model_sampling（LCM 模式走 'lcm'）。返回打了补丁的 model。"""
    sampling_base, sampling_type = _SAMPLING_TYPES[sampling]

    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass

    model_sampling = ModelSamplingAdvanced(model.model.model_config, zsnr=zsnr)

    m = model.clone()
    m.add_object_patch("model_sampling", model_sampling)
    return m


# ---------------------------------------------------------------------------
# SDTurboScheduler（来源：comfy_extras/nodes_custom_sampler.py）
# 参数从「模型」改成「model_sampling」，以便直接注册进 SCHEDULER_HANDLERS
# （该表的 use_ms=True 约定就是 handler(model_sampling, steps)）。
# ---------------------------------------------------------------------------

def turbo_sigmas(model_sampling, steps, denoise=1.0):
    start_step = 10 - int(10 * denoise)
    timesteps = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[start_step:start_step + steps]
    sigmas = model_sampling.sigma(timesteps)
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    return sigmas


# ---------------------------------------------------------------------------
# Latent / VAE（来源：nodes.py）
# ---------------------------------------------------------------------------

def empty_latent(width, height, batch_size=1):
    latent = torch.zeros([batch_size, 4, height // 8, width // 8],
                         device=comfy.model_management.intermediate_device(),
                         dtype=comfy.model_management.intermediate_dtype())
    return {"samples": latent, "downscale_ratio_spacial": 8}


def vae_decode(vae, samples):
    latent = samples["samples"]
    if latent.is_nested:
        latent = latent.unbind()[0]
    images = vae.decode(latent)
    if len(images.shape) == 5:  # Combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def vae_decode_tiled(vae, samples, tile_size=512, overlap=64):
    """上游 VAEDecodeTiled 的等价实现（含时间维压缩的适配）。"""
    if tile_size < overlap * 4:
        overlap = tile_size // 4

    temporal_compression = vae.temporal_compression_decode()
    if temporal_compression is not None:
        temporal_size = max(2, 64 // temporal_compression)
        temporal_overlap = max(1, min(temporal_size // 2, 8 // temporal_compression))
    else:
        temporal_size = None
        temporal_overlap = None

    latent = samples["samples"]
    if latent.is_nested:
        latent = latent.unbind()[0]

    compression = vae.spacial_compression_decode()
    images = vae.decode_tiled(latent,
                              tile_x=tile_size // compression, tile_y=tile_size // compression,
                              overlap=overlap // compression,
                              tile_t=temporal_size, overlap_t=temporal_overlap)
    if len(images.shape) == 5:  # Combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def vae_encode(vae, pixels):
    return {"samples": vae.encode(pixels)}


def vae_encode_tiled(vae, pixels, tile_size=512, overlap=64):
    """上游 VAEEncodeTiled 的等价实现。注意上游这里不缩放 tile 尺寸（由 encode_tiled 自己处理）。"""
    t = vae.encode_tiled(pixels, tile_x=tile_size, tile_y=tile_size, overlap=overlap,
                         tile_t=64, overlap_t=8)
    return {"samples": t}


# ---------------------------------------------------------------------------
# ControlNet（来源：nodes.py ControlNetApplyAdvanced）
# ---------------------------------------------------------------------------

def controlnet_apply_advanced(positive, negative, control_net, image, strength, start_percent, end_percent):
    if strength == 0:
        return (positive, negative)

    control_hint = image.movedim(-1, 1)
    cnets = {}

    out = []
    for conditioning in [positive, negative]:
        c = []
        for t in conditioning:
            d = t[1].copy()

            prev_cnet = d.get('control', None)
            if prev_cnet in cnets:
                c_net = cnets[prev_cnet]
            else:
                c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))
                c_net.set_previous_controlnet(prev_cnet)
                cnets[prev_cnet] = c_net

            d['control'] = c_net
            d['control_apply_to_uncond'] = False
            n = [t[0], d]
            c.append(n)
        out.append(c)
    return (out[0], out[1])


# ---------------------------------------------------------------------------
# 图像放大（来源：nodes.py ImageUpscaleWithModel）
# 与 VAE 的 tiled 编解码同理，comfy.utils.tiled_scale 带 @torch.inference_mode()，
# 所以本函数也必须在 inference_mode 内调用（ReFocus 的 handler 满足）。
# ---------------------------------------------------------------------------

def upscale_with_model(upscale_model, image, tile=512, overlap=32):
    """用放大模型跑分块放大。image 是 BHWC，返回同为 BHWC。"""
    import comfy.utils

    device = comfy.model_management.get_torch_device()
    upscale_model.to(device)
    in_img = image.movedim(-1, -3).to(device)

    oom = True
    while oom:
        try:
            steps = comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2],
                                                     tile_x=tile, tile_y=tile, overlap=overlap)
            pbar = comfy.utils.ProgressBar(steps)
            out = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a),
                                         tile_x=tile, tile_y=tile, overlap=overlap,
                                         upscale_amount=upscale_model.scale, pbar=pbar)
            oom = False
        except comfy.model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

    upscale_model.cpu()
    return torch.clamp(out.movedim(-3, -1), min=0, max=1.0)
