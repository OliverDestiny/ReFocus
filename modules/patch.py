# This file is part of ReFocus.
# Original work Copyright (c) 2023 lllyasviel (Fooocus) & 2024 ehristoforu (DeFooocus).
# Modified and distributed under the terms of the GNU General Public License v3.0.

"""ReFocus 对推理核心的注入层。

**这个文件是 ReFocus 与上游 ComfyUI 的全部耦合面，越小越好。**

原则（见 .zcode/phase-d-migration-plan.md）：
- 能用核心的扩展接口就不要替换核心内部。`patcher_extension` 的 `WrapperExecutor`
  与 `model_options` 系列钩子是上游冻结的接口（`patcher_extension.py` 12 个月只改了 5 行），
  改走这些接口的补丁不会随上游 churn 而失效。
- 只保留上游确实没有的能力。迁移到现代核心后，原来 9 个补丁里 6 个已被上游原生实现，
  直接删除而不是移植：

  | 原补丁 | 处置 |
  |---|---|
  | ModelPatcher.calculate_weight | 删：现代走 comfy.lora + weight_adapter |
  | UNetModel.forward | 删：现代原生支持 input/output_block_patch |
  | KSamplerX0Inpaint.forward | 删：现代 __call__ 就是同一套混合，且支持 denoise_mask_function |
  | BrownianTreeNoiseSampler | 删：现代 sampler 自己构建同样的噪声场 |
  | samplers.sampling_function | 删：改挂 model_options["sampler_cfg_function"] |
  | SDXL.encode_adm | **保留**：现代无对应，是 Fooocus ADM 引导的核心 |
  | ControlNet.forward | 收敛为薄包装：只做 timed_adm 与 softness，其余交给上游（含 union） |
  | load_models_gpu 计时 | 保留（仅计时） |
  | build_loaded 权重纠错 | 保留（与核心无关） |

  `modules/patch_precision.py` 整文件删除：上游已收敛到同样数值，它早已是空操作。
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
        # VAE refiner swap 需要把采样过程中的 eps 记下来；None 表示不记录。
        self.eps_record = None


patch_settings = {}


def settings():
    """取当前进程的 PatchSettings。由 async_worker 在每个任务开始时写入。"""
    return patch_settings[os.getpid()]


# ---------------------------------------------------------------------------
# ADM 引导（上游无对应，必须保留）
# ---------------------------------------------------------------------------

def round_to_64(x):
    h = float(x)
    h = h / 64.0
    h = round(h)
    h = int(h)
    h = h * 64
    return h


def timed_adm(y, timesteps):
    """按 adm_scaler_end 让 ADM 在采样后段淡出。

    现代 UNet 不做这件事，所以它同时用在 UNet 与 ControlNet 两条 y 上。
    """
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
# CFG 钩子：锐度 + adaptive CFG（TSNR 仿制）+ eps_record
#
# 原来这是对 comfy.samplers.sampling_function 的整函数替换。现在改挂
# model_options["sampler_cfg_function"]——上游自己的节点也这么做，属于官方扩展点。
# ---------------------------------------------------------------------------

def diffusion_progress(model, sigma):
    """把当前 sigma 换算成 0~1 的扩散进度。

    旧实现由被替换掉的 UNet.forward 写入 global_diffusion_progress（1 - timestep/999）。
    现在没有那个钩子了，改用 model_sampling.timestep(sigma) 得到同一个离散 timestep。

    注意这里拿到的是 BaseModel（不是 ModelPatcher，`get_model_object` 在后者身上），
    所以读的是模型自带的 model_sampling，而不是被 add_object_patch 替换过的那个。
    只有 LCM 模式会替换 model_sampling，而 LCM 下 sharpness 被强制为 0、cfg 为 1，
    本函数的结果在那种配置下不会被 consume（alpha 恒为 0，且 compute_cfg 的 t 不参与）。
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
    """comfy 的 sampler_cfg_function 接口。

    上游 comfy/samplers.py 的 cfg_function 传入的 cond/uncond 已经是 eps（x - denoised），
    并要求返回最终 eps，因为 cfg_result = x - 返回值。

    CFG≈1 时上游会跳过 uncond 分支（除非 model_options 里 disable_cfg1_optimization），
    此时 args["uncond"] 来自零张量分支、等于 x，不可使用，所以此处单独处理。
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
    """把 CFG 钩子与 UNet 包装器挂到模型上。base 与 refiner 两个 unet 都要挂。

    为什么需要 model_function_wrapper：旧实现删掉的 `patched_unet_forward` 里有一步
    `y = timed_adm(y, timesteps)`，把 Fooocus 的双份 ADM（5632 维）在进入 label_emb 前
    压回 2816。删掉那个 forward 之后没人做这件事，于是 5632 直接喂给期望 2816 的
    label_emb，报 "mat1 and mat2 shapes cannot be multiplied (2x5632 and 2816x1280)"。
    现在用核心冻结的 model_function_wrapper 挂在同一位置。
    """
    model.set_model_sampler_cfg_function(custom_sampler_cfg_function)

    def unet_wrapper(apply_model, args):
        c = args["c"]
        if c.get("y") is not None:
            # args["timestep"] 是 sigma（k-diffusion 把 sigma 一路传到 predict_noise，
            # 见 comfy/samplers.py 里 transformer_options["sigmas"] = timestep），
            # 而 timed_adm 的阈值是 0~999 的离散 timestep，必须先换算，
            # 否则比较恒为假、遮罩恒为 0，ADM 缩放就永远不生效。
            # 用 model_sampling 走 get_model_object，以便取到 LCM 替换过的那份。
            t = model.get_model_object("model_sampling").timestep(args["timestep"])
            c["y"] = timed_adm(c["y"], t)
        return apply_model(args["input"], args["timestep"], **c)

    model.set_model_unet_function_wrapper(unet_wrapper)
    return model


# ---------------------------------------------------------------------------
# ControlNet：薄包装，而不是重写整个 forward
#
# 现代 cldm.ControlNet.forward 自己处理 union controlnet，返回
# {"middle": [...], "output": [...]}（controlnet.py 的 control_merge 按 dict 迭代）。
# 我们只做两件上游没有的事：入口的 timed_adm、出口的 softness 缩放。
# 这样上游对 forward 的改动都能直接继承。
# ---------------------------------------------------------------------------

_cldm_forward_origin = None


def patched_cldm_forward(self, x, hint, timesteps, context, y=None, **kwargs):
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
# 模型管理计时与权重加载纠错（与核心行为无关，保留）
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
# 安装
# ---------------------------------------------------------------------------

def _install(target, name, replacement):
    """替换属性并立即断言确实生效。

    这是防「静默失效」的闸：上游改了内部结构时，补丁可能装上但不再被调用，
    或者干脆装不上。这里让它在启动时就报错，而不是悄悄失去效果。
    """
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

    # 只捕获一次原始 forward，重复调用 patch_all() 不会把它覆盖成补丁本身
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
