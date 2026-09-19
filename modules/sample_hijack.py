# This file is part of ReFocus.
# Original work Copyright (c) 2023 lllyasviel (Fooocus) & 2024 ehristoforu (DeFooocus).
# Modified and distributed under the terms of the GNU General Public License v3.0.

"""跨模型条件切换的工具 + 核心调度器表的补充注册。

迁移到现代核心后，这里原来的 `sample_hacked`（旧 `samplers.sample` 的整份拷贝，用来做
refiner 中途切换）已删除——它依赖的 `wrap_model` 在现代核心中不存在，而 refiner 切换
现在由 `process_diffusion` 的原生两遍采样完成。同理 `calculate_sigmas_scheduler_hacked`
也已删除，改为往核心的 `SCHEDULER_HANDLERS` 注册。

剩下的 `clip_separate*` 不是补丁，是纯工具：把文本条件按目标模型切分
（SDXLRefiner 取后 1280 维、SDXL 原样、SD1.5 取前 768 维并重跑 clip_l 的 final_layer_norm）。
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
    """把核心没有、但 UI 暴露的调度器注册进去。

    `turbo` 与 `lcm` 不在现代 `SCHEDULER_HANDLERS` 里，而 `KSampler.__init__` 对未知
    调度器会**静默降级为 `SCHEDULERS[0]`（simple）**——不报错，只是换了算法。
    所以必须显式注册。

    `lcm` 不在此处注册：`async_worker` 在 LCM 模式下会先把调度器改写成 `sgm_uniform`
    （核心已有），`lcm` 只作为 UI 选项存在。

    `SCHEDULER_NAMES` 是独立的一份 list，`KSampler.SCHEDULERS` 引用的正是它，
    因此必须原地 append 而不是重新赋值。
    """
    if 'turbo' not in comfy.samplers.SCHEDULER_HANDLERS:
        comfy.samplers.SCHEDULER_HANDLERS['turbo'] = comfy.samplers.SchedulerHandler(comfy_ops.turbo_sigmas)
        comfy.samplers.SCHEDULER_NAMES.append('turbo')
    return


register_extra_schedulers()
