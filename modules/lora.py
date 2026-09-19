# This file is part of ReFocus.
# Original work Copyright (c) 2023 lllyasviel (Fooocus) & 2024 ehristoforu (DeFooocus).
# Modified and distributed under the terms of the GNU General Public License v3.0.

"""LoRA / 补丁文件的解析。

迁移到现代核心后不再自己实现格式解析：标准格式（lora / lokr / loha / glora / dora）
交给 `comfy.lora.load_lora`，它返回现代 `calculate_weight` 认识的
`WeightAdapterBase` 实例或 `("diff", (tensor,))` 补丁。

这里只补一件核心不做的事：**Fooocus 的原始权重补丁**（`inpaint*.fooocus.patch`）。
那种文件里的 key 与模型权重 key 完全同名，值是 `(w1, w_min, w_max)` 三元组：

- 权重的 `w1` 是 uint8 量化值（0~255），`w_min`/`w_max` 是每行的真实范围；
- bias 的 `w1` 已是 float32 真实值，此时 `w_min=0`、`w_max=255`，
  同一条反量化公式会退化成恒等变换。

所以统一用 `(w1 / 255) * (w_max - w_min) + w_min` 即可，两种情形都对
（已对 inpaint_v26.fooocus.patch 实测确认）。
"""

import torch

import comfy.lora


def dequantize_raw_patch(value):
    """把 Fooocus 原始权重补丁 (w1, w_min, w_max) 还原成真实权重。"""
    w1, w_min, w_max = value
    return (w1.float() / 255.0) * (w_max - w_min) + w_min


def is_raw_patch(value):
    return isinstance(value, tuple) and len(value) == 3


def load_lora_patches(lora, to_load):
    """解析出补丁字典，供 ModelPatcher.add_patches 使用。

    lora   : 从补丁文件读出的 {key: tensor | (w1, w_min, w_max)}
    to_load: {文件里的 key: 模型的 state_dict key}，即 comfy.lora.model_lora_keys_*
    """
    patch_dict = {}

    # Fooocus 原始权重补丁：值是三元组，直接反量化成 diff
    for key, value in lora.items():
        if is_raw_patch(value):
            patch_dict[to_load.get(key, key)] = ("diff", (dequantize_raw_patch(value),))

    # 其余交给核心；log_missing=False 是因为「不匹配当前模型」由调用方判断，
    # 而 CLIP 侧的解析必然会看到一批未匹配的 UNet key，日志会很吵。
    standard = comfy.lora.load_lora(lora, to_load, log_missing=False)
    patch_dict.update(standard)

    return patch_dict
