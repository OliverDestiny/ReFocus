# This file is part of ReFocus.
# Original work Copyright (c) 2023 lllyasviel (Fooocus) & 2024 ehristoforu (DeFooocus).
# Modified and distributed under the terms of the GNU General Public License v3.0.

"""LoRA / patch-file parsing: standard formats via `comfy.lora.load_lora`, plus Fooocus raw patches.

`comfy.lora.load_lora` covers the standard formats (lora / lokr / loha / glora / dora) and returns
patches the modern `calculate_weight` understands; the extra work here is Fooocus's raw weight patches
(`inpaint*.fooocus.patch`), whose `(w1, w_min, w_max)` triples `dequantize_raw_patch` restores.
"""

import torch

import comfy.lora


def dequantize_raw_patch(value):
    """Restore a Fooocus raw weight patch (w1, w_min, w_max) to real weights."""
    w1, w_min, w_max = value
    return (w1.float() / 255.0) * (w_max - w_min) + w_min


def is_raw_patch(value):
    return isinstance(value, tuple) and len(value) == 3


def load_lora_patches(lora, to_load):
    """Build the patch dict for ModelPatcher.add_patches from the file's {key: tensor | triple} dict.

    to_load maps file keys to model state_dict keys (comfy.lora.model_lora_keys_*).
    """
    patch_dict = {}

    # Fooocus raw weight patches: a triple value, dequantized straight into a diff patch
    for key, value in lora.items():
        if is_raw_patch(value):
            patch_dict[to_load.get(key, key)] = ("diff", (dequantize_raw_patch(value),))

    # The rest goes to the core; log_missing=False because "does not match this model" is the caller's
    # call, and CLIP-side parsing always sees a batch of unmatched UNet keys, which would flood the log.
    standard = comfy.lora.load_lora(lora, to_load, log_missing=False)
    patch_dict.update(standard)

    return patch_dict
