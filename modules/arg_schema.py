# This file is part of ReFocus.
# Original work Copyright (c) 2023 lllyasviel (Fooocus) & 2024 ehristoforu (DeFooocus).
# Modified and distributed under the terms of the GNU General Public License v3.0.

"""Single source of truth for the argument contract: order and count only.

UI (webui.py builds ctrls) and worker (async_worker.py parses args) both follow this table, so an
unsynchronized change fails at startup instead of silently shifting every later argument.
Type conversion stays at the consumption site.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import modules.flags as flags


@dataclass(frozen=True)
class ArgGroup:
    name: str
    args: Tuple[str, ...]
    enabled: Optional[Callable[[], bool]] = None


def _lora_args() -> Tuple[str, ...]:
    out = []
    for i in range(flags.lora_count):
        out.append(f'lora_model_{i + 1}')
        out.append(f'lora_weight_{i + 1}')
    return tuple(out)


def _image_prompt_args() -> Tuple[str, ...]:
    # 6 entries per slot, in UI order: Image, Start At, Stop At, Weight, Type, Params.
    # Type parameters ride in cn_params_{i} as JSON; docs/README_DEV.md, ControlNet types section.
    out = []
    for i in range(flags.controlnet_image_count):
        out.append(f'cn_image_{i + 1}')
        out.append(f'cn_start_{i + 1}')
        out.append(f'cn_stop_{i + 1}')
        out.append(f'cn_weight_{i + 1}')
        out.append(f'cn_type_{i + 1}')
        out.append(f'cn_params_{i + 1}')
    return tuple(out)


def _metadata_enabled() -> bool:
    # Deferred import: avoids an import-order problem at module load time
    import modules.args_manager as args_manager
    return not args_manager.args.disable_metadata


GROUPS = (
    ArgGroup('generation', (
        'prompt', 'negative_prompt', 'translate_prompts', 'steps',
        'width', 'height', 'image_number', 'output_format', 'image_seed',
        'sharpness', 'guidance_scale',
    )),
    ArgGroup('models', (
        'base_model_name', 'refiner_model_name', 'refiner_switch',
    ) + _lora_args()),
    ArgGroup('input_image_mode', ('input_image_checkbox', 'current_tab')),
    ArgGroup('uov', (
        'uov_mode', 'uov_vary_mode', 'uov_scale', 'uov_fast',
        'uov_ignore_prompt', 'uov_denoise', 'uov_input_image',
    )),
    ArgGroup('inpaint_input', (
        'outpaint_selections', 'inpaint_input_image', 'inpaint_additional_prompt',
    )),
    ArgGroup('output_flags', (
        'disable_preview', 'disable_intermediate_results', 'black_out_nsfw',
    )),
    ArgGroup('adm', (
        'adm_scaler_positive', 'adm_scaler_negative', 'adm_scaler_end', 'adaptive_cfg',
    )),
    ArgGroup('sampler', ('sampler_name', 'scheduler_name')),
    ArgGroup('mixing', (
        'mixing_image_prompt_and_vary_upscale',
        'mixing_image_prompt_and_inpaint',
    )),
    ArgGroup('controlnet_debug', (
        'debugging_cn_preprocessor', 'skipping_cn_preprocessor',
    )),
    ArgGroup('refiner', ('refiner_swap_method', 'controlnet_softness')),
    ArgGroup('freeu', ('freeu_enabled', 'freeu_b1', 'freeu_b2', 'freeu_s1', 'freeu_s2')),
    ArgGroup('inpaint', (
        'debugging_inpaint_preprocessor', 'inpaint_disable_initial_latent', 'inpaint_engine',
        'inpaint_strength', 'inpaint_respective_field',
        'invert_mask_checkbox', 'inpaint_erode_or_dilate',
    )),
    ArgGroup('metadata', ('save_metadata_to_images', 'metadata_scheme'), enabled=_metadata_enabled),
    ArgGroup('image_prompt', _image_prompt_args()),
)


def active_groups() -> Tuple[ArgGroup, ...]:
    return tuple(g for g in GROUPS if g.enabled is None or g.enabled())


def active_args() -> Tuple[str, ...]:
    return tuple(name for group in active_groups() for name in group.args)


def total() -> int:
    """Number of arguments the worker consumes under the current config (currentTask excluded)."""
    return len(active_args())


def group(group_name: str) -> ArgGroup:
    for g in GROUPS:
        if g.name == group_name:
            return g
    raise KeyError(f'Unknown argument group "{group_name}".')


def group_of(name: str) -> Optional[ArgGroup]:
    for g in GROUPS:
        if name in g.args:
            return g
    return None
