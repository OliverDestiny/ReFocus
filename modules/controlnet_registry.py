# This file is part of ReFocus.
# Original work Copyright (c) 2023 lllyasviel (Fooocus) & 2024 ehristoforu (DeFooocus).
# Modified and distributed under the terms of the GNU General Public License v3.0.

"""Registry of ControlNet / Image Prompt types: a type is declared once, here.

A declaration carries its loader, preprocessor, apply_kind ('conditioning' via core.apply_controlnet
or 'unet_patch' via ip_adapter.patch_model), conditioning_source ('image' vs 'params') and param_spec;
see README_DEV.md, "ControlNet types declare their own parameters".
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import modules.flags as flags


@dataclass(frozen=True)
class ParamSpec:
    """One declaration of a type-owned parameter; the UI builds its widget, the worker reads it.
    kind selects the widget: 'slider' | 'checkbox' | 'dropdown'.
    """
    name: str
    label: str
    kind: str = 'slider'
    default: object = 0.0
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.001
    choices: Tuple = ()
    info: str = ''


@dataclass(frozen=True)
class PreprocessContext:
    """Inputs every type's preprocessor may need, passed in one object.
    Type-specific parameters are not here; they travel with the slot, declared by param_spec.
    """

    width: int
    height: int
    skipping_preprocessor: bool
    model_path: Optional[str] = None


@dataclass(frozen=True)
class ControlNetType:
    name: str
    apply_kind: str
    default_stop: float
    default_weight: float
    load_model: Callable[[], str]
    # (image, params, PreprocessContext) -> (slot_value, display_image): image is None for 'params'
    # types; slot_value is a tensor / (conds, unconds); display_image serves debugging_cn_preprocessor.
    preprocess: Callable
    feeds_controlnet_pipeline: bool = False
    conditioning_source: str = 'image'
    param_spec: Tuple[ParamSpec, ...] = ()


@dataclass
class ControlNetSlot:
    """Runtime state of one reference-image slot: image holds the type's slot_value after
    preprocessing (a numpy array before it); start/stop are the sampling range (0~1) as in A1111.
    """

    image: object
    start: float
    stop: float
    weight: float
    type: str
    params: dict = None


# --- PyraCanny ---

def _load_canny() -> str:
    import modules.deps_models_download as downloader
    return str(Path(downloader.downloading_controlnet_canny()).resolve())


def _preprocess_canny(image, params, ctx: PreprocessContext):
    import modules.core as core
    import extras.preprocessors as preprocessors
    from modules.util import HWC3, resize_image

    img = resize_image(HWC3(image), width=ctx.width, height=ctx.height)
    if not ctx.skipping_preprocessor:
        img = preprocessors.canny_pyramid(img, params['canny_low_threshold'],
                                          params['canny_high_threshold'])
    img = HWC3(img)
    return core.numpy_to_pytorch(img), img


# --- CPDS ---

def _load_cpds() -> str:
    import modules.deps_models_download as downloader
    return str(Path(downloader.downloading_controlnet_cpds()).resolve())


def _preprocess_cpds(image, params, ctx: PreprocessContext):
    import modules.core as core
    import extras.preprocessors as preprocessors
    from modules.util import HWC3, resize_image

    img = resize_image(HWC3(image), width=ctx.width, height=ctx.height)
    if not ctx.skipping_preprocessor:
        img = preprocessors.cpds(img)
    img = HWC3(img)
    return core.numpy_to_pytorch(img), img


# --- ImagePrompt / FaceSwap ---

# https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
_IP_IMAGE_SIZE = 224


def _load_ip() -> str:
    import extras.ip_adapter as ip_adapter
    import modules.deps_models_download as downloader

    clip_vision_path, ip_negative_path, ip_adapter_path = downloader.downloading_ip_adapters('ip')
    ip_adapter_path = str(Path(ip_adapter_path).resolve())
    ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)
    return ip_adapter_path


def _preprocess_ip(image, params, ctx: PreprocessContext):
    import extras.ip_adapter as ip_adapter
    from modules.util import HWC3, resize_image

    img = HWC3(image)
    img = resize_image(img, width=_IP_IMAGE_SIZE, height=_IP_IMAGE_SIZE, resize_mode=0)
    return ip_adapter.preprocess(img, ip_adapter_path=ctx.model_path), img


def _load_ip_face() -> str:
    import extras.ip_adapter as ip_adapter
    import modules.deps_models_download as downloader

    clip_vision_path, ip_negative_path, ip_adapter_face_path = downloader.downloading_ip_adapters('face')
    ip_adapter_face_path = str(Path(ip_adapter_face_path).resolve())
    ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_face_path)
    return ip_adapter_face_path


def _preprocess_ip_face(image, params, ctx: PreprocessContext):
    import extras.face_crop
    import extras.ip_adapter as ip_adapter
    from modules.util import HWC3, resize_image

    img = HWC3(image)
    if not ctx.skipping_preprocessor:
        img = extras.face_crop.crop_image(img, threshold=params['face_detection_threshold'])
    img = resize_image(img, width=_IP_IMAGE_SIZE, height=_IP_IMAGE_SIZE, resize_mode=0)
    return ip_adapter.preprocess(img, ip_adapter_path=ctx.model_path), img


# --- Registry data ---
# Order matches the type dropdown in the UI.

CANNY_PARAMS: Tuple[ParamSpec, ...] = (
    ParamSpec('canny_low_threshold', 'Canny Low Threshold', 'slider', 64.0, 1.0, 255.0, 1.0),
    ParamSpec('canny_high_threshold', 'Canny High Threshold', 'slider', 128.0, 1.0, 255.0, 1.0),
)

FACE_PARAMS: Tuple[ParamSpec, ...] = (
    ParamSpec('face_detection_threshold', 'Face Detection Confidence Threshold', 'slider',
              0.5, 0.1, 0.97, 0.01,
              info='facexlib defaults to 0.97, which suits photographs; anime faces are missed often '
                   'and some compositions are never detected at all.'),
)

TYPES: Tuple[ControlNetType, ...] = (
    ControlNetType(flags.cn_ip, 'unet_patch', 0.5, 0.6, _load_ip, _preprocess_ip),
    ControlNetType(flags.cn_canny, 'conditioning', 0.5, 1.0, _load_canny, _preprocess_canny,
                   feeds_controlnet_pipeline=True, param_spec=CANNY_PARAMS),
    ControlNetType(flags.cn_cpds, 'conditioning', 0.5, 1.0, _load_cpds, _preprocess_cpds,
                   feeds_controlnet_pipeline=True),
    ControlNetType(flags.cn_ip_face, 'unet_patch', 0.9, 0.75, _load_ip_face, _preprocess_ip_face,
                   param_spec=FACE_PARAMS),
)

BY_NAME = {t.name: t for t in TYPES}

DEFAULT_TYPE = flags.cn_ip


def names() -> Tuple[str, ...]:
    """All type names, in the same order as the UI dropdown."""
    return tuple(t.name for t in TYPES)


def new_task_map() -> dict:
    """Per-type bucket of slots, replacing the old {x: [] for x in flags.ip_list}."""
    return {t.name: [] for t in TYPES}


def get(name: str) -> ControlNetType:
    return BY_NAME[name]


def param_union() -> Tuple[ParamSpec, ...]:
    """Union of every type's parameter declarations, in fixed order for UI widgets and worker lookups."""
    out = []
    seen = set()
    for t in TYPES:
        for spec in t.param_spec:
            if spec.name not in seen:
                seen.add(spec.name)
                out.append(spec)
    return tuple(out)


def param_defaults(name: str) -> dict:
    """Parameter defaults of one type."""
    return {spec.name: spec.default for spec in BY_NAME[name].param_spec}


def param_names(name: str) -> Tuple[str, ...]:
    return tuple(spec.name for spec in BY_NAME[name].param_spec)


def default_parameters(name: str) -> Tuple[float, float]:
    """(stop, weight) of one type, replacing the old flags.default_parameters dict."""
    t = BY_NAME[name]
    return t.default_stop, t.default_weight


def conditioning_types() -> Tuple[ControlNetType, ...]:
    return tuple(t for t in TYPES if t.apply_kind == 'conditioning')


def unet_patch_types() -> Tuple[ControlNetType, ...]:
    return tuple(t for t in TYPES if t.apply_kind == 'unet_patch')
