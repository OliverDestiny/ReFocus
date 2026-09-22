# This file is part of ReFocus.
# Original work Copyright (c) 2023 lllyasviel (Fooocus) & 2024 ehristoforu (DeFooocus).
# Modified and distributed under the terms of the GNU General Public License v3.0.

"""ControlNet / Image Prompt 类型的注册表。

之前新增一个类型要在 flags 的常量与默认值、deps_models_download 的下载函数、
async_worker 的「下载/加载」「预处理」「应用」三段分支里各改一遍，漏掉任何一处都是静默失效。
现在只需在这里加一条声明。

两种应用方式（apply_kind）：
  - 'conditioning'：预处理结果经 core.apply_controlnet 挂到 conditioning 上（PyraCanny / CPDS）
  - 'unet_patch' ：预处理结果是图像条件，经 ip_adapter.patch_model 挂到 UNet 上（ImagePrompt / FaceSwap）

条件来源（conditioning_source）与类型参数（param_spec）：
  - 'image' ：槽位提供一个参考图（现有的四个类型都是这一类）
  - 'params'：槽位不提供图，改由参数直接描述条件（例：相机方位/仰角）
  两类都可以另外声明 param_spec —— 那是该类型自己的参数（如 Canny 的高低阈值），
  UI 按声明生成控件、worker 按声明取值，**新增类型不必再改契约的数量**
  （每槽的参数打包成一个 JSON 传给 worker，见 arg_schema 的 cn_params_{i}）。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import modules.flags as flags


@dataclass(frozen=True)
class ParamSpec:
    """类型自有参数的一条声明。UI 依此生成控件，worker 依此取值。

    kind 决定控件：'slider' | 'checkbox' | 'dropdown'。
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
    """预处理阶段各类型都可能用到的输入，集中传递。

    类型**自己的**参数不在这里 —— 它们由 param_spec 声明、随槽位传进来。
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
    # (image, params, PreprocessContext) -> (slot_value, display_image)
    # image 对 conditioning_source == 'params' 的类型是 None；params 是 param_spec 的取值字典。
    # slot_value 是最终写回槽位的东西：conditioning 类是 tensor，unet_patch 类是 (conds, unconds)。
    # display_image 供 debugging_cn_preprocessor 展示。
    preprocess: Callable
    feeds_controlnet_pipeline: bool = False
    conditioning_source: str = 'image'
    param_spec: Tuple[ParamSpec, ...] = ()


@dataclass
class ControlNetSlot:
    """一个参考图槽位的运行时状态。

    原来这里是 3 元素列表 [image, stop, weight]，改成数据类是为了后续逐槽加
    softness / mask 时不必再动解包处。
    image 在预处理前是 numpy 图，预处理后被替换成该类型的 slot_value。
    start/stop 是采样进度区间（0~1），作用与 A1111 的 "Starting/Ending Step" 一致。
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


# --- 注册表本体 ---
# 顺序与 UI 上的类型下拉一致，便于对照。

CANNY_PARAMS: Tuple[ParamSpec, ...] = (
    ParamSpec('canny_low_threshold', 'Canny Low Threshold', 'slider', 64.0, 1.0, 255.0, 1.0),
    ParamSpec('canny_high_threshold', 'Canny High Threshold', 'slider', 128.0, 1.0, 255.0, 1.0),
)

FACE_PARAMS: Tuple[ParamSpec, ...] = (
    ParamSpec('face_detection_threshold', 'Face Detection Confidence Threshold', 'slider',
              0.5, 0.1, 0.97, 0.01,
              info='facexlib 内部默认 0.97：真人照片够用，动漫风格命中率低且部分构图检不出。'),
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
    """所有类型名，顺序与 UI 下拉一致。"""
    return tuple(t.name for t in TYPES)


def new_task_map() -> dict:
    """按类型分桶的槽位容器，等价于原来的 {x: [] for x in flags.ip_list}。"""
    return {t.name: [] for t in TYPES}


def get(name: str) -> ControlNetType:
    return BY_NAME[name]


def param_union() -> Tuple[ParamSpec, ...]:
    """所有类型参数声明的并集，顺序固定 —— UI 按它建控件，worker 按它取值。"""
    out = []
    seen = set()
    for t in TYPES:
        for spec in t.param_spec:
            if spec.name not in seen:
                seen.add(spec.name)
                out.append(spec)
    return tuple(out)


def param_defaults(name: str) -> dict:
    """某类型的参数默认值。"""
    return {spec.name: spec.default for spec in BY_NAME[name].param_spec}


def param_names(name: str) -> Tuple[str, ...]:
    return tuple(spec.name for spec in BY_NAME[name].param_spec)


def default_parameters(name: str) -> Tuple[float, float]:
    """(stop, weight) —— 替代原来散在 flags.default_parameters 的字典。"""
    t = BY_NAME[name]
    return t.default_stop, t.default_weight


def conditioning_types() -> Tuple[ControlNetType, ...]:
    return tuple(t for t in TYPES if t.apply_kind == 'conditioning')


def unet_patch_types() -> Tuple[ControlNetType, ...]:
    return tuple(t for t in TYPES if t.apply_kind == 'unet_patch')
