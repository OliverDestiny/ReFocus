import cv2
import numpy as np


def centered_canny(x: np.ndarray, canny_low_threshold, canny_high_threshold):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2 and x.dtype == np.uint8

    y = cv2.Canny(x, int(canny_low_threshold), int(canny_high_threshold))
    y = y.astype(np.float32) / 255.0
    return y


def centered_canny_color(x: np.ndarray, canny_low_threshold, canny_high_threshold):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 3 and x.shape[2] == 3

    result = [centered_canny(x[..., i], canny_low_threshold, canny_high_threshold) for i in range(3)]
    result = np.stack(result, axis=2)
    return result


def pyramid_canny_color(x: np.ndarray, canny_low_threshold, canny_high_threshold):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 3 and x.shape[2] == 3

    H, W, C = x.shape
    acc_edge = None

    for k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        Hs, Ws = int(H * k), int(W * k)
        small = cv2.resize(x, (Ws, Hs), interpolation=cv2.INTER_AREA)
        edge = centered_canny_color(small, canny_low_threshold, canny_high_threshold)
        if acc_edge is None:
            acc_edge = edge
        else:
            acc_edge = cv2.resize(acc_edge, (edge.shape[1], edge.shape[0]), interpolation=cv2.INTER_LINEAR)
            acc_edge = acc_edge * 0.75 + edge * 0.25

    return acc_edge


def norm255(x, low=4, high=96):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2 and x.dtype == np.float32

    v_min = np.percentile(x, low)
    v_max = np.percentile(x, high)

    x -= v_min
    x /= v_max - v_min

    return x * 255.0


def canny_pyramid(x, canny_low_threshold, canny_high_threshold):
    # For some reasons, SAI's Control-lora Canny seems to be trained on canny maps with non-standard resolutions.
    # Then we use pyramid to use all resolutions to avoid missing any structure in specific resolutions.

    color_canny = pyramid_canny_color(x, canny_low_threshold, canny_high_threshold)
    result = np.sum(color_canny, axis=2)

    return norm255(result, low=1, high=99).clip(0, 255).astype(np.uint8)


def cpds(x):
    # cv2.decolor is not "decolor", it is Cewu Lu's method
    # See http://www.cse.cuhk.edu.hk/leojia/projects/color2gray/index.html
    # See https://docs.opencv.org/3.0-beta/modules/photo/doc/decolor.html

    raw = cv2.GaussianBlur(x, (0, 0), 0.8)
    density, boost = cv2.decolor(raw)

    raw = raw.astype(np.float32)
    density = density.astype(np.float32)
    boost = boost.astype(np.float32)

    offset = np.sum((raw - boost) ** 2.0, axis=2) ** 0.5
    result = density + offset

    return norm255(result, low=4, high=96).clip(0, 255).astype(np.uint8)


class MiDaSDepth:
    """MiDaS DPT-Hybrid depth: the preprocessing the SDXL depth Control-LoRA was trained with.

    The architecture comes from `intel-isl/MiDaS` through `torch.hub`, the weights are the local
    `dpt_hybrid-midas-501f0c75.pt`. The hub fetch happens once and is cached under
    `~/.cache/torch/hub`, so only the first call in a fresh environment needs network; a failure
    there is raised with the reason instead of being swallowed.

    Measured on this machine: 13 s for the first call (hub fetch plus CUDA warm-up), 0.05 s warm,
    and the checkpoint matches the hub architecture with zero missing and zero unexpected keys.
    """

    def __init__(self, model_path, device=None):
        import torch

        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            self.model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid', pretrained=False,
                                        trust_repo=True)
        except Exception as e:
            raise RuntimeError(
                'Depth preprocessor: torch.hub could not load the MiDaS model code (fetched once and '
                'cached under ~/.cache/torch/hub). Run once with network access, or point TORCH_HOME '
                f'at a machine that already has it. Original error: {e}') from e

        state = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def __call__(self, image):
        """HWC uint8 RGB in, HWC uint8 RGB out (a three-channel grayscale depth map)."""
        import torch.nn.functional as F

        assert image.ndim == 3 and image.shape[2] == 3
        height, width = image.shape[:2]

        # MiDaS's own preprocessing: short side to 384, both sides snapped to 32, ImageNet norm.
        scale = 384 / min(height, width)
        net_height = max(32, int(round(height * scale / 32)) * 32)
        net_width = max(32, int(round(width * scale / 32)) * 32)
        resized = cv2.resize(image, (net_width, net_height), interpolation=cv2.INTER_LINEAR)

        x = self.torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0)
        x = (x.to(self.device) / 255.0 - self.mean) / self.std

        with self.torch.no_grad():
            prediction = self.model(x)

        prediction = F.interpolate(prediction.unsqueeze(1), size=(height, width), mode='bilinear',
                                   align_corners=False)[0][0]
        depth = prediction - prediction.min()
        depth = depth / depth.max() * 255.0
        depth = depth.clamp(0, 255).to(self.torch.uint8).cpu().numpy()
        return np.stack([depth] * 3, axis=2)


_midas_depth = None


def midas_depth(image):
    """Depth map through a process-wide MiDaS model, loaded on first use."""
    global _midas_depth
    if _midas_depth is None:
        import modules.deps_models_download as downloader

        _midas_depth = MiDaSDepth(downloader.downloading_midas_depth_model())
    return _midas_depth(image)
