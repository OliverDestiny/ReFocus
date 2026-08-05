import os
import modules.flags
from modules.flags import LCM_LORA_FILENAME
from modules.model_loader import load_file_from_url
from modules.config import (
    path_inpaint,
    path_loras,
    path_controlnet,
    path_clip_vision,
    path_upscale_models,
    path_vae_approx,
)

def _download_with_fallback(mirror_url: str, official_url: str | None, model_dir: str, file_name: str) -> str:
    """
    ensure file exists
    download from mirror first, if failed download from official source
    """
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if os.path.exists(cached_file):
        return cached_file

    try:
        load_file_from_url(mirror_url, model_dir=model_dir, file_name=file_name)
        print(f'[ReFocus] Downloaded {file_name} from mirror.')
        return cached_file
    except Exception as e:
        print(f'[ReFocus] Mirror download failed for {file_name}: {e}')
        if official_url is None:
            print(f'[ReFocus] No official fallback provided. File may be missing, relying on library internal download.')
            return cached_file
        else:
            print(f'[ReFocus] Falling back to official source...')
            load_file_from_url(official_url, model_dir=model_dir, file_name=file_name)
            print(f'[ReFocus] Downloaded {file_name} from official source.')
            return cached_file


def downloading_inpaint_models(v):
    assert v in modules.flags.inpaint_engine_versions

    load_file_from_url(
        url='https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/fooocus_inpaint_head.pth',
        model_dir=path_inpaint,
        file_name='fooocus_inpaint_head.pth'
    )
    head_file = os.path.join(path_inpaint, 'fooocus_inpaint_head.pth')
    patch_file = None

    if v == 'v1':
        load_file_from_url(
            url='https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/inpaint.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint.fooocus.patch')

    if v == 'v2.5':
        load_file_from_url(
            url='https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/inpaint_v25.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v25.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v25.fooocus.patch')

    if v == 'v2.6':
        load_file_from_url(
            url='https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/inpaint_v26.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v26.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v26.fooocus.patch')

    return head_file, patch_file


def downloading_sdxl_lcm_lora():
    load_file_from_url(
        url='https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/' + LCM_LORA_FILENAME,
        model_dir=path_loras,
        file_name=LCM_LORA_FILENAME
    )
    return LCM_LORA_FILENAME


def downloading_controlnet_canny():
    load_file_from_url(
        url='https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/control-lora-canny-rank128.safetensors',
        model_dir=path_controlnet,
        file_name='control-lora-canny-rank128.safetensors'
    )
    return os.path.join(path_controlnet, 'control-lora-canny-rank128.safetensors')


def downloading_controlnet_cpds():
    load_file_from_url(
        url='https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/fooocus_xl_cpds_128.safetensors',
        model_dir=path_controlnet,
        file_name='fooocus_xl_cpds_128.safetensors'
    )
    return os.path.join(path_controlnet, 'fooocus_xl_cpds_128.safetensors')


def downloading_ip_adapters(v):
    assert v in ['ip', 'face']

    results = []

    load_file_from_url(
        url='https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/clip_vision_vit_h.safetensors',
        model_dir=path_clip_vision,
        file_name='clip_vision_vit_h.safetensors'
    )
    results += [os.path.join(path_clip_vision, 'clip_vision_vit_h.safetensors')]

    load_file_from_url(
        url='https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/fooocus_ip_negative.safetensors',
        model_dir=path_controlnet,
        file_name='fooocus_ip_negative.safetensors'
    )
    results += [os.path.join(path_controlnet, 'fooocus_ip_negative.safetensors')]

    if v == 'ip':
        load_file_from_url(
            url='https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
            model_dir=path_controlnet,
            file_name='ip-adapter-plus_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet, 'ip-adapter-plus_sdxl_vit-h.bin')]

    if v == 'face':
        load_file_from_url(
            url='https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin',
            model_dir=path_controlnet,
            file_name='ip-adapter-plus-face_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet, 'ip-adapter-plus-face_sdxl_vit-h.bin')]

    return results


def downloading_upscale_model():
    load_file_from_url(
        url='https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/fooocus_upscaler.bin',
        model_dir=path_upscale_models,
        file_name='fooocus_upscaler.bin'
    )
    return os.path.join(path_upscale_models, 'fooocus_upscaler.bin')


def ensure_facexlib_models():
    """ensure facexlib needed parsing_parsenet.pth & detection_Resnet50_Final.pth exists"""
    PARSING_FILE = os.path.join(path_controlnet, 'parsing_parsenet.pth')
    DETECTION_FILE = os.path.join(path_controlnet, 'detection_Resnet50_Final.pth')

    mirror_base = 'https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/'
    for fname in ['parsing_parsenet.pth', 'detection_Resnet50_Final.pth']:
        file_path = os.path.join(path_controlnet, fname)
        if not os.path.exists(file_path):
            _download_with_fallback(
                mirror_url=mirror_base + fname,
                official_url=None,  # facexlib will automatically handle missing
                model_dir=path_controlnet,
                file_name=fname
            )


def ensure_rembg_models():
    """
    ensure rembg needed 4 models exists in ~/.u2net/
    first time download all
    """
    target_dir = os.path.expanduser("~/.u2net")
    os.makedirs(target_dir, exist_ok=True)

    from modules.flags import MASK_MODEL_CHOICES

    for fname in MASK_MODEL_CHOICES:
        file_path = os.path.join(target_dir, fname + '.onnx')
        if not os.path.exists(file_path):
            _download_with_fallback(
                mirror_url=f'https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/{fname}.onnx',
                official_url=None,  # if mirror fail then rembg download
                model_dir=target_dir,
                file_name=fname + '.onnx'
            )


def ensure_blip_caption_model():
    """ensure BLIP model exists"""
    _download_with_fallback(
        mirror_url='https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/model_base_caption_capfilt_large.pth',
        official_url='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth',
        model_dir=path_clip_vision,
        file_name='model_base_caption_capfilt_large.pth'
    )


def ensure_vae_interposer_model():
    """ensure VAE interposer model (xl-to-v1) exists"""
    _download_with_fallback(
        mirror_url='https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/xl-to-v1_interposer-v3.1.safetensors',
        official_url=None,  # 社区模型，无官方源
        model_dir=path_vae_approx,  # 注意：需要导入 path_vae_approx
        file_name='xl-to-v1_interposer-v3.1.safetensors'
    )


def ensure_wd14_tagger_models():
    """ensure WD14 tagger needed .onnx & .csv exists"""
    model_name = "wd-v1-4-moat-tagger-v2"
    for ext in ['.onnx', '.csv']:
        fname = model_name + ext
        _download_with_fallback(
            mirror_url=f'https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/{fname}',
            official_url=f'https://huggingface.co/SmilingWolf/{model_name}/resolve/main/{fname}',
            model_dir=path_clip_vision,
            file_name=fname
        )