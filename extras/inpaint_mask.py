from PIL import Image
import numpy as np
import torch
from rembg import remove, new_session
from extras.GroundingDINO.util.inference import default_groundingdino
from modules.model_loader import load_file_from_url
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# rembg default cache folder
U2NET_HOME = os.path.join(os.path.expanduser('~'), '.u2net')
U2NET_PATH = os.path.join(U2NET_HOME, 'u2net.onnx')


def ensure_u2net_downloaded():
    """
    ensure u2net.onnx exists in rembg cache folder
    download from backup repo first,
    fail then download automatically by rembg from github
    """
    if not os.path.exists(U2NET_PATH):
        print('[ReFocus] u2net.onnx not found in cache. Attempting mirror download...')
        try:
            os.makedirs(U2NET_HOME, exist_ok=True)
            load_file_from_url(
                url='https://huggingface.co/OliverBlack56864/ReFocus-deps/resolve/main/u2net.onnx',
                model_dir=U2NET_HOME,
                file_name='u2net.onnx'
            )
            print('[ReFocus] u2net.onnx downloaded from mirror successfully.')
        except Exception as e:
            print(f'[ReFocus] Mirror download failed for u2net.onnx: {e}')
            print('[ReFocus] Will rely on rembg to download from original source.')


def run_grounded_sam(input_image, text_prompt, box_threshold, text_threshold):

    # run grounding dino model
    boxes, _ = default_groundingdino(
        image=np.array(input_image),
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    return boxes.xyxy


def generate_mask_from_image(image, mask_model, extras):
    if image is None:
        return

    if 'image' in image:
        image = image['image']

    if mask_model == 'sam':
        boxes = run_grounded_sam(Image.fromarray(image), extras['sam_prompt_text'], box_threshold=extras['box_threshold'], text_threshold=extras['text_threshold'])
        boxes = np.array([[0, 0, image.shape[1], image.shape[0]]]) if len(boxes) == 0 else boxes
        extras['sam_prompt'] = []
        for idx, box in enumerate(boxes):
            extras['sam_prompt'] += [{"type": "rectangle", "data": box.tolist()}]

    ensure_u2net_downloaded()

    return remove(
        image,
        session=new_session(mask_model, **extras),
        only_mask=True,
        **extras
    )
