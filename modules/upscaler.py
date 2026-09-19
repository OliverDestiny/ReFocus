import os
import torch
import modules.core as core
import modules.comfy_ops as comfy_ops

from extras.esrgan import RRDBNet as ESRGAN
from collections import OrderedDict
from modules.config import path_upscale_models

model_filename = os.path.join(path_upscale_models, 'fooocus_upscaler.bin')
model = None


def perform_upscale(img):
    global model

    print(f'Upscaling image with shape {str(img.shape)} ...')

    if model is None:
        sd = torch.load(model_filename)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        model = ESRGAN(sdo)
        model.cpu()
        model.eval()

    img = core.numpy_to_pytorch(img)
    img = comfy_ops.upscale_with_model(model, img)
    img = core.pytorch_to_numpy(img)[0]

    return img
