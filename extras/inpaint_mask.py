import torch
from rembg import remove, new_session


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_mask_from_image(image, mask_model):
    """
    generate mask
    """
    if image is None:
        return

    if 'image' in image:
        image = image['image']

    from modules.deps_models_download import ensure_rembg_models
    ensure_rembg_models()

    return remove(
        image,
        session=new_session(mask_model),
        only_mask=True
    )
