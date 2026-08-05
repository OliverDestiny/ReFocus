from rembg import remove, new_session
import gradio as gr
from PIL import Image

def rembg_run(path, model_name='u2net', progress=gr.Progress(track_tqdm=True)):
    from modules.deps_models_download import ensure_rembg_models
    ensure_rembg_models()

    input_img = Image.open(path)
    session = new_session(model_name)
    output = remove(input_img, session=session)
    return output