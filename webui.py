import gradio as gr
import random
import os
import json
import time
import shared
import modules.config
import ReFocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.meta_parser
import args_manager
import numpy as np

from modules.private_logger import get_current_html_path
from modules.localization import localization_js
from modules.util import is_json

# ========== 读取自定义 CSS/JS 并注入 ==========
def get_custom_head():
    css_path = os.path.join(os.path.dirname(__file__), "css", "style.css")
    js_path = os.path.join(os.path.dirname(__file__), "javascript", "script.js")
    head = ""

    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            head += f"<style>{f.read()}</style>"

    # localization JS
    head += f"<script>{localization_js(args_manager.args.language)}</script>"

    if os.path.exists(js_path):
        with open(js_path, "r", encoding="utf-8") as f:
            head += f"<script>{f.read()}</script>"

    # 其他必需的 JS 文件（可按需添加，此处仅作示例）
    for js_file in ["contextMenus.js", "zoom.js", "edit-attention.js", "viewer.js", "imageviewer.js"]:
        js_file_path = os.path.join(os.path.dirname(__file__), "javascript", js_file)
        if os.path.exists(js_file_path):
            with open(js_file_path, "r", encoding="utf-8") as f:
                head += f"<script>{f.read()}</script>"

    if args_manager.args.theme:
        head += f'<script>set_theme("{args_manager.args.theme}");</script>'

    return head

# ========== 常量 ==========
PROMPT_HELPER_PORT = 17860
PHOTOPEA_MAIN_URL = "https://www.photopea.com/"
PHOTOPEA_IFRAME_ID = "webui-photopea-iframe"
PHOTOPEA_IFRAME_HEIGHT = 684
PHOTOPEA_IFRAME_WIDTH = "100%"
PHOTOPEA_IFRAME_LOADED_EVENT = "onPhotopeaLoaded"

def get_photopea_url_params():
    return "#%7B%22resources%22:%5B%22data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAIAAQMAAADOtka5AAAAAXNSR0IB2cksfwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAANQTFRF////p8QbyAAAADZJREFUeJztwQEBAAAAgiD/r25IQAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAfBuCAAAB0niJ8AAAAABJRU5ErkJggg==%22%5D%7D"

def get_task(*args):
    args = list(args)
    if not args:
        print("[Warning] get_task() received empty args, auto skip this task.")
        return None
    args.pop(0)
    if not args:
        print("[warning] get_task() get empty after pop(0), auto skip this task.")
        return None
    return worker.AsyncTask(args=args)

def generate_clicked(task):
    if task is None:
        print("[Warning] generate_clicked received an empty task, auto skip this task.")
        yield gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=[])
        return

    import ldm_patched.modules.model_management as model_management
    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False

    execution_start_time = time.perf_counter()
    finished = False

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Waiting for task to start ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False, value=None), \
        gr.update(visible=False)

    if not task.args:
        print("[Warning] generate_clicked got empty args, not appending to async_tasks.")
        yield gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=[])
        return
    elif len(task.args) < 40:
        print("[Warning] generate_clicked got incomplete args (len < 40), not appending to async_tasks.")
        yield gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=[])
        return
    worker.async_tasks.append(task)

    while not finished:
        time.sleep(0.01)
        if time.perf_counter() - execution_start_time > 60:
            print("[Warning] generate_clicked timeout (1min), forcing finish to prevent UI freeze.")
            yield gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=task.results if hasattr(task, 'results') and task.results else [])
            finished = True
            break

        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':
                if len(task.yields) > 0 and task.yields[0][0] == 'preview':
                    continue
                percentage, title, image = product
                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(), \
                    gr.update(visible=False)
            if flag == 'results':
                yield gr.update(visible=True), \
                    gr.update(visible=True), \
                    gr.update(visible=True, value=product), \
                    gr.update(visible=False)
            if flag == 'finish':
                yield gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value=product)
                finished = True
                if args_manager.args.disable_image_log:
                    for filepath in product:
                        os.remove(filepath)

    execution_time = time.perf_counter() - execution_start_time
    global time_taken
    time_taken = f"Total time: {execution_time:.2f} seconds"
    print(time_taken)
    return

title = f'ReFocus {ReFocus_version.version}'
if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset

shared.gradio_root = gr.Blocks(title=title).queue()

# ========== 转换函数：将 ImageEditor 输出转为旧版格式 ==========
def convert_editor_to_legacy(editor_data):
    """
    将 gr.ImageEditor 的输出转换为旧版 async_worker 期望的格式。
    旧版期望：
        {
            "image": numpy.ndarray,  # RGB, shape (H, W, 3), dtype uint8
            "mask": numpy.ndarray    # RGBA, shape (H, W, 4), dtype uint8, 且红色通道为 mask
        }
    """
    if editor_data is None:
        return None

    bg = editor_data.get("background")
    if bg is None:
        bg = editor_data.get("composite")
    if bg is None:
        return None

    if bg.ndim == 3 and bg.shape[2] == 4:
        bg = bg[:, :, :3]
    elif bg.ndim == 2:
        bg = np.stack([bg] * 3, axis=2)

    H, W = bg.shape[:2]

    mask = np.zeros((H, W), dtype=np.uint8)
    layers = editor_data.get("layers", [])
    for layer in layers:
        if layer is None:
            continue

        if layer.shape[:2] != (H, W):
            from modules.util import resample_image
            layer = resample_image(layer, W, H)
        if layer.ndim == 3 and layer.shape[2] == 4:
            alpha = layer[:, :, 3]
            mask = np.maximum(mask, alpha)
        elif layer.ndim == 2:
            mask = np.maximum(mask, layer)

    mask = (mask > 127).astype(np.uint8) * 255
    mask_rgba = np.zeros((H, W, 4), dtype=np.uint8)
    mask_rgba[:, :, 0] = mask  # 红色通道

    return {
        "image": bg,
        "mask": mask_rgba
    }

# ========== 构建 UI ==========
with shared.gradio_root:
    currentTask = gr.State(worker.AsyncTask(args=[]))
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tab("Generation"):
                with gr.Row():
                    progress_window = gr.Image(label='Preview', show_label=True, visible=False, height=768,
                                               elem_classes=['main_view'])
                    progress_gallery = gr.Gallery(label='Finished Images', show_label=True, object_fit='contain',
                                                  height=768, visible=False, elem_classes=['main_view', 'image_gallery'])
                progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                        elem_id='progress-bar', elem_classes='progress-bar')
                gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', visible=True, height=768,
                                     elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                                     elem_id='final_gallery',
                                     value=["assets/favicon.png"],
                                     preview=True)
            with gr.Tab("Photopea"):
                with gr.Row():
                    photopea = gr.HTML(
                        f"""<iframe id="{PHOTOPEA_IFRAME_ID}" 
                        src = "{PHOTOPEA_MAIN_URL}{get_photopea_url_params()}" 
                        width = "{PHOTOPEA_IFRAME_WIDTH}" 
                        height = "{PHOTOPEA_IFRAME_HEIGHT}"
                        onload = "{PHOTOPEA_IFRAME_LOADED_EVENT}(this)">"""
                    )
                gr.Markdown("Powered by [🦜 Photopea API](https://www.photopea.com/api)")
            with gr.Tab("rembg"):
                with gr.Column(scale=1):
                    rembg_input = gr.Image(label='Drag above image to here', sources=['upload'], type='filepath', scale=20)
                    rembg_button = gr.Button(value="Remove Background", interactive=True, scale=1)
                with gr.Column(scale=3):
                    rembg_output = gr.Image(label='rembg Output', interactive=False, height=380)
                gr.Markdown("Powered by [🪄 rembg 2.0.53](https://github.com/danielgatis/rembg/releases/tag/v2.0.53)")
            def rembg_callback(img):
                if img is None:
                    return None
                from modules.rembg import rembg_run
                return rembg_run(img)
            rembg_button.click(rembg_callback, inputs=rembg_input, outputs=rembg_output, show_progress="full")

            with gr.Tab("Prompt Helper"):
                gr.HTML(
                    '<iframe src="/prompt-helper/?__theme=dark" '
                    'width="100%" height="800px" frameborder="0"></iframe>'
                )

            with gr.Row(elem_classes='type_row'):
                with gr.Column(scale=17):
                    prompt = gr.Textbox(show_label=False, placeholder="Type prompt here or paste parameters.", elem_id='positive_prompt',
                                        container=False, autofocus=True, elem_classes='type_row', lines=1024)

                    default_prompt = modules.config.default_prompt
                    if isinstance(default_prompt, str) and default_prompt != '':
                        shared.gradio_root.load(lambda: default_prompt, outputs=prompt)

                with gr.Column(scale=3, min_width=0):
                    generate_button = gr.Button(value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True)
                    load_parameter_button = gr.Button(value="Load Parameters", elem_classes='type_row', elem_id='load_parameter_button', visible=False)
                    skip_button = gr.Button(value="Skip", elem_classes='type_row_half', visible=False)
                    stop_button = gr.Button(value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False)

                    def stop_clicked(currentTask):
                        import ldm_patched.modules.model_management as model_management
                        currentTask.last_stop = 'stop'
                        if (currentTask.processing):
                            model_management.interrupt_current_processing()
                        return currentTask

                    def skip_clicked(currentTask):
                        import ldm_patched.modules.model_management as model_management
                        currentTask.last_stop = 'skip'
                        if (currentTask.processing):
                            model_management.interrupt_current_processing()
                        return currentTask

                    stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False, js='cancelGenerateForever')
                    skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False)
            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='Input Image', value=False, container=False, elem_classes='min_check')
                advanced_checkbox = gr.Checkbox(label='Advanced', value=modules.config.default_advanced_checkbox, container=False, elem_classes='min_check')

            with gr.Row(visible=False) as image_input_panel:
                with gr.Tabs():
                    with gr.TabItem(label='Upscale or Variation') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                uov_input_image = gr.Image(label='Drag above image to here', sources=['upload'], type='numpy')
                            with gr.Column():
                                # ---- Mode Radio ----
                                uov_mode = gr.Radio(label='Mode:', choices=[flags.UOV_MODE_DISABLED, flags.UOV_MODE_VARY, flags.UOV_MODE_UPSCALE],value=flags.UOV_MODE_DISABLED, interactive=True)
                                uov_vary_mode = gr.Dropdown(label='Vary Mode:',choices=[flags.UOV_VARY_SUBTLE,flags.UOV_VARY_STRONG,'Custom'],value=flags.UOV_VARY_SUBTLE,interactive=True,visible=False)
                                uov_scale = gr.Slider(label='Scale:',minimum=0.25,maximum=4.0,step=0.05,value=2.0,interactive=True,visible=False)

                                with gr.Row(visible=False) as uov_scale_buttons_row:
                                    btn_025x = gr.Button('0.25x', size='sm')
                                    btn_05x = gr.Button('0.5x', size='sm')
                                    btn_15x = gr.Button('1.5x', size='sm')
                                    btn_2x = gr.Button('2x', size='sm')
                                    btn_3x = gr.Button('3x', size='sm')
                                    btn_4x = gr.Button('4x', size='sm')

                                uov_fast = gr.Checkbox(label='Fast Mode (ESRGAN only, no diffusion)',value=False,interactive=True,visible=False)
                                uov_ignore_prompt = gr.Checkbox(label='Ignore Prompt',value=False,interactive=True,visible=False)
                                uov_advanced = gr.Checkbox(label='Advanced', value=False, visible=False)
                                uov_denoise_state = gr.State(value=0.5)
                                uov_denoise_vary = gr.Slider( label='Denoise Strength', minimum=0.0, maximum=1.0, step=0.01, value=0.5, visible=False )
                                uov_denoise_upscale = gr.Slider( label='Denoise Strength', minimum=0.0, maximum=1.0, step=0.01, value=0.382, visible=False )

                                ignore_prompt_cache = gr.State(False)
                                def on_mode_change(mode, vary_mode, advanced_checked):
                                    is_vary = (mode == flags.UOV_MODE_VARY)
                                    is_upscale = (mode == flags.UOV_MODE_UPSCALE)
                                    is_active = (mode != flags.UOV_MODE_DISABLED)

                                    is_vary_custom = is_vary and (vary_mode == 'Custom')
                                    is_upscale_advanced = is_upscale and advanced_checked
                                    if mode == flags.UOV_MODE_VARY:
                                        if vary_mode == flags.UOV_VARY_SUBTLE:
                                            denoise_value = 0.50
                                        elif vary_mode == flags.UOV_VARY_STRONG:
                                            denoise_value = 0.85
                                        else:  # Custom
                                            denoise_value = 0.50
                                    elif mode == flags.UOV_MODE_UPSCALE:
                                        denoise_value = 0.382
                                    else:  # Disabled
                                        denoise_value = 0.5
                                    
                                    return [
                                        gr.update(visible=is_vary),                     # uov_vary_mode
                                        gr.update(visible=is_upscale),                  # uov_scale
                                        gr.update(visible=is_upscale),                  # uov_scale_buttons_row
                                        gr.update(visible=is_upscale),                  # uov_fast
                                        gr.update(visible=is_active),                   # uov_ignore_prompt
                                        gr.update(visible=is_upscale),                  # uov_advanced
                                        gr.update(visible=is_vary_custom, value=denoise_value),  # uov_denoise_vary
                                        gr.update(visible=is_upscale_advanced, value=denoise_value),  # uov_denoise_upscale
                                        denoise_value,                                  # uov_denoise_state
                                    ]
                                uov_mode.change(on_mode_change, inputs=[uov_mode, uov_vary_mode, uov_advanced], outputs=[uov_vary_mode, uov_scale, uov_scale_buttons_row, uov_fast, uov_ignore_prompt, uov_advanced, uov_denoise_vary, uov_denoise_upscale, uov_denoise_state])

                                def on_vary_mode_change(vary_mode, current_mode):
                                    is_vary = (current_mode == flags.UOV_MODE_VARY)
                                    if not is_vary:
                                        return [gr.update(), gr.update(), gr.update()]
                                    
                                    if vary_mode == flags.UOV_VARY_SUBTLE:
                                        return [
                                            gr.update(value=0.50, visible=False),
                                            gr.update(value=0.50),
                                            gr.update()
                                        ]
                                    elif vary_mode == flags.UOV_VARY_STRONG:
                                        return [
                                            gr.update(value=0.85, visible=False),
                                            gr.update(value=0.85),
                                            gr.update()
                                        ]
                                    else:  # Custom
                                        return [
                                            gr.update(value=0.50, visible=True),
                                            gr.update(value=0.50),
                                            gr.update()
                                        ]
                                uov_vary_mode.change(on_vary_mode_change, inputs=[uov_vary_mode, uov_mode], outputs=[uov_denoise_vary, uov_denoise_state, uov_denoise_upscale])

                                def on_advanced_change(advanced_checked):
                                    if advanced_checked:
                                        return [
                                            gr.update(value=0.382, visible=True),  # uov_denoise_upscale
                                            0.382                                  # uov_denoise_state
                                        ]
                                    else:
                                        return [
                                            gr.update(visible=False),  # uov_denoise_upscale
                                            gr.update()                # uov_denoise_state（保持当前值，不影响）
                                        ]

                                uov_advanced.change(
                                    on_advanced_change,
                                    inputs=[uov_advanced],
                                    outputs=[uov_denoise_upscale, uov_denoise_state]
                                )
                                # Vary Custom Slider 值变化时更新 State
                                uov_denoise_vary.input(
                                    lambda val: gr.update(value=val),
                                    inputs=[uov_denoise_vary],
                                    outputs=[uov_denoise_state]
                                )

                                # Upscale Slider 值变化时更新 State
                                uov_denoise_upscale.input(
                                    lambda val: gr.update(value=val),
                                    inputs=[uov_denoise_upscale],
                                    outputs=[uov_denoise_state]
                                )

                                def on_fast_change(fast_checked, current_ignore, cached_ignore):
                                    if fast_checked:
                                        return (gr.update(value=True, interactive=False),current_ignore)
                                    else:
                                        
                                        return (gr.update(value=cached_ignore, interactive=True),cached_ignore)
                                uov_fast.change(on_fast_change,inputs=[uov_fast, uov_ignore_prompt, ignore_prompt_cache],outputs=[uov_ignore_prompt, ignore_prompt_cache])

                                btn_025x.click(lambda: gr.update(value=0.25), outputs=[uov_scale])
                                btn_05x.click(lambda: gr.update(value=0.5), outputs=[uov_scale])
                                btn_15x.click(lambda: gr.update(value=1.5), outputs=[uov_scale])
                                btn_2x.click(lambda: gr.update(value=2.0), outputs=[uov_scale])
                                btn_3x.click(lambda: gr.update(value=3.0), outputs=[uov_scale])
                                btn_4x.click(lambda: gr.update(value=4.0), outputs=[uov_scale])

                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Document</a>')
                    with gr.TabItem(label='Image Prompt') as ip_tab:
                        with gr.Row():
                            ip_images = []
                            ip_types = []
                            ip_stops = []
                            ip_weights = []
                            ip_ctrls = []
                            ip_ad_cols = []
                            for _ in range(flags.controlnet_image_count):
                                with gr.Column():
                                    ip_image = gr.Image(label='Image', sources=['upload'], type='numpy', show_label=False, height=300)
                                    ip_images.append(ip_image)
                                    ip_ctrls.append(ip_image)
                                    with gr.Column(visible=False) as ad_col:
                                        with gr.Row():
                                            default_end, default_weight = flags.default_parameters[flags.default_ip]

                                            ip_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.001, value=default_end)
                                            ip_stops.append(ip_stop)
                                            ip_ctrls.append(ip_stop)

                                            ip_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.001, value=default_weight)
                                            ip_weights.append(ip_weight)
                                            ip_ctrls.append(ip_weight)

                                        ip_type = gr.Radio(label='Type', choices=flags.ip_list, value=flags.default_ip, container=False)
                                        ip_types.append(ip_type)
                                        ip_ctrls.append(ip_type)

                                        ip_type.change(lambda x: flags.default_parameters[x], inputs=[ip_type], outputs=[ip_stop, ip_weight], queue=False, show_progress=False)
                                    ip_ad_cols.append(ad_col)
                        ip_advanced = gr.Checkbox(label='Advanced', value=False, container=False)
                        gr.HTML('* \"Image Prompt\" is powered by Fooocus Image Mixture Engine (v1.0.1). <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">\U0001F4D4 Document</a>')

                        def ip_advance_checked(x):
                            return [gr.update(visible=x)] * len(ip_ad_cols) + \
                                [flags.default_ip] * len(ip_types) + \
                                [flags.default_parameters[flags.default_ip][0]] * len(ip_stops) + \
                                [flags.default_parameters[flags.default_ip][1]] * len(ip_weights)

                        ip_advanced.change(ip_advance_checked, inputs=ip_advanced,
                                           outputs=ip_ad_cols + ip_types + ip_stops + ip_weights,
                                           queue=False, show_progress=False)

                    with gr.TabItem(label='Inpaint or Outpaint') as inpaint_tab:
                        with gr.Row():
                            with gr.Column():
                                inpaint_input_image = gr.ImageEditor(
                                    label='Drag inpaint or outpaint image to here',
                                    sources=['upload'],
                                    type='numpy',
                                    height=500,
                                    elem_id='inpaint_canvas'
                                )
                                inpaint_data_legacy = gr.State(value=None)
                                inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.flags.inpaint_option_default, label='Method')
                                inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=False)
                                outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='Outpaint Direction')
                                example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts,
                                                                     label='Additional Prompt Quick List',
                                                                     components=[inpaint_additional_prompt],
                                                                     visible=False)
                                gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Document</a>')
                                example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)

                            with gr.Column(visible=False) as inpaint_mask_generation_col:
                                inpaint_mask_image = gr.Image(label='Mask Upload', sources=['upload'], type='numpy',
                                                               height=500)
                                inpaint_mask_model = gr.Dropdown(label='Mask generation model',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_inpaint_mask_model)
                                inpaint_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                             choices=flags.inpaint_mask_cloth_category,
                                                             value=modules.config.default_inpaint_mask_cloth_category,
                                                             visible=False)
                                inpaint_mask_sam_prompt_text = gr.Textbox(label='Segmentation prompt', value='', visible=False)
                                with gr.Accordion("Advanced options", visible=False, open=False) as inpaint_mask_advanced_options:
                                    inpaint_mask_sam_model = gr.Dropdown(label='SAM model', choices=flags.inpaint_mask_sam_model, value=modules.config.default_inpaint_mask_sam_model)
                                    inpaint_mask_sam_quant = gr.Checkbox(label='Quantization', value=False)
                                    inpaint_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                                    inpaint_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05)
                                generate_mask_button = gr.Button(value='Generate mask from image')

                                def generate_mask(image, mask_model, cloth_category, sam_prompt_text, sam_model, sam_quant, box_threshold, text_threshold):
                                    from extras.inpaint_mask import generate_mask_from_image

                                    extras = {}
                                    if mask_model == 'u2net_cloth_seg':
                                        extras['cloth_category'] = cloth_category
                                    elif mask_model == 'sam':
                                        extras['sam_prompt_text'] = sam_prompt_text
                                        extras['sam_model'] = sam_model
                                        extras['sam_quant'] = sam_quant
                                        extras['box_threshold'] = box_threshold
                                        extras['text_threshold'] = text_threshold

                                    return generate_mask_from_image(image, mask_model, extras)

                                generate_mask_button.click(fn=generate_mask,
                                                           inputs=[
                                                               inpaint_input_image, inpaint_mask_model,
                                                               inpaint_mask_cloth_category,
                                                               inpaint_mask_sam_prompt_text,
                                                               inpaint_mask_sam_model,
                                                               inpaint_mask_sam_quant,
                                                               inpaint_mask_box_threshold,
                                                               inpaint_mask_text_threshold
                                                           ],
                                                           outputs=inpaint_mask_image, show_progress=True, queue=True)

                                inpaint_mask_model.change(lambda x: [gr.update(visible=x == 'u2net_cloth_seg'), gr.update(visible=x == 'sam'), gr.update(visible=x == 'sam')],
                                                          inputs=inpaint_mask_model,
                                                          outputs=[inpaint_mask_cloth_category, inpaint_mask_sam_prompt_text, inpaint_mask_advanced_options],
                                                          queue=False, show_progress=False)

                    with gr.TabItem(label='Describe') as desc_tab:
                        with gr.Row():
                            with gr.Column():
                                desc_input_image = gr.Image(label='Drag any image to here', sources=['upload'], type='numpy')
                            with gr.Column():
                                desc_method = gr.Radio(
                                    label='Content Type',
                                    choices=[flags.desc_type_photo, flags.desc_type_anime],
                                    value=flags.desc_type_photo)
                                desc_btn = gr.Button(value='Describe this Image into Prompt')
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/1363" target="_blank">\U0001F4D4 Document</a>')
                    with gr.TabItem(label='Metadata') as load_tab:
                        with gr.Column():
                            metadata_input_image = gr.Image(label='Drag any image generated by Fooocus here', sources=['upload'], type='filepath')
                            metadata_json = gr.JSON(label='Metadata')
                            metadata_import_button = gr.Button(value='Apply Metadata')

                        def trigger_metadata_preview(filepath):
                            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(filepath)

                            results = {}
                            if parameters is not None:
                                results['parameters'] = parameters

                            if isinstance(metadata_scheme, flags.MetadataScheme):
                                results['metadata_scheme'] = metadata_scheme.value

                            return results

                        metadata_input_image.upload(trigger_metadata_preview, inputs=metadata_input_image,
                                                    outputs=metadata_json, queue=False, show_progress=True)

            switch_js = "(x) => {if(x){viewer_to_bottom(100);viewer_to_bottom(500);}else{viewer_to_top();} return x;}"
            down_js = "() => {viewer_to_bottom();}"

            input_image_checkbox.change(lambda x: gr.update(visible=x), inputs=input_image_checkbox,
                                        outputs=image_input_panel, queue=False, show_progress=False, js=switch_js)
            ip_advanced.change(lambda: None, queue=False, show_progress=False, js=down_js)

            current_tab = gr.Textbox(value='uov', visible=False)
            uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, js=down_js, show_progress=False)
            inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, js=down_js, show_progress=False)
            ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, js=down_js, show_progress=False)
            desc_tab.select(lambda: 'desc', outputs=current_tab, queue=False, js=down_js, show_progress=False)

        with gr.Column(scale=1, visible=modules.config.default_advanced_checkbox) as advanced_column:
            with gr.Tab(label='Settings'):
                with gr.Row():
                    with gr.Row():
                        steps_slider = gr.Slider(
                            minimum=1, maximum=50, step=1,
                            value=modules.config.default_steps if hasattr(modules.config, 'default_steps') else 25,
                            label="Steps",
                            elem_id="steps_slider"
                        )
                    with gr.Row():
                        preset_45 = gr.Button("45 (Quality)", size="sm")
                        preset_25 = gr.Button("25 (Speed)", size="sm")
                        preset_10 = gr.Button("10 (Extreme)", size="sm")
                        if not args_manager.args.disable_preset_selection:
                            preset_selection = gr.Dropdown(label='Preset',
                                                        choices=modules.config.available_presets,
                                                        value=args_manager.args.preset if args_manager.args.preset else "initial",
                                                        interactive=True)
                    preset_45.click(lambda: gr.update(value=45), outputs=steps_slider)
                    preset_25.click(lambda: gr.update(value=25), outputs=steps_slider)
                    preset_10.click(lambda: gr.update(value=10), outputs=steps_slider)

                aspect_ratios_selection = gr.Radio(label='Aspect Ratios', choices=modules.config.available_aspect_ratios,
                                                   value=modules.config.default_aspect_ratio, info='width × height',
                                                   elem_classes='aspect_ratios')

                with gr.Column():
                    sampling_apply = gr.Checkbox(label="Sampling", value=False)
                    with gr.Row(visible=False) as sampling:
                        sampler_name = gr.Dropdown(label='Sampler', choices=flags.sampler_list,
                                                    value=modules.config.default_sampler)
                        scheduler_name = gr.Dropdown(label='Scheduler', choices=flags.scheduler_list,
                                                    value=modules.config.default_scheduler)
                sampling_apply.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=sampling_apply,
                    outputs=sampling,
                    queue=False,
                    api_name=False,
                )

                image_number = gr.Slider(label='Image Number', minimum=1, maximum=modules.config.default_max_image_number, step=1, value=modules.config.default_image_number)

                negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True, placeholder="Type prompt here.",
                                             info='Describing what you do not want to see.', lines=2,
                                             elem_id='negative_prompt',
                                             value=modules.config.default_prompt_negative)
                translate_prompts = gr.Checkbox(label='Translate Prompts',
                                                          info='Uses the internet to translate prompts to English.',
                                                          value=False)
                seed_random = gr.Checkbox(label='Randomize seed', value=True)
                image_seed = gr.Textbox(label='Seed', value=0, max_lines=1, visible=False)

                def random_checked(r):
                    return gr.update(visible=not r)

                def refresh_seed(r, seed_string):
                    if r:
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)
                    else:
                        try:
                            seed_value = int(seed_string)
                            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                                return seed_value
                        except ValueError:
                            pass
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)

                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed],
                                   queue=False, show_progress=False)

                def update_history_link():
                    if args_manager.args.disable_image_log:
                        return gr.update(value='')

                    return gr.update(value=f'<a href="file={get_current_html_path()}" target="_blank">\U0001F4DA History Log</a>')

                history_link = gr.HTML()
                shared.gradio_root.load(update_history_link, outputs=history_link, queue=False, show_progress=False)

            with gr.Tab(label='Models'):
                with gr.Group():
                    with gr.Row():
                        base_model = gr.Dropdown(label='Base Model (SDXL only)', choices=modules.config.model_filenames, value=modules.config.default_base_model_name, show_label=True)
                        refiner_model = gr.Dropdown(label='Refiner (SDXL or SD 1.5)', choices=['None'] + modules.config.model_filenames, value=modules.config.default_refiner_model_name, show_label=True)

                    refiner_switch = gr.Slider(label='Refiner Switch At', minimum=0.1, maximum=1.0, step=0.0001,
                                               info='Use 0.4 for SD1.5 realistic models; '
                                                    'or 0.667 for SD1.5 anime models; '
                                                    'or 0.8 for XL-refiners; '
                                                    'or any value for switching two SDXL models.',
                                               value=modules.config.default_refiner_switch,
                                               visible=modules.config.default_refiner_model_name != 'None')

                    refiner_model.change(lambda x: gr.update(visible=x != 'None'),
                                         inputs=refiner_model, outputs=refiner_switch, show_progress=False, queue=False)

                with gr.Group():
                    lora_ctrls = []

                    for i, (n, v) in enumerate(modules.config.default_loras):
                        with gr.Row():
                            lora_model = gr.Dropdown(label=f'LoRA {i + 1}',
                                                     choices=['None'] + modules.config.lora_filenames, value=n)
                            lora_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.01, value=v,
                                                    elem_classes='lora_weight')
                            lora_ctrls += [lora_model, lora_weight]

                with gr.Row():
                    model_refresh = gr.Button(value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button')
            with gr.Tab(label='Advanced'):
                guidance_scale = gr.Slider(label='Guidance Scale', minimum=1.0, maximum=30.0, step=0.01,
                                           value=modules.config.default_cfg_scale,
                                           info='Higher value means style is cleaner, vivider, and more artistic.')
                sharpness = gr.Slider(label='Image Sharpness', minimum=0.0, maximum=30.0, step=0.001,
                                      value=modules.config.default_sample_sharpness,
                                      info='Higher value means image and texture are sharper.')
                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/117" target="_blank">\U0001F4D4 Document</a>')
                output_format = gr.Radio(label='Output Format',
                                            choices=modules.flags.output_formats,
                                            value=modules.config.default_output_format)

                dev_mode = gr.Checkbox(label='Advanced mode', value=True, container=False)

                with gr.Column(visible=True) as dev_tools:
                    with gr.Tab(label='Debug Tools'):
                        adm_scaler_positive = gr.Slider(label='Positive ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=1.5, info='The scaler multiplied to positive ADM (use 1.0 to disable). ')
                        adm_scaler_negative = gr.Slider(label='Negative ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=0.8, info='The scaler multiplied to negative ADM (use 1.0 to disable). ')
                        adm_scaler_end = gr.Slider(label='ADM Guidance End At Step', minimum=0.0, maximum=1.0,
                                                   step=0.001, value=0.3,
                                                   info='When to end the guidance from positive/negative ADM. ')

                        refiner_swap_method = gr.Dropdown(label='Refiner swap method', value=flags.refiner_swap_method,
                                                          choices=['joint', 'separate', 'vae'])

                        adaptive_cfg = gr.Slider(label='CFG Mimicking from TSNR', minimum=1.0, maximum=30.0, step=0.01,
                                                 value=modules.config.default_cfg_tsnr,
                                                 info='Enabling Fooocus\'s implementation of CFG mimicking for TSNR '
                                                      '(effective when real CFG > mimicked CFG).')
                        generate_image_grid = gr.Checkbox(label='Generate Image Grid for Each Batch',
                                                          info='(Experimental) This may cause performance problems on some computers and certain internet conditions.',
                                                          value=False)

                        overwrite_step = gr.Slider(label='Forced Overwrite of Sampling Step',
                                                   minimum=-1, maximum=200, step=1,
                                                   value=modules.config.default_overwrite_step,
                                                   info='Set as -1 to disable. For developer debugging.')
                        overwrite_switch = gr.Slider(label='Forced Overwrite of Refiner Switch Step',
                                                     minimum=-1, maximum=200, step=1,
                                                     value=modules.config.default_overwrite_switch,
                                                     info='Set as -1 to disable. For developer debugging.')
                        overwrite_width = gr.Slider(label='Forced Overwrite of Generating Width',
                                                    minimum=-1, maximum=2048, step=1, value=-1,
                                                    info='Set as -1 to disable. For developer debugging. '
                                                         'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_height = gr.Slider(label='Forced Overwrite of Generating Height',
                                                     minimum=-1, maximum=2048, step=1, value=-1,
                                                     info='Set as -1 to disable. For developer debugging. '
                                                          'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_vary_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Vary"',
                                                            minimum=-1, maximum=1.0, step=0.001, value=-1,
                                                            info='Set as negative number to disable. For developer debugging.')
                        overwrite_upscale_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Upscale"',
                                                               minimum=-1, maximum=1.0, step=0.001,
                                                               value=modules.config.default_overwrite_upscale,
                                                               info='Set as negative number to disable. For developer debugging.')

                        disable_preview = gr.Checkbox(label='Disable Preview', value=modules.config.default_black_out_nsfw,
                                                      interactive=not modules.config.default_black_out_nsfw,
                                                      info='Disable preview during generation.')
                        disable_intermediate_results = gr.Checkbox(label='Disable Intermediate Results', 
                                                      value=modules.config.default_performance == 'Extreme Speed',
                                                      interactive=modules.config.default_performance != 'Extreme Speed',
                                                      info='Disable intermediate results during generation, only show final gallery.')

                        black_out_nsfw = gr.Checkbox(label='Black Out NSFW', value=modules.config.default_black_out_nsfw,
                                                     interactive=not modules.config.default_black_out_nsfw,
                                                     info='Use black image if NSFW is detected.')

                        black_out_nsfw.change(lambda x: gr.update(value=x, interactive=not x),
                                     inputs=black_out_nsfw, outputs=disable_preview, queue=False, show_progress=False)

                        if not args_manager.args.disable_metadata:
                            save_metadata_to_images = gr.Checkbox(label='Save Metadata to Images', value=modules.config.default_save_metadata_to_images,
                                                                  info='Adds parameters to generated images allowing manual regeneration.')
                            metadata_scheme = gr.Radio(label='Metadata Scheme', choices=flags.metadata_scheme, value=modules.config.default_metadata_scheme,
                                                       info='Image Prompt parameters are not included. Use a1111 for compatibility with Civitai.',
                                                       visible=modules.config.default_save_metadata_to_images)

                            save_metadata_to_images.change(lambda x: gr.update(visible=x), inputs=[save_metadata_to_images], outputs=[metadata_scheme], 
                                                           queue=False, show_progress=False)

                    with gr.Tab(label='Control'):
                        debugging_cn_preprocessor = gr.Checkbox(label='Debug Preprocessors', value=False,
                                                                info='See the results from preprocessors.')
                        skipping_cn_preprocessor = gr.Checkbox(label='Skip Preprocessors', value=False,
                                                               info='Do not preprocess images. (Inputs are already canny/depth/cropped-face/etc.)')

                        mixing_image_prompt_and_vary_upscale = gr.Checkbox(label='Mixing Image Prompt and Vary/Upscale',
                                                                           value=False)
                        mixing_image_prompt_and_inpaint = gr.Checkbox(label='Mixing Image Prompt and Inpaint',
                                                                      value=False)

                        controlnet_softness = gr.Slider(label='Softness of ControlNet', minimum=0.0, maximum=1.0,
                                                        step=0.001, value=0.25,
                                                        info='Similar to the Control Mode in A1111 (use 0.0 to disable). ')

                        with gr.Tab(label='Canny'):
                            canny_low_threshold = gr.Slider(label='Canny Low Threshold', minimum=1, maximum=255,
                                                            step=1, value=64)
                            canny_high_threshold = gr.Slider(label='Canny High Threshold', minimum=1, maximum=255,
                                                             step=1, value=128)

                    with gr.Tab(label='Inpaint'):
                        debugging_inpaint_preprocessor = gr.Checkbox(label='Debug Inpaint Preprocessing', value=False)
                        inpaint_disable_initial_latent = gr.Checkbox(label='Disable initial latent in inpaint', value=False)
                        inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                     value=modules.config.default_inpaint_engine_version,
                                                     choices=flags.inpaint_engine_versions,
                                                     info='Version of Fooocus inpaint model')
                        inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                     minimum=0.0, maximum=1.0, step=0.001, value=1.0,
                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                          'Only used in inpaint, not used in outpaint. '
                                                          '(Outpaint always use 1.0)')
                        inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                             minimum=0.0, maximum=1.0, step=0.001, value=0.618,
                                                             info='The area to inpaint. '
                                                                  'Value 0 is same as "Only Masked" in A1111. '
                                                                  'Value 1 is same as "Whole Image" in A1111. '
                                                                  'Only used in inpaint, not used in outpaint. '
                                                                  '(Outpaint always use 1.0)')
                        inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                            minimum=-64, maximum=64, step=1, value=0,
                                                            info='Positive value will make white area in the mask larger, '
                                                                 'negative value will make white area smaller.'
                                                                 '(default is 0, always process before any mask invert)')
                        inpaint_mask_upload_checkbox = gr.Checkbox(label='Enable Mask Upload', value=False)
                        invert_mask_checkbox = gr.Checkbox(label='Invert Mask', value=False)

                        inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                                         inpaint_strength, inpaint_respective_field,
                                         inpaint_mask_upload_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate]

                        inpaint_mask_upload_checkbox.change(lambda x: [gr.update(visible=x)] * 2,
                                                            inputs=inpaint_mask_upload_checkbox,
                                                            outputs=[inpaint_mask_image, inpaint_mask_generation_col],
                                                            queue=False, show_progress=False)

                    with gr.Tab(label='FreeU'):
                        freeu_enabled = gr.Checkbox(label='Enabled', value=False)
                        freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
                        freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
                        freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
                        freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
                        freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]

                def dev_mode_checked(r):
                    return gr.update(visible=r)

                dev_mode.change(dev_mode_checked, inputs=[dev_mode], outputs=[dev_tools],
                                queue=False, show_progress=False)

                def model_refresh_clicked():
                    modules.config.update_all_model_names()
                    modules.config.update_presets()
                    results = []
                    results += [gr.update(choices=modules.config.model_filenames),
                                gr.update(choices=['None'] + modules.config.model_filenames)]
                    if not args_manager.args.disable_preset_selection:
                        results += [gr.update(choices=modules.config.available_presets)]
                    for i in range(flags.lora_count):
                        results += [gr.update(choices=['None'] + modules.config.lora_filenames), gr.update()]
                    return results

                model_refresh_output = [base_model, refiner_model]
                if not args_manager.args.disable_preset_selection:
                    model_refresh_output += [preset_selection]
                model_refresh.click(model_refresh_clicked, [],  model_refresh_output + lora_ctrls,
                                    queue=False, show_progress=False)


        state_is_generating = gr.State(False)

        load_data_outputs = [advanced_checkbox, image_number, prompt, negative_prompt, 
                             steps_slider, overwrite_step, overwrite_switch, aspect_ratios_selection,
                             overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                             adm_scaler_negative, adm_scaler_end, refiner_swap_method, adaptive_cfg, base_model,
                             refiner_model, refiner_switch, sampler_name, scheduler_name, seed_random, image_seed,
                             generate_button, load_parameter_button] + freeu_ctrls + lora_ctrls

        if not args_manager.args.disable_preset_selection:
            def preset_selection_change(preset, is_generating):
                preset_content = modules.config.try_get_preset_content(preset) if preset != 'initial' else {}
                preset_prepared = modules.meta_parser.parse_meta_from_preset(preset_content)

                preset_prepared.pop('previous_default_models', None)
                preset_prepared.pop('checkpoint_downloads', None)
                preset_prepared.pop('embeddings_downloads', None)
                preset_prepared.pop('lora_downloads', None)

                if 'prompt' in preset_prepared and preset_prepared.get('prompt') == '':
                    del preset_prepared['prompt']

                return modules.meta_parser.load_parameter_button_click(
                    json.dumps(preset_prepared),
                    is_generating
                )

            preset_selection.change(
                preset_selection_change,
                inputs=[preset_selection, state_is_generating],
                outputs=load_data_outputs,
                queue=False,
                show_progress=True
            )

        output_format.input(lambda x: gr.update(output_format=x), inputs=output_format)

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, advanced_column,
                                 queue=False, show_progress=False) \
            .then(fn=lambda: None, js='refresh_grid_delayed', queue=False, show_progress=False)

        def inpaint_mode_change(mode):
            assert mode in modules.flags.inpaint_options

            if mode == modules.flags.inpaint_option_detail:
                return [
                    gr.update(visible=True, value=''),
                    gr.update(visible=False, value=[]),
                    gr.update(visible=True),
                    gr.update(value=False),
                    gr.update(value='None'),
                    gr.update(value=0.5),
                    gr.update(value=0.0)
                ]
            elif mode == modules.flags.inpaint_option_modify:
                return [
                    gr.update(visible=True, value=''),
                    gr.update(visible=False, value=[]),
                    gr.update(visible=False),
                    gr.update(value=False),
                    gr.update(value=modules.config.default_inpaint_engine_version),
                    gr.update(value=1.0),
                    gr.update(value=0.0)
                ]
            else:  # 默认 'Inpaint or Outpaint (default)'
                return [
                    gr.update(visible=False, value=''),
                    gr.update(visible=True, value=[]),
                    gr.update(visible=False),
                    gr.update(value=False),
                    gr.update(value=modules.config.default_inpaint_engine_version),
                    gr.update(value=1.0),
                    gr.update(value=0.618)
                ]

        inpaint_mode.change(
            inpaint_mode_change,
            inputs=inpaint_mode,
            outputs=[
                inpaint_additional_prompt,
                outpaint_selections,
                example_inpaint_prompts,
                inpaint_disable_initial_latent,
                inpaint_engine,
                inpaint_strength,
                inpaint_respective_field
            ],
            show_progress=False,
            queue=False
        )

        inpaint_input_image.change(
            fn=convert_editor_to_legacy,
            inputs=inpaint_input_image,
            outputs=inpaint_data_legacy,
            queue=False
        )

        ctrls = [currentTask, generate_image_grid]
        ctrls += [
            prompt, negative_prompt, translate_prompts, 
            steps_slider, aspect_ratios_selection, image_number, output_format, image_seed, sharpness, guidance_scale
        ]

        ctrls += [base_model, refiner_model, refiner_switch] + lora_ctrls
        ctrls += [input_image_checkbox, current_tab]
        ctrls += [uov_mode, uov_vary_mode, uov_scale, uov_fast, uov_ignore_prompt, uov_denoise_state, uov_input_image]
        ctrls += [outpaint_selections, inpaint_data_legacy, inpaint_additional_prompt, inpaint_mask_image]
        ctrls += [disable_preview, disable_intermediate_results, black_out_nsfw]
        ctrls += [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg]
        ctrls += [sampler_name, scheduler_name]
        ctrls += [overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength]
        ctrls += [overwrite_upscale_strength, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint]
        ctrls += [debugging_cn_preprocessor, skipping_cn_preprocessor, canny_low_threshold, canny_high_threshold]
        ctrls += [refiner_swap_method, controlnet_softness]
        ctrls += freeu_ctrls
        ctrls += inpaint_ctrls

        if not args_manager.args.disable_metadata:
            ctrls += [save_metadata_to_images, metadata_scheme]

        ctrls += ip_ctrls

        def parse_meta(raw_prompt_txt, is_generating):
            loaded_json = None
            if is_json(raw_prompt_txt):
                loaded_json = json.loads(raw_prompt_txt)

            if loaded_json is None:
                if is_generating:
                    return gr.update(), gr.update(), gr.update()
                else:
                    return gr.update(), gr.update(visible=True), gr.update(visible=False)

            return json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True)

        prompt.input(parse_meta, inputs=[prompt, state_is_generating], outputs=[prompt, generate_button, load_parameter_button], queue=False, show_progress=False)

        load_parameter_button.click(modules.meta_parser.load_parameter_button_click, inputs=[prompt, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=False)

        def trigger_metadata_import(filepath, state_is_generating):
            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(filepath)
            if parameters is None:
                print('Could not find metadata in the image!')
                parsed_parameters = {}
            else:
                metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                parsed_parameters = metadata_parser.parse_json(parameters)

            return modules.meta_parser.load_parameter_button_click(parsed_parameters, state_is_generating)

        metadata_import_button.click(
            trigger_metadata_import,
            inputs=[metadata_input_image, state_is_generating],
            outputs=load_data_outputs,
            queue=False,
            show_progress=True
        )

        generate_button.click(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), [], True),
                              outputs=[stop_button, skip_button, generate_button, gallery, state_is_generating]) \
            .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
            .then(fn=get_task, inputs=ctrls, outputs=currentTask) \
            .then(fn=generate_clicked, inputs=currentTask, outputs=[progress_html, progress_window, progress_gallery, gallery]) \
            .then(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), False),
                  outputs=[generate_button, stop_button, skip_button, state_is_generating]) \
            .then(fn=update_history_link, outputs=history_link) \
            .then(fn=lambda: None, js='refresh_grid_delayed')

        def trigger_describe(mode, img):
            if mode == flags.desc_type_photo:
                from extras.interrogate import default_interrogator as default_interrogator_photo
                return default_interrogator_photo(img)

            if mode == flags.desc_type_anime:
                from extras.wd14tagger import default_interrogator as default_interrogator_anime
                return default_interrogator_anime(img)

            return mode

        desc_btn.click(
            trigger_describe,
            inputs=[desc_method, desc_input_image],
            outputs=[prompt],
            show_progress=True,
            queue=True
        )
        def trigger_uov_describe(mode, img, prompt):
            if prompt == '':
                return trigger_describe(mode, img)
            return gr.update()

        uov_input_image.upload(
            trigger_uov_describe,
            inputs=[desc_method, uov_input_image, prompt],
            outputs=[prompt],
            show_progress=True,
            queue=True
        )
