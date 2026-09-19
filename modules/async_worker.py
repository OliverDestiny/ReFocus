# This file is part of ReFocus.
# Original work Copyright (c) 2023 lllyasviel (Fooocus) & 2024 ehristoforu (DeFooocus).
# Modified and distributed under the terms of the GNU General Public License v3.0.

import threading
import os
from modules.patch import PatchSettings, patch_settings, patch_all

patch_all()


class AsyncTask:
    def __init__(self, args):
        self.args = args
        self.yields = []
        self.results = []
        self.last_stop = False
        self.processing = False


async_tasks = []


def worker():
    global async_tasks

    import traceback
    import math
    import numpy as np
    import torch
    import time
    import cv2
    import modules.default_pipeline as pipeline
    import modules.core as core
    import modules.flags as flags
    import modules.config
    import modules.arg_schema as arg_schema
    import modules.controlnet_registry as controlnet_registry
    import modules.patch
    import ldm_patched.modules.model_management
    import modules.inpaint_worker as inpaint_worker
    import modules.constants as constants
    import extras.ip_adapter as ip_adapter
    import modules.deps_models_download as downloader
    import ReFocus_version
    import args_manager

    from modules.censor import censor_batch

    from modules.private_logger import log
    from modules.util import safe_str, remove_empty_str, HWC3, \
        get_image_shape_ceil, set_image_shape_ceil, get_shape_ceil, resample_image, erode_or_dilate
    from modules.upscaler import perform_upscale
    from modules.flags import lora_count
    from modules.meta_parser import get_metadata_parser, MetadataScheme

    pid = os.getpid()
    print(f'Started worker with PID {pid}')

    def progressbar(async_task, number, text):
        print(f'[ReFocus] {text}')
        async_task.yields.append(['preview', (number, text, None)])

    def yield_result(async_task, imgs, black_out_nsfw, do_not_show_finished_images=False, progressbar_index=13):
        if not isinstance(imgs, list):
            imgs = [imgs]

        if modules.config.default_black_out_nsfw or black_out_nsfw:
            progressbar(async_task, progressbar_index, 'Checking for NSFW content ...')
            imgs_censor = [cv2.imread(img) for img in imgs]
            imgs_censor = censor_batch(imgs_censor)
            for i, img in enumerate(imgs):
                cv2.imwrite(img, imgs_censor[i])

        async_task.results = async_task.results + imgs

        if do_not_show_finished_images:
            return

        async_task.yields.append(['results', async_task.results])
        return

    @torch.no_grad()
    @torch.inference_mode()
    def handler(async_task):
        execution_start_time = time.perf_counter()
        async_task.processing = True

        args = async_task.args
        args.reverse()

        # 参数顺序的单一事实来源是 modules/arg_schema.py。
        # 这里按表消费，顺序永远与 UI 一致；长度不符时立刻报错而不是静默错位。
        expected_args = arg_schema.active_args()
        if len(args) != len(expected_args):
            raise RuntimeError(
                f'[ReFocus] Argument contract mismatch: worker received {len(args)} args, '
                f'schema expects {len(expected_args)}. '
                f'Update modules/arg_schema.py together with both ends.'
            )
        raw = {name: args.pop() for name in expected_args}

        prompt = raw['prompt'] or ''
        negative_prompt = raw['negative_prompt'] or ''
        translate_prompts = raw['translate_prompts']
        steps = raw['steps']
        aspect_ratios_selection = raw['aspect_ratios_selection']
        image_number = raw['image_number']
        output_format = raw['output_format']
        image_seed = raw['image_seed']
        sharpness = raw['sharpness']
        guidance_scale = raw['guidance_scale']
        base_model_name = raw['base_model_name']
        refiner_model_name = raw['refiner_model_name']
        refiner_switch = raw['refiner_switch']
        loras = [[str(raw[f'lora_model_{i + 1}']), float(raw[f'lora_weight_{i + 1}'])] for i in range(lora_count)]
        input_image_checkbox = raw['input_image_checkbox']
        current_tab = raw['current_tab']
        uov_mode = raw['uov_mode']              # 'Disabled' | 'Vary' | 'Upscale'
        uov_vary_mode = raw['uov_vary_mode']    # 'Subtle' | 'Strong'
        uov_scale = raw['uov_scale']            # float, 0.25 ~ 4.0
        uov_fast = raw['uov_fast']              # bool
        uov_ignore_prompt = raw['uov_ignore_prompt']  # bool
        uov_denoise = raw['uov_denoise']        # float, 0.0 ~ 1.0
        uov_input_image = raw['uov_input_image']
        outpaint_selections = raw['outpaint_selections']
        inpaint_input_image = raw['inpaint_input_image']
        inpaint_additional_prompt = raw['inpaint_additional_prompt'] or ''

        disable_preview = raw['disable_preview']
        disable_intermediate_results = raw['disable_intermediate_results']
        black_out_nsfw = raw['black_out_nsfw']
        adm_scaler_positive = raw['adm_scaler_positive']
        adm_scaler_negative = raw['adm_scaler_negative']
        adm_scaler_end = raw['adm_scaler_end']
        adaptive_cfg = raw['adaptive_cfg']
        sampler_name = raw['sampler_name']
        scheduler_name = raw['scheduler_name']
        overwrite_step = raw['overwrite_step']
        overwrite_switch = raw['overwrite_switch']
        overwrite_width = raw['overwrite_width']
        overwrite_height = raw['overwrite_height']
        overwrite_vary_strength = raw['overwrite_vary_strength']
        overwrite_upscale_strength = raw['overwrite_upscale_strength']
        mixing_image_prompt_and_vary_upscale = raw['mixing_image_prompt_and_vary_upscale']
        mixing_image_prompt_and_inpaint = raw['mixing_image_prompt_and_inpaint']
        debugging_cn_preprocessor = raw['debugging_cn_preprocessor']
        skipping_cn_preprocessor = raw['skipping_cn_preprocessor']
        canny_low_threshold = raw['canny_low_threshold']
        canny_high_threshold = raw['canny_high_threshold']
        refiner_swap_method = raw['refiner_swap_method']
        controlnet_softness = raw['controlnet_softness']
        freeu_enabled = raw['freeu_enabled']
        freeu_b1 = raw['freeu_b1']
        freeu_b2 = raw['freeu_b2']
        freeu_s1 = raw['freeu_s1']
        freeu_s2 = raw['freeu_s2']
        debugging_inpaint_preprocessor = raw['debugging_inpaint_preprocessor']
        inpaint_disable_initial_latent = raw['inpaint_disable_initial_latent']
        inpaint_engine = raw['inpaint_engine']
        inpaint_strength = raw['inpaint_strength']
        inpaint_respective_field = raw['inpaint_respective_field']
        invert_mask_checkbox = raw['invert_mask_checkbox']
        inpaint_erode_or_dilate = raw['inpaint_erode_or_dilate']

        save_metadata_to_images = raw['save_metadata_to_images'] if not args_manager.args.disable_metadata else False
        metadata_scheme = MetadataScheme(raw['metadata_scheme']) if not args_manager.args.disable_metadata else MetadataScheme.FOOOCUS

        cn_tasks = controlnet_registry.new_task_map()
        for i in range(flags.controlnet_image_count):
            cn_img = raw[f'cn_image_{i + 1}']
            cn_start = raw[f'cn_start_{i + 1}']
            cn_stop = raw[f'cn_stop_{i + 1}']
            cn_weight = raw[f'cn_weight_{i + 1}']
            cn_type = raw[f'cn_type_{i + 1}']
            if cn_img is not None:
                cn_tasks[cn_type].append(controlnet_registry.ControlNetSlot(
                    image=cn_img, start=cn_start, stop=cn_stop, weight=cn_weight, type=cn_type))

        outpaint_selections = [o.lower() for o in outpaint_selections]
        base_model_additional_loras = []

        if base_model_name == refiner_model_name:
            print(f'Refiner disabled because base model and refiner are same.')
            refiner_model_name = 'None'

        if overwrite_step > 0:
            steps = overwrite_step
            print(f'[Parameters] Forced overwrite step to {steps}')

        if steps <= 10:
            print('Auto-switching to LCM mode (1-10 steps).')
            progressbar(async_task, 1, 'Auto-switching to LCM mode (1-10 steps) ...')

            lcm_lora_path = downloader.downloading_sdxl_lcm_lora()

            base_model_additional_loras.append((lcm_lora_path, 1.0))

            sampler_name = 'lcm'
            scheduler_name = 'lcm'
            guidance_scale = 1.0
            sharpness = 0.0
            adaptive_cfg = 1.0
            adm_scaler_positive = 1.0
            adm_scaler_negative = 1.0
            adm_scaler_end = 0.0
        else:
            pass

        if translate_prompts:
            from modules.translator import translate2en
            prompt = translate2en(prompt, 'prompt')
            negative_prompt = translate2en(negative_prompt, 'negative prompt')

        print(f'[Parameters] Adaptive CFG = {adaptive_cfg}')
        print(f'[Parameters] Sharpness = {sharpness}')
        print(f'[Parameters] ControlNet Softness = {controlnet_softness}')
        print(f'[Parameters] ADM Scale = '
              f'{adm_scaler_positive} : '
              f'{adm_scaler_negative} : '
              f'{adm_scaler_end}')

        patch_settings[pid] = PatchSettings(
            sharpness,
            adm_scaler_end,
            adm_scaler_positive,
            adm_scaler_negative,
            controlnet_softness,
            adaptive_cfg
        )

        cfg_scale = float(guidance_scale)
        print(f'[Parameters] CFG = {cfg_scale}')

        initial_latent = None
        denoising_strength = 1.0
        tiled = False

        width, height = aspect_ratios_selection.replace('×', ' ').split(' ')[:2]
        width, height = int(width), int(height)

        if overwrite_width > 0:
            width = overwrite_width
            print(f'[Parameters] Forced overwrite width to {width}')
        if overwrite_height > 0:
            height = overwrite_height
            print(f'[Parameters] Forced overwrite height to {height}')

        skip_prompt_processing = False

        switch = int(round(steps * refiner_switch))
        if overwrite_switch > 0:
            switch = overwrite_switch

        inpaint_worker.current_task = None
        inpaint_parameterized = inpaint_engine != 'None'
        inpaint_image = None
        inpaint_mask = None
        inpaint_head_model_path = None

        use_synthetic_refiner = False

        cn_model_paths = {}

        seed = int(image_seed)
        print(f'[Parameters] Seed = {seed}')

        goals = []
        tasks = []

        if input_image_checkbox:
            inpaint_head_model_path = None
            inpaint_patch_model_path = None

            # --- get switch ---
            switch = int(round(steps * refiner_switch))
            if overwrite_switch > 0:
                switch = overwrite_switch

            if (current_tab == 'uov' or (
                    current_tab == 'ip' and mixing_image_prompt_and_vary_upscale)) \
                    and uov_mode != flags.UOV_MODE_DISABLED and uov_input_image is not None:
                if uov_mode == flags.UOV_MODE_VARY:
                    goals.append('vary')
                    skip_prompt_processing = uov_fast
                elif uov_mode == flags.UOV_MODE_UPSCALE:
                    goals.append('upscale')
                    skip_prompt_processing = uov_fast
                    downloader.downloading_upscale_model()
                else:
                    pass

            if (current_tab == 'inpaint' or (current_tab == 'ip' and mixing_image_prompt_and_inpaint)):
                inpaint_image = None
                inpaint_mask = None

                if isinstance(inpaint_input_image, dict):
                    if 'image' in inpaint_input_image and 'mask' in inpaint_input_image:
                        inpaint_image = inpaint_input_image['image']
                        inpaint_mask = inpaint_input_image['mask']
                        if inpaint_mask.ndim == 3:
                            inpaint_mask = inpaint_mask[:, :, 0]
                    else:
                        bg = inpaint_input_image.get('background')
                        if bg is not None:
                            inpaint_image = bg
                        else:
                            inpaint_image = inpaint_input_image.get('composite')
                            if inpaint_image is None:
                                raise ValueError("No image data in ImageEditor output")

                        if inpaint_image.dtype != np.uint8:
                            if inpaint_image.max() <= 1.0:
                                inpaint_image = (inpaint_image * 255).astype(np.uint8)
                            else:
                                inpaint_image = inpaint_image.astype(np.uint8)
                        if inpaint_image.ndim == 3 and inpaint_image.shape[2] == 4:
                            inpaint_image = inpaint_image[:, :, :3]  # discard alpha
                        elif inpaint_image.ndim == 2:
                            inpaint_image = cv2.cvtColor(inpaint_image, cv2.COLOR_GRAY2RGB)

                        layers = inpaint_input_image.get('layers', [])
                        if len(layers) == 0:
                            print("[Inpaint] No mask layers found, skipping inpaint.")
                            inpaint_mask = np.zeros((inpaint_image.shape[0], inpaint_image.shape[1]), dtype=np.uint8)
                        else:
                            mask_found = False
                            for layer in layers:
                                if layer is None:
                                    continue
                                if layer.dtype != np.uint8:
                                    if layer.max() <= 1.0:
                                        layer = (layer * 255).astype(np.uint8)
                                    else:
                                        layer = layer.astype(np.uint8)
                                if layer.ndim == 3:
                                    if layer.shape[2] == 4:
                                        tmp_mask = layer[:, :, 3]
                                    else:
                                        tmp_mask = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
                                else:
                                    tmp_mask = layer
                                if np.any(tmp_mask > 127):
                                    inpaint_mask = tmp_mask
                                    mask_found = True
                                    break
                            if not mask_found:
                                inpaint_mask = np.zeros((inpaint_image.shape[0], inpaint_image.shape[1]), dtype=np.uint8)
                                print("[Inpaint] No valid mask found in layers.")
                else:
                    inpaint_image = inpaint_input_image
                    inpaint_mask = np.zeros((inpaint_image.shape[0], inpaint_image.shape[1]), dtype=np.uint8)

                if inpaint_mask is None:
                    inpaint_mask = np.zeros((inpaint_image.shape[0], inpaint_image.shape[1]), dtype=np.uint8)
                else:
                    if inpaint_mask.dtype != np.uint8:
                        if inpaint_mask.max() <= 1.0:
                            inpaint_mask = (inpaint_mask * 255).astype(np.uint8)
                        else:
                            inpaint_mask = inpaint_mask.astype(np.uint8)
                    inpaint_mask = (inpaint_mask > 127).astype(np.uint8) * 255

                    # 顺序与 UI 提示一致：先腐蚀/膨胀，再取反
                    if inpaint_erode_or_dilate != 0:
                        inpaint_mask = erode_or_dilate(inpaint_mask, inpaint_erode_or_dilate)

                    if invert_mask_checkbox:
                        inpaint_mask = 255 - inpaint_mask

                inpaint_image = HWC3(inpaint_image)

                if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                        and (np.any(inpaint_mask > 127) or len(outpaint_selections) > 0):
                    downloader.downloading_upscale_model()
                    goals.append('inpaint')

                    if inpaint_parameterized:
                        inpaint_head_model_path, inpaint_patch_model_path = downloader.downloading_inpaint_models(inpaint_engine)
                        base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                        print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
                        if refiner_model_name == 'None':
                            use_synthetic_refiner = True
                            refiner_switch = 0.5
                    else:
                        print(f'[Inpaint] Parameterized inpaint is disabled.')
                else:
                    pass

                if inpaint_additional_prompt != '':
                    if prompt == '':
                        prompt = inpaint_additional_prompt
                    else:
                        prompt = inpaint_additional_prompt + '\n' + prompt

            if current_tab == 'ip' or \
                    mixing_image_prompt_and_vary_upscale or \
                    mixing_image_prompt_and_inpaint:
                goals.append('cn')
                for cn_type in controlnet_registry.TYPES:
                    if len(cn_tasks[cn_type.name]) > 0:
                        cn_model_paths[cn_type.name] = cn_type.load_model()
                progressbar(async_task, 1, 'Loading control models ...')

            controlnet_paths = [cn_model_paths[t.name] for t in controlnet_registry.TYPES
                                if t.feeds_controlnet_pipeline and t.name in cn_model_paths]
            if controlnet_paths:
                pipeline.refresh_controlnets(controlnet_paths)

        if not skip_prompt_processing:
            prompts = remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')
            negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.splitlines()], default='')

            if uov_ignore_prompt and input_image_checkbox and uov_input_image is not None:
                prompts = ['']
                negative_prompts = ['']

            prompt = prompts[0]
            negative_prompt = negative_prompts[0]

            extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
            extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

            progressbar(async_task, 3, 'Loading models ...')
            pipeline.refresh_everything(
                refiner_model_name=refiner_model_name,
                base_model_name=base_model_name,
                loras=loras,
                base_model_additional_loras=base_model_additional_loras,
                use_synthetic_refiner=use_synthetic_refiner
            )

            progressbar(async_task, 3, 'Processing prompts ...')
            tasks = []

            for i in range(image_number):
                task_seed = (seed + i) % (constants.MAX_SEED + 1)

                task_prompt = prompt
                task_negative_prompt = negative_prompt
                task_extra_positive_prompts = extra_positive_prompts
                task_extra_negative_prompts = extra_negative_prompts

                positive_basic_workloads = []
                negative_basic_workloads = []

                positive_basic_workloads.append(task_prompt)
                negative_basic_workloads.append(task_negative_prompt)

                positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
                negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

                positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
                negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

                tasks.append(dict(
                    task_seed=task_seed,
                    task_prompt=task_prompt,
                    task_negative_prompt=task_negative_prompt,
                    positive=positive_basic_workloads,
                    negative=negative_basic_workloads,
                    c=None,
                    uc=None,
                    positive_top_k=len(positive_basic_workloads),
                    negative_top_k=len(negative_basic_workloads),
                    log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
                    log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
                ))

            for i, t in enumerate(tasks):
                progressbar(async_task, 7, f'Encoding positive #{i + 1} ...')
                t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])

            for i, t in enumerate(tasks):
                if abs(float(cfg_scale) - 1.0) < 1e-4:
                    t['uc'] = pipeline.clone_cond(t['c'])
                else:
                    progressbar(async_task, 10, f'Encoding negative #{i + 1} ...')
                    t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])

        if 'upscale' in goals:
            H, W, C = uov_input_image.shape
            progressbar(async_task, 13, f'Upscaling image from {str((H, W))} ...')
            uov_input_image = perform_upscale(uov_input_image)
            print(f'Image upscaled.')
            f = uov_scale

            target_width = int(round(W * f))
            target_height = int(round(H * f))
            target_width = math.ceil(target_width / 64) * 64
            target_height = math.ceil(target_height / 64) * 64
            shape_ceil = get_shape_ceil(target_height, target_width)
            uov_input_image = resample_image(uov_input_image, width=target_width, height=target_height)

            image_is_super_large = shape_ceil > 2800

            if uov_fast:
                direct_return = True
                fast_mode_reason = 'manual'
            elif image_is_super_large:
                print('Image is too large. Directly returned the SR image. '
                      'Usually directly return SR image at 4K resolution '
                      'yields better results than SDXL diffusion.')
                direct_return = True
                fast_mode_reason = 'auto_oversize'
            else:
                direct_return = False
                fast_mode_reason = None

            if direct_return:
                if fast_mode_reason == 'auto_oversize':
                    async_task.yields.append(['preview', (13, 'Image too large, auto-switch to Fast Mode...', None)])

                d = [('Upscale (Fast)', f'{f}x')]
                uov_input_image_path = log(uov_input_image, d, output_format=output_format)

                if fast_mode_reason == 'manual':
                    progress_text = 'Upscale (Fast) completed'
                else:
                    progress_text = 'Auto Fast Mode completed'
                async_task.yields.append(['preview', (100, progress_text, None)])

                async_task.results = [uov_input_image_path]
                async_task.yields.append(['results', async_task.results])

                async_task.yields.append(['finish', async_task.results])
                async_task.processing = False
                return

            tiled = not uov_fast
            denoising_strength = uov_denoise
            if overwrite_upscale_strength > 0:
                denoising_strength = overwrite_upscale_strength

            progressbar(async_task, 13, 'VAE encoding ...')

            initial_latent, _, _, width, height = pipeline.prepare_img2img_latent(
                uov_input_image,
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method,
                tiled=True
            )
            print(f'Final resolution is {str((height, width))}.')

        if 'vary' in goals:
            denoising_strength = uov_denoise
            if overwrite_vary_strength > 0:
                denoising_strength = overwrite_vary_strength

            shape_ceil = get_image_shape_ceil(uov_input_image)
            if shape_ceil < 1024:
                print(f'[Vary] Image is resized because it is too small.')
                shape_ceil = 1024
            elif shape_ceil > 2048:
                print(f'[Vary] Image is resized because it is too big.')
                shape_ceil = 2048

            uov_input_image = set_image_shape_ceil(uov_input_image, shape_ceil)

            progressbar(async_task, 13, 'VAE encoding ...')

            initial_latent, _, _, width, height = pipeline.prepare_img2img_latent(
                uov_input_image,
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )
            print(f'Final resolution is {str((height, width))}.')

        if 'inpaint' in goals:
            # OUTPAINT: waiting for refactor (function removed, keep codes for reference in future)
            # if len(outpaint_selections) > 0:
            #     H, W, C = inpaint_image.shape
            #     if 'top' in outpaint_selections:
            #         inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
            #         inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant',
            #                               constant_values=255)
            #     if 'bottom' in outpaint_selections:
            #         inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
            #         inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant',
            #                               constant_values=255)

            #     H, W, C = inpaint_image.shape
            #     if 'left' in outpaint_selections:
            #         inpaint_image = np.pad(inpaint_image, [[0, 0], [int(H * 0.3), 0], [0, 0]], mode='edge')
            #         inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(H * 0.3), 0]], mode='constant',
            #                               constant_values=255)
            #     if 'right' in outpaint_selections:
            #         inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(H * 0.3)], [0, 0]], mode='edge')
            #         inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(H * 0.3)]], mode='constant',
            #                               constant_values=255)

            #     inpaint_image = np.ascontiguousarray(inpaint_image.copy())
            #     inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())
            #     inpaint_strength = 1.0
            #     inpaint_respective_field = 1.0

            denoising_strength = inpaint_strength

            inpaint_worker.current_task = inpaint_worker.InpaintWorker(
                image=inpaint_image,
                mask=inpaint_mask,
                use_fill=denoising_strength > 0.99,
                k=inpaint_respective_field
            )

            if debugging_inpaint_preprocessor:
                yield_result(async_task, inpaint_worker.current_task.visualize_mask_processing(), black_out_nsfw,
                             do_not_show_finished_images=True)
                return

            progressbar(async_task, 13, 'VAE Inpaint encoding ...')

            inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
            inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
            inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)

            candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            latent_inpaint, latent_mask = core.encode_vae_inpaint(
                mask=inpaint_pixel_mask,
                vae=candidate_vae,
                pixels=inpaint_pixel_image)

            latent_swap = None
            if candidate_vae_swap is not None:
                progressbar(async_task, 13, 'VAE SD15 encoding ...')
                latent_swap = core.encode_vae(
                    vae=candidate_vae_swap,
                    pixels=inpaint_pixel_fill)['samples']

            progressbar(async_task, 13, 'VAE encoding ...')
            latent_fill = core.encode_vae(
                vae=candidate_vae,
                pixels=inpaint_pixel_fill)['samples']

            inpaint_worker.current_task.load_latent(
                latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)

            if inpaint_parameterized:
                pipeline.final_unet = inpaint_worker.current_task.patch(
                    inpaint_head_model_path=inpaint_head_model_path,
                    inpaint_latent=latent_inpaint,
                    inpaint_latent_mask=latent_mask,
                    model=pipeline.final_unet
                )

            if not inpaint_disable_initial_latent:
                initial_latent = {'samples': latent_fill}

            B, C, H, W = latent_fill.shape
            height, width = H * 8, W * 8
            final_height, final_width = inpaint_worker.current_task.image.shape[:2]
            print(f'Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')

        if 'cn' in goals:
            # 一次遍历覆盖所有类型：预处理本身由注册表里各类型的 preprocess 负责
            for cn_type in controlnet_registry.TYPES:
                if not cn_tasks[cn_type.name]:
                    continue

                ctx = controlnet_registry.PreprocessContext(
                    width=width,
                    height=height,
                    skipping_preprocessor=skipping_cn_preprocessor,
                    canny_low_threshold=canny_low_threshold,
                    canny_high_threshold=canny_high_threshold,
                    model_path=cn_model_paths.get(cn_type.name),
                )

                for slot in cn_tasks[cn_type.name]:
                    slot.image, display_image = cn_type.preprocess(slot.image, ctx)
                    if debugging_cn_preprocessor:
                        yield_result(async_task, display_image, black_out_nsfw, do_not_show_finished_images=True)
                        return

            all_ip_tasks = [slot for cn_type in controlnet_registry.unet_patch_types()
                            for slot in cn_tasks[cn_type.name]]

            if len(all_ip_tasks) > 0:
                pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)

        if len(goals) > 0:
            progressbar(async_task, 13, 'Image processing ...')

        if freeu_enabled:
            print(f'FreeU is enabled!')
            pipeline.final_unet = core.apply_freeu(
                pipeline.final_unet,
                freeu_b1,
                freeu_b2,
                freeu_s1,
                freeu_s2
            )

        all_steps = steps * image_number

        print(f'[Parameters] Denoising Strength = {denoising_strength}')

        if isinstance(initial_latent, dict) and 'samples' in initial_latent:
            log_shape = initial_latent['samples'].shape
        else:
            log_shape = f'Image Space {(height, width)}'

        print(f'[Parameters] Initial Latent shape: {log_shape}')

        preparation_time = time.perf_counter() - execution_start_time
        print(f'Preparation time: {preparation_time:.2f} seconds')

        final_sampler_name = sampler_name
        final_scheduler_name = scheduler_name

        if scheduler_name == 'lcm':
            final_scheduler_name = 'sgm_uniform'
            if pipeline.final_unet is not None:
                pipeline.final_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_refiner_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            print('Using lcm scheduler.')

        async_task.yields.append(['preview', (13, 'Moving model to GPU ...', None)])

        def callback(step, x0, x, total_steps, y):
            done_steps = current_task_id * steps + step
            if step % 3 == 0 or step == total_steps - 1:
                async_task.yields.append(['preview', (
                    int(15.0 + 85.0 * float(done_steps) / float(all_steps)),
                    f'Sampling Image {current_task_id + 1}/{image_number}, Step {step + 1}/{total_steps} ...', y)])

        for current_task_id, task in enumerate(tasks):
            execution_start_time = time.perf_counter()

            try:
                if async_task.last_stop is not False:
                    ldm_patched.model_management.interrupt_current_processing()
                positive_cond, negative_cond = task['c'], task['uc']

                if 'cn' in goals:
                    for cn_type in controlnet_registry.conditioning_types():
                        cn_path = cn_model_paths.get(cn_type.name)
                        for slot in cn_tasks[cn_type.name]:
                            positive_cond, negative_cond = core.apply_controlnet(
                                positive_cond, negative_cond,
                                pipeline.loaded_ControlNets[cn_path], slot.image, slot.weight, slot.start, slot.stop)

                imgs = pipeline.process_diffusion(
                    positive_cond=positive_cond,
                    negative_cond=negative_cond,
                    steps=steps,
                    switch=switch,
                    width=width,
                    height=height,
                    image_seed=task['task_seed'],
                    callback=callback,
                    sampler_name=final_sampler_name,
                    scheduler_name=final_scheduler_name,
                    latent=initial_latent,
                    denoise=denoising_strength,
                    tiled=tiled,
                    cfg_scale=cfg_scale,
                    refiner_swap_method=refiner_swap_method,
                    disable_preview=disable_preview
                )

                del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

                if inpaint_worker.current_task is not None:
                    imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

                img_paths = []

                for x in imgs:
                    d = [('Prompt', 'prompt', task['log_positive_prompt']),
                         ('Negative Prompt', 'negative_prompt', task['log_negative_prompt']),
                         ('Steps', 'steps', steps),
                         ('Resolution', 'resolution', str((width, height))),
                         ('Guidance Scale', 'guidance_scale', guidance_scale),
                         ('Sharpness', 'sharpness', modules.patch.patch_settings[pid].sharpness),
                         ('ADM Guidance', 'adm_guidance', str((
                             modules.patch.patch_settings[pid].positive_adm_scale,
                             modules.patch.patch_settings[pid].negative_adm_scale,
                             modules.patch.patch_settings[pid].adm_scaler_end))),
                         ('Base Model', 'base_model', base_model_name),
                         ('Refiner Model', 'refiner_model', refiner_model_name),
                         ('Refiner Switch', 'refiner_switch', refiner_switch)]

                    if refiner_model_name != 'None':
                        if overwrite_switch > 0:
                            d.append(('Overwrite Switch', 'overwrite_switch', overwrite_switch))
                        if refiner_swap_method != flags.refiner_swap_method:
                            d.append(('Refiner Swap Method', 'refiner_swap_method', refiner_swap_method))
                    if modules.patch.patch_settings[pid].adaptive_cfg != modules.config.default_cfg_tsnr:
                        d.append(('CFG Mimicking from TSNR', 'adaptive_cfg', modules.patch.patch_settings[pid].adaptive_cfg))

                    d.append(('Sampler', 'sampler', sampler_name))
                    d.append(('Scheduler', 'scheduler', scheduler_name))
                    d.append(('Seed', 'seed', task['task_seed']))

                    if freeu_enabled:
                        d.append(('FreeU', 'freeu', str((freeu_b1, freeu_b2, freeu_s1, freeu_s2))))

                    metadata_parser = None
                    if save_metadata_to_images:
                        metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                        metadata_parser.set_data(task['log_positive_prompt'], task['positive'],
                                                 task['log_negative_prompt'], task['negative'],
                                                 steps, base_model_name, refiner_model_name, loras)

                    for li, (n, w) in enumerate(loras):
                        if n != 'None':
                            d.append((f'LoRA {li + 1}', f'lora_combined_{li + 1}', f'{n} : {w}'))

                    d.append(('Version', 'version', 'ReFocus v' + ReFocus_version.version))
                    img_paths.append(log(x, d, metadata_parser, output_format))

                yield_result(async_task, img_paths, black_out_nsfw, do_not_show_finished_images=len(tasks) == 1
                             or disable_intermediate_results or sampler_name == 'lcm',
                             progressbar_index=int(15.0 + 85.0 * float((current_task_id + 1) * steps) / float(all_steps)))
            except ldm_patched.modules.model_management.InterruptProcessingException as e:
                if async_task.last_stop == 'skip':
                    print('User skipped')
                    async_task.last_stop = False
                    continue
                else:
                    print('User stopped')
                    break

            execution_time = time.perf_counter() - execution_start_time
            print(f'Generating and saving time: {execution_time:.2f} seconds')
        async_task.processing = False
        return

    while True:
        time.sleep(0.01)
        if len(async_tasks) > 0:
            task = async_tasks.pop(0)

            if not task.args:  # prevent empty task causing pop error
                print("[Warning] async_worker got empty task.args, skip this task.")
                continue

            try:
                handler(task)
                # handler has add finish flag and set processing = False
                # double protection, but will not be executed, handler will return
                task.yields.append(['finish', task.results])
                pipeline.prepare_text_encoder(async_call=True)
            except Exception as e:
                traceback.print_exc()
                
                error_msg = str(e)
                
                if "CUDA out of memory" in error_msg or "out of memory" in error_msg.lower():
                    friendly_msg = "Out of memory. Try reducing resolution, batch size, or disabling ControlNet."
                elif "FileNotFoundError" in str(type(e)) or "No such file" in error_msg:
                    friendly_msg = "Model file not found. Check path or re-download."
                else:
                    friendly_msg = f"Generation failed: {error_msg}"
                
                # check finish flag
                has_finish = any(flag == 'finish' for flag, _ in task.yields)
                
                if not has_finish:
                    # handler fail: clean up result, add finish
                    task.yields.append(['preview', (100, f'Error: {friendly_msg}', None)])
                    task.yields.append(['finish', []])
                    task.results = []
                else:
                    # handler done, following steps fail: keep results, add error msg
                    task.yields.append(['preview', (100, f'Error in post-processing: {friendly_msg}', None)])
                
                task.processing = False
                
            finally:
                if pid in modules.patch.patch_settings:
                    del modules.patch.patch_settings[pid]
    pass


threading.Thread(target=worker, daemon=True).start()
