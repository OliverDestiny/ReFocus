import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
import modules.flags as flags

import gradio as gr
from PIL import Image

import ReFocus_version
import modules.config
from modules.flags import MetadataScheme
from modules.flags import lora_count, SAMPLERS, CIVITAI_NO_KARRAS
from modules.util import quote, unquote, is_json, calculate_sha256

re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_imagesize = re.compile(r"^(\d+)x(\d+)$")

hash_cache = {}


def load_parameter_button_click(raw_metadata: dict | str, is_generating: bool):
    loaded_parameter_dict = raw_metadata
    if isinstance(raw_metadata, str):
        loaded_parameter_dict = json.loads(raw_metadata)
    assert isinstance(loaded_parameter_dict, dict)

    # 输出顺序必须与 webui.py 中 load_data_outputs 的顺序完全一致
    # [0] advanced_checkbox
    # [1] image_number
    # [2] prompt
    # [3] negative_prompt
    # [4] steps_slider
    # [5] overwrite_step
    # [6] overwrite_switch
    # [7] aspect_ratios_selection
    # [8] overwrite_width
    # [9] overwrite_height
    # [10] guidance_scale
    # [11] sharpness
    # [12] adm_scaler_positive
    # [13] adm_scaler_negative
    # [14] adm_scaler_end
    # [15] refiner_swap_method
    # [16] adaptive_cfg
    # [17] base_model
    # [18] refiner_model
    # [19] refiner_switch
    # [20] sampler_name
    # [21] scheduler_name
    # [22] seed_random
    # [23] image_seed
    # [24] generate_button
    # [25] load_parameter_button
    # [26] freeu_enabled
    # [27] freeu_b1
    # [28] freeu_b2
    # [29] freeu_s1
    # [30] freeu_s2
    # [31..] lora_combined (pairs)

    results = []

    # 辅助函数：获取值，若不存在则返回默认
    def get_value(key, fallback=None, default=None):
        v = loaded_parameter_dict.get(key, loaded_parameter_dict.get(fallback, default))
        return v

    # 1. advanced_checkbox (bool)
    adv = get_value('advanced_checkbox', 'Advanced Checkbox', modules.config.default_advanced_checkbox)
    results.append(adv if isinstance(adv, bool) else gr.update())

    # 2. image_number (int)
    img_num = get_value('image_number', 'Image Number', modules.config.default_image_number)
    try:
        results.append(int(img_num))
    except:
        results.append(gr.update())

    # 3. prompt (str)
    prompt = get_value('prompt', 'Prompt', '')
    results.append(prompt if isinstance(prompt, str) else gr.update())

    # 4. negative_prompt (str)
    neg_prompt = get_value('negative_prompt', 'Negative Prompt', '')
    results.append(neg_prompt if isinstance(neg_prompt, str) else gr.update())

    # 5. steps_slider (int)  直接从 'steps' 读取
    steps_val = get_value('steps', 'Steps', 25)
    try:
        steps_val = int(steps_val)
        if steps_val < 1: steps_val = 1
        if steps_val > 50: steps_val = 50
        results.append(steps_val)
    except:
        results.append(gr.update())

    # 6. overwrite_step (int)  默认 -1
    overwrite_step = get_value('overwrite_step', 'Overwrite Step', -1)
    try:
        results.append(int(overwrite_step))
    except:
        results.append(gr.update())

    # 7. overwrite_switch (float)
    overwrite_switch = get_value('overwrite_switch', 'Overwrite Switch', -1.0)
    try:
        results.append(float(overwrite_switch))
    except:
        results.append(gr.update())

    # 8-10. resolution -> aspect_ratios_selection, overwrite_width, overwrite_height
    res = get_value('resolution', 'Resolution', None)
    if res is not None:
        try:
            width, height = eval(res)
            formatted = modules.config.add_ratio(f'{width}*{height}')
            if formatted in modules.config.available_aspect_ratios:
                results.append(formatted)
                results.append(-1)
                results.append(-1)
            else:
                results.append(gr.update())
                results.append(width)
                results.append(height)
        except:
            results.append(gr.update())
            results.append(gr.update())
            results.append(gr.update())
    else:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())

    # 11. guidance_scale (float)
    guidance = get_value('guidance_scale', 'Guidance Scale', modules.config.default_cfg_scale)
    try:
        results.append(float(guidance))
    except:
        results.append(gr.update())

    # 12. sharpness (float)
    sharp = get_value('sharpness', 'Sharpness', modules.config.default_sample_sharpness)
    try:
        results.append(float(sharp))
    except:
        results.append(gr.update())

    # 13-15. adm_guidance (3 floats)
    adm = get_value('adm_guidance', 'ADM Guidance', None)
    if adm is not None:
        try:
            p, n, e = eval(adm)
            results.append(float(p))
            results.append(float(n))
            results.append(float(e))
        except:
            results.append(gr.update())
            results.append(gr.update())
            results.append(gr.update())
    else:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())

    # 16. refiner_swap_method (str)
    refiner_swap = get_value('refiner_swap_method', 'Refiner Swap Method', flags.refiner_swap_method)
    results.append(refiner_swap if isinstance(refiner_swap, str) else gr.update())

    # 17. adaptive_cfg (float)
    adaptive = get_value('adaptive_cfg', 'CFG Mimicking from TSNR', modules.config.default_cfg_tsnr)
    try:
        results.append(float(adaptive))
    except:
        results.append(gr.update())

    # 18. base_model (str)
    base = get_value('base_model', 'Base Model', '')
    results.append(base if isinstance(base, str) else gr.update())

    # 19. refiner_model (str)
    refiner = get_value('refiner_model', 'Refiner Model', 'None')
    results.append(refiner if isinstance(refiner, str) else gr.update())

    # 20. refiner_switch (float)
    refiner_sw = get_value('refiner_switch', 'Refiner Switch', modules.config.default_refiner_switch)
    try:
        results.append(float(refiner_sw))
    except:
        results.append(gr.update())

    # 21. sampler_name (str)
    sampler = get_value('sampler', 'Sampler', modules.config.default_sampler)
    results.append(sampler if isinstance(sampler, str) else gr.update())

    # 22. scheduler_name (str)
    scheduler = get_value('scheduler', 'Scheduler', modules.config.default_scheduler)
    results.append(scheduler if isinstance(scheduler, str) else gr.update())

    # 23. seed_random (bool)
    seed_random_val = get_value('seed_random', 'Randomize seed', True)
    results.append(seed_random_val if isinstance(seed_random_val, bool) else gr.update())

    # 24. image_seed (int)
    seed_val = get_value('seed', 'Seed', 0)
    try:
        results.append(int(seed_val))
    except:
        results.append(gr.update())

    if is_generating:
        results.append(gr.update())
    else:
        results.append(gr.update(visible=True))

    results.append(gr.update(visible=False))

    freeu_data = get_value('freeu', 'FreeU', None)
    if freeu_data is not None:
        try:
            b1, b2, s1, s2 = eval(freeu_data)
            results.append(True)
            results.append(float(b1))
            results.append(float(b2))
            results.append(float(s1))
            results.append(float(s2))
        except:
            results.append(False)
            results.append(gr.update())
            results.append(gr.update())
            results.append(gr.update())
            results.append(gr.update())
    else:
        results.append(False)
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())

    # 32-41. lora pairs (5 pairs)
    for i in range(lora_count):
        key = f'lora_combined_{i+1}'
        fallback = f'LoRA {i+1}'
        lora_val = get_value(key, fallback, None)
        if lora_val is not None:
            try:
                n, w = lora_val.split(' : ')
                w = float(w)
                results.append(n)
                results.append(w)
            except:
                results.append('None')
                results.append(1.0)
        else:
            results.append('None')
            results.append(1.0)

    return results


def get_str(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())


def get_list(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        h = eval(h)
        assert isinstance(h, list)
        results.append(h)
    except:
        results.append(gr.update())


def get_float(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = float(h)
        results.append(h)
    except:
        results.append(gr.update())


def get_resolution(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        width, height = eval(h)
        formatted = modules.config.add_ratio(f'{width}*{height}')
        if formatted in modules.config.available_aspect_ratios:
            results.append(formatted)
            results.append(-1)
            results.append(-1)
        else:
            results.append(gr.update())
            results.append(width)
            results.append(height)
    except:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_seed(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        results.append(False)
        results.append(h)
    except:
        results.append(gr.update())
        results.append(gr.update())


def get_adm_guidance(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        p, n, e = eval(h)
        results.append(float(p))
        results.append(float(n))
        results.append(float(e))
    except:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_freeu(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        b1, b2, s1, s2 = eval(h)
        results.append(True)
        results.append(float(b1))
        results.append(float(b2))
        results.append(float(s1))
        results.append(float(s2))
    except:
        results.append(False)
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_lora(key: str, fallback: str | None, source_dict: dict, results: list):
    try:
        n, w = source_dict.get(key, source_dict.get(fallback)).split(' : ')
        w = float(w)
        results.append(n)
        results.append(w)
    except:
        results.append('None')
        results.append(1)


def get_sha256(filepath):
    global hash_cache
    if filepath not in hash_cache:
        hash_cache[filepath] = calculate_sha256(filepath)

    return hash_cache[filepath]


def parse_meta_from_preset(preset_content):
    assert isinstance(preset_content, dict)
    preset_prepared = {}
    items = preset_content

    for settings_key, meta_key in modules.config.possible_preset_keys.items():
        if settings_key == "default_loras":
            loras = getattr(modules.config, settings_key)
            if settings_key in items:
                loras = items[settings_key]
            for index, lora in enumerate(loras[:5]):
                preset_prepared[f'lora_combined_{index + 1}'] = ' : '.join(map(str, lora))
        elif settings_key == "default_aspect_ratio":
            if settings_key in items and items[settings_key] is not None:
                default_aspect_ratio = items[settings_key]
                width, height = default_aspect_ratio.split('*')
            else:
                default_aspect_ratio = getattr(modules.config, settings_key)
                width, height = default_aspect_ratio.split('×')
                height = height[:height.index(" ")]
            preset_prepared[meta_key] = (width, height)
            preset_prepared[meta_key] = str(preset_prepared[meta_key])
        else:
            preset_prepared[meta_key] = (
                items[settings_key]
                if settings_key in items and items[settings_key] is not None
                else getattr(modules.config, settings_key)
            )

    return preset_prepared


class MetadataParser(ABC):
    def __init__(self):
        self.raw_prompt: str = ''
        self.full_prompt: str = ''
        self.raw_negative_prompt: str = ''
        self.full_negative_prompt: str = ''
        self.steps: int = 30
        self.base_model_name: str = ''
        self.base_model_hash: str = ''
        self.refiner_model_name: str = ''
        self.refiner_model_hash: str = ''
        self.loras: list = []

    @abstractmethod
    def get_scheme(self) -> MetadataScheme:
        raise NotImplementedError

    @abstractmethod
    def parse_json(self, metadata: dict | str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def parse_string(self, metadata: dict) -> str:
        raise NotImplementedError

    def set_data(self, raw_prompt, full_prompt, raw_negative_prompt, full_negative_prompt, steps, base_model_name, refiner_model_name, loras):
        self.raw_prompt = raw_prompt
        self.full_prompt = full_prompt
        self.raw_negative_prompt = raw_negative_prompt
        self.full_negative_prompt = full_negative_prompt
        self.steps = steps
        self.base_model_name = Path(base_model_name).stem

        base_model_path = os.path.join(modules.config.path_checkpoints, base_model_name)
        self.base_model_hash = get_sha256(base_model_path)

        if refiner_model_name not in ['', 'None']:
            self.refiner_model_name = Path(refiner_model_name).stem
            refiner_model_path = os.path.join(modules.config.path_checkpoints, refiner_model_name)
            self.refiner_model_hash = get_sha256(refiner_model_path)

        self.loras = []
        for (lora_name, lora_weight) in loras:
            if lora_name != 'None':
                lora_path = os.path.join(modules.config.path_loras, lora_name)
                lora_hash = get_sha256(lora_path)
                self.loras.append((Path(lora_name).stem, lora_weight, lora_hash))


class A1111MetadataParser(MetadataParser):
    def get_scheme(self) -> MetadataScheme:
        return MetadataScheme.A1111

    fooocus_to_a1111 = {
        'raw_prompt': 'Raw prompt',
        'raw_negative_prompt': 'Raw negative prompt',
        'negative_prompt': 'Negative prompt',
        'steps': 'Steps',
        'sampler': 'Sampler',
        'scheduler': 'Scheduler',
        'guidance_scale': 'CFG scale',
        'seed': 'Seed',
        'resolution': 'Size',
        'sharpness': 'Sharpness',
        'adm_guidance': 'ADM Guidance',
        'refiner_swap_method': 'Refiner Swap Method',
        'adaptive_cfg': 'Adaptive CFG',
        'overwrite_switch': 'Overwrite Switch',
        'freeu': 'FreeU',
        'base_model': 'Model',
        'base_model_hash': 'Model hash',
        'refiner_model': 'Refiner',
        'refiner_model_hash': 'Refiner hash',
        'lora_hashes': 'Lora hashes',
        'lora_weights': 'Lora weights',
        'created_by': 'User',
        'version': 'Version'
    }

    def parse_json(self, metadata: str) -> dict:
        metadata_prompt = ''
        metadata_negative_prompt = ''

        done_with_prompt = False

        *lines, lastline = metadata.strip().split("\n")
        if len(re_param.findall(lastline)) < 3:
            lines.append(lastline)
            lastline = ''

        for line in lines:
            line = line.strip()
            if line.startswith(f"{self.fooocus_to_a1111['negative_prompt']}:"):
                done_with_prompt = True
                line = line[len(f"{self.fooocus_to_a1111['negative_prompt']}:"):].strip()
            if done_with_prompt:
                metadata_negative_prompt += ('' if metadata_negative_prompt == '' else "\n") + line
            else:
                metadata_prompt += ('' if metadata_prompt == '' else "\n") + line

        prompt = metadata_prompt
        negative_prompt = metadata_negative_prompt

        data = {
            'prompt': prompt,
            'negative_prompt': negative_prompt
        }

        for k, v in re_param.findall(lastline):
            try:
                if v[0] == '"' and v[-1] == '"':
                    v = unquote(v)

                m = re_imagesize.match(v)
                if m is not None:
                    data['resolution'] = str((m.group(1), m.group(2)))
                else:
                    data[list(self.fooocus_to_a1111.keys())[list(self.fooocus_to_a1111.values()).index(k)]] = v
            except Exception:
                print(f"Error parsing \"{k}: {v}\"")

        # workaround for multiline prompts
        if 'raw_prompt' in data:
            data['prompt'] = data['raw_prompt']

        if 'raw_negative_prompt' in data:
            data['negative_prompt'] = data['raw_negative_prompt']

        if 'sampler' in data:
            data['sampler'] = data['sampler'].replace(' Karras', '')
            for k, v in SAMPLERS.items():
                if v == data['sampler']:
                    data['sampler'] = k
                    break

        for key in ['base_model', 'refiner_model']:
            if key in data:
                for filename in modules.config.model_filenames:
                    path = Path(filename)
                    if data[key] == path.stem:
                        data[key] = filename
                        break

        if 'lora_hashes' in data:
            lora_filenames = modules.config.lora_filenames.copy()
            lora_filenames.remove(flags.LCM_LORA_FILENAME)
            for li, lora in enumerate(data['lora_hashes'].split(', ')):
                lora_name, lora_hash, lora_weight = lora.split(': ')
                for filename in lora_filenames:
                    path = Path(filename)
                    if lora_name == path.stem:
                        data[f'lora_combined_{li + 1}'] = f'{filename} : {lora_weight}'
                        break

        return data

    def parse_string(self, metadata: dict) -> str:
        data = {k: v for _, k, v in metadata}

        width, height = eval(data['resolution'])

        sampler = data['sampler']
        scheduler = data['scheduler']
        if sampler in SAMPLERS and SAMPLERS[sampler] != '':
            sampler = SAMPLERS[sampler]
            if sampler not in CIVITAI_NO_KARRAS and scheduler == 'karras':
                sampler += f' Karras'

        generation_params = {
            self.fooocus_to_a1111['steps']: self.steps,
            self.fooocus_to_a1111['sampler']: sampler,
            self.fooocus_to_a1111['seed']: data['seed'],
            self.fooocus_to_a1111['resolution']: f'{width}x{height}',
            self.fooocus_to_a1111['guidance_scale']: data['guidance_scale'],
            self.fooocus_to_a1111['sharpness']: data['sharpness'],
            self.fooocus_to_a1111['adm_guidance']: data['adm_guidance'],
            self.fooocus_to_a1111['base_model']: Path(data['base_model']).stem,
            self.fooocus_to_a1111['base_model_hash']: self.base_model_hash,

            self.fooocus_to_a1111['scheduler']: scheduler,
            # workaround for multiline prompts
            self.fooocus_to_a1111['raw_prompt']: self.raw_prompt,
            self.fooocus_to_a1111['raw_negative_prompt']: self.raw_negative_prompt,
        }

        if self.refiner_model_name not in ['', 'None']:
            generation_params |= {
                self.fooocus_to_a1111['refiner_model']: self.refiner_model_name,
                self.fooocus_to_a1111['refiner_model_hash']: self.refiner_model_hash
            }

        for key in ['adaptive_cfg', 'overwrite_switch', 'refiner_swap_method', 'freeu']:
            if key in data:
                generation_params[self.fooocus_to_a1111[key]] = data[key]

        lora_hashes = []
        for index, (lora_name, lora_weight, lora_hash) in enumerate(self.loras):
            # workaround for Fooocus not knowing LoRA name in LoRA metadata
            lora_hashes.append(f'{lora_name}: {lora_hash}: {lora_weight}')
        lora_hashes_string = ', '.join(lora_hashes)

        generation_params |= {
            self.fooocus_to_a1111['lora_hashes']: lora_hashes_string,
            self.fooocus_to_a1111['version']: data['version']
        }

        if modules.config.metadata_created_by != '':
            generation_params[self.fooocus_to_a1111['created_by']] = modules.config.metadata_created_by

        generation_params_text = ", ".join(
            [k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if
             v is not None])
        positive_prompt_resolved = ', '.join(self.full_prompt)
        negative_prompt_resolved = ', '.join(self.full_negative_prompt)
        negative_prompt_text = f"\nNegative prompt: {negative_prompt_resolved}" if negative_prompt_resolved else ""
        return f"{positive_prompt_resolved}{negative_prompt_text}\n{generation_params_text}".strip()


class FooocusMetadataParser(MetadataParser):
    def get_scheme(self) -> MetadataScheme:
        return MetadataScheme.FOOOCUS

    def parse_json(self, metadata: dict) -> dict:
        model_filenames = modules.config.model_filenames.copy()
        lora_filenames = modules.config.lora_filenames.copy()
        lora_filenames.remove(flags.LCM_LORA_FILENAME)

        for key, value in metadata.items():
            if value in ['', 'None']:
                continue
            if key in ['base_model', 'refiner_model']:
                metadata[key] = self.replace_value_with_filename(key, value, model_filenames)
            elif key.startswith('lora_combined_'):
                metadata[key] = self.replace_value_with_filename(key, value, lora_filenames)
            else:
                continue

        return metadata

    def parse_string(self, metadata: list) -> str:
        for li, (label, key, value) in enumerate(metadata):
            # remove model folder paths from metadata
            if key.startswith('lora_combined_'):
                name, weight = value.split(' : ')
                name = Path(name).stem
                value = f'{name} : {weight}'
                metadata[li] = (label, key, value)

        res = {k: v for _, k, v in metadata}

        res['full_prompt'] = self.full_prompt
        res['full_negative_prompt'] = self.full_negative_prompt
        res['steps'] = self.steps
        res['base_model'] = self.base_model_name
        res['base_model_hash'] = self.base_model_hash

        if self.refiner_model_name not in ['', 'None']:
            res['refiner_model'] = self.refiner_model_name
            res['refiner_model_hash'] = self.refiner_model_hash

        res['loras'] = self.loras

        if modules.config.metadata_created_by != '':
            res['created_by'] = modules.config.metadata_created_by

        return json.dumps(dict(sorted(res.items())))

    @staticmethod
    def replace_value_with_filename(key, value, filenames):
        for filename in filenames:
            path = Path(filename)
            if key.startswith('lora_combined_'):
                name, weight = value.split(' : ')
                if name == path.stem:
                    return f'{filename} : {weight}'
            elif value == path.stem:
                return filename


def get_metadata_parser(metadata_scheme: MetadataScheme) -> MetadataParser:
    match metadata_scheme:
        case MetadataScheme.FOOOCUS:
            return FooocusMetadataParser()
        case MetadataScheme.A1111:
            return A1111MetadataParser()
        case _:
            raise NotImplementedError


def read_info_from_image(filepath) -> tuple[str | None, MetadataScheme | None]:
    with Image.open(filepath) as image:
        items = (image.info or {}).copy()

    parameters = items.pop('parameters', None)
    metadata_scheme = items.pop('fooocus_scheme', None)
    exif = items.pop('exif', None)

    if parameters is not None and is_json(parameters):
        parameters = json.loads(parameters)
    elif exif is not None:
        exif = image.getexif()
        # 0x9286 = UserComment
        parameters = exif.get(0x9286, None)
        # 0x927C = MakerNote
        metadata_scheme = exif.get(0x927C, None)

        if is_json(parameters):
            parameters = json.loads(parameters)

    try:
        metadata_scheme = MetadataScheme(metadata_scheme)
    except ValueError:
        metadata_scheme = None

        # broad fallback
        if isinstance(parameters, dict):
            metadata_scheme = MetadataScheme.FOOOCUS

        if isinstance(parameters, str):
            metadata_scheme = MetadataScheme.A1111

    return parameters, metadata_scheme


def get_exif(metadata: str | None, metadata_scheme: str):
    exif = Image.Exif()
    # tags see see https://github.com/python-pillow/Pillow/blob/9.2.x/src/PIL/ExifTags.py
    # 0x9286 = UserComment
    exif[0x9286] = metadata
    # 0x0131 = Software
    exif[0x0131] = 'ReFocus v' + ReFocus_version.version
    # 0x927C = MakerNote
    exif[0x927C] = metadata_scheme
    return exif
