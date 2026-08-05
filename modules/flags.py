from enum import IntEnum, Enum

disabled = 'Disabled'
enabled = 'Enabled'

UOV_MODE_DISABLED = 'Disabled'
UOV_MODE_VARY = 'Vary'
UOV_MODE_UPSCALE = 'Upscale'

UOV_VARY_SUBTLE = 'Subtle'
UOV_VARY_STRONG = 'Strong'

CIVITAI_NO_KARRAS = ["euler", "euler_ancestral", "heun", "dpm_fast", "dpm_adaptive", "ddim", "uni_pc"]

# fooocus: a1111 (Civitai)
KSAMPLER = {
    "euler": "Euler",
    "euler_ancestral": "Euler a",
    "heun": "Heun",
    "heunpp2": "",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "lms": "LMS",
    "dpm_fast": "DPM fast",
    "dpm_adaptive": "DPM adaptive",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_sde_gpu": "DPM++ SDE",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_gpu": "DPM++ 2M SDE",
    "dpmpp_3m_sde": "",
    "dpmpp_3m_sde_gpu": "",
    "ddpm": "",
    "lcm": "LCM"
}

SAMPLER_EXTRA = {
    "ddim": "DDIM",
    "uni_pc": "UniPC",
    "uni_pc_bh2": ""
}

SAMPLERS = KSAMPLER | SAMPLER_EXTRA

KSAMPLER_NAMES = list(KSAMPLER.keys())

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "lcm", "turbo"]
SAMPLER_NAMES = KSAMPLER_NAMES + list(SAMPLER_EXTRA.keys())

sampler_list = SAMPLER_NAMES
scheduler_list = SCHEDULER_NAMES

refiner_swap_method = 'joint'

cn_ip = "ImagePrompt"
cn_ip_face = "FaceSwap"
cn_canny = "PyraCanny"
cn_cpds = "CPDS"

ip_list = [cn_ip, cn_canny, cn_cpds, cn_ip_face]
default_ip = cn_ip

default_parameters = {
    cn_ip: (0.5, 0.6), cn_ip_face: (0.9, 0.75), cn_canny: (0.5, 1.0), cn_cpds: (0.5, 1.0)
}  # stop, weight

inpaint_engine_versions = ['None', 'v1', 'v2.5', 'v2.6']

output_formats = ['png', 'jpg', 'webp']

MASK_MODEL_CHOICES = [
    'isnet-general-use',
    'u2net',
    'u2net_human_seg',
    'isnet-anime'
]

inpaint_mask_models = MASK_MODEL_CHOICES

inpaint_option_default = 'Inpaint or Outpaint (default)'
inpaint_option_detail = 'Improve Detail (face, hand, eyes, etc.)'
inpaint_option_modify = 'Modify Content (add objects, change background, etc.)'
inpaint_options = [inpaint_option_default, inpaint_option_detail, inpaint_option_modify]

desc_type_photo = 'Photograph'
desc_type_anime = 'Art/Anime'


class MetadataScheme(Enum):
    FOOOCUS = 'fooocus'
    A1111 = 'a1111'


metadata_scheme = [
    (f'{MetadataScheme.FOOOCUS.value} (json)', MetadataScheme.FOOOCUS.value),
    (f'{MetadataScheme.A1111.value} (plain text)', MetadataScheme.A1111.value),
]

lora_count = 5
lora_count_with_lcm = lora_count + 1

controlnet_image_count = 4


class Steps(IntEnum):
    QUALITY = 45
    SPEED = 25
    EXTREME_SPEED = 10
    SPEED_A = 15
    SPEED_B = 20
    NORMAL_A = 30
    NORMAL_B = 35
    QUALITY_A = 40
    QUALITY_B = 50


class StepsUOV(IntEnum):
    QUALITY = 36
    SPEED = 18
    EXTREME_SPEED = 8
    SPEED_A = 12
    SPEED_B = 16
    NORMAL_A = 24
    NORMAL_B = 28
    QUALITY_A = 32
    QUALITY_B = 40


class Performance(Enum):
    QUALITY = 'Quality'
    SPEED = 'Speed'
    EXTREME_SPEED = 'Extreme Speed'
    SPEED_A = 'SPEED_A'
    SPEED_B = 'SPEED_B'
    NORMAL_A = 'NORMAL_A'
    NORMAL_B = 'NORMAL_B'
    QUALITY_A = 'QUALITY_A'
    QUALITY_B = 'QUALITY_B'

    @classmethod
    def list(cls) -> list:
        return list(map(lambda c: c.value, cls))

    def steps(self) -> int | None:
        return Steps[self.name].value if Steps[self.name] else None

    def steps_uov(self) -> int | None:
        return StepsUOV[self.name].value if Steps[self.name] else None


performance_selections = [
    ('Quality (45 steps)', Performance.QUALITY.value),
    ('Speed (25 steps)', Performance.SPEED.value),
    ('Extreme Speed (10 steps)', Performance.EXTREME_SPEED.value),
    ('SPEED_A (15 steps)', Performance.SPEED_A.value),
    ('SPEED_B (20 steps)', Performance.SPEED_B.value),
    ('NORMAL_A (30 steps)', Performance.NORMAL_A.value),
    ('NORMAL_B (35 steps)', Performance.NORMAL_B.value),
    ('QUALITY_A (40 steps)', Performance.QUALITY_A.value),
    ('QUALITY_B (50 steps)', Performance.QUALITY_B.value)
]