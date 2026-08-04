# ReFocus

A clean UI for Stable Diffusion XL image generation.

Rebuilt from [DeFooocus](https://github.com/ehristoforu/DeFooocus) with significant modifications: upgraded from Gradio 3.41 to 6.20, fixed numerous backend issues, completed unfinished features, and streamlined the codebase.

> **Power without complexity.**
> A clean interface for high-quality image generation without node graphs or unnecessary controls.

---

## Features

- **Clean, minimal UI**: Prompt-focused workflow, no technical clutter.
- **Modernized backend**: Gradio 6.20.0, refactored codebase, simplified model loading.
- **SDXL support**: Base model supports SDXL only. Refiner supports SDXL or SD 1.5.
- **Image control tools**: Upscale / Vary, Image Prompt (IP-Adapter, Canny, CPDS, FaceSwap), Inpaint / Outpaint with external mask upload, Describe (BLIP / WD14 tagger), Metadata loading.
- **Integrated tools**: Prompt Helper (sd-webui-prompt-all-in-one), Photopea (online editor, requires internet), rembg background removal.

---

## Project Structure

```txt

ReFocus/
├── launch.py              # Entry point
│
├── args_manager.py        # CLI arguments
├── webui.py               # Gradio UI
├── shared.py              # Shared state
├── ReFocus_version.py     # Version info
│
├── modules/               # Core backend logic
├── extras/                # Extension modules (IP-Adapter, Describe, etc.)
├── ldm_patched/           # ComfyUI-based diffusion core
│
├── javascript/            # Custom JavaScript for UI interaction
├── css/                   # Custom CSS styles
├── assets/                # Static assets (favicon, etc.)
│
├── prompt_helper/         # Prompt Helper integration
├── presets/               # UI presets (JSON)
├── language/              # Localization files (JSON)
├── models/                # Model files (not included)
│   ├── checkpoints/
│   ├── loras/
│   └── ...
└── outputs/               # Generated images

```

---

## Installation

### Requirements

- Python 3.10
- NVIDIA GPU recommended (6GB+ VRAM)
- Windows or Linux

### Setup

```bash
git clone https://github.com/OliverDestiny/ReFocus.git
cd ReFocus
pip install -r requirements.txt
python launch.py
```

Place your SDXL models in `models/checkpoints/`.

---

## Roadmap

- **Phase 1 — Foundation (Completed)**: Code cleanup, UI modernization, pipeline stabilization.
- **Phase 2 — Quality of Life (In Progress)**: Better presets (user-controlled), improved metadata handling, different upscale options (ESRGAN module planned).
- **Phase 3 — Model Adapter Layer (Planned)**: Unified interface for future models.
- **Phase 4 — Personal Features (Planned)**: To be defined.

---

## License

GNU General Public License v3.0

This project does not include model files. Users must provide their own SDXL models.

---

## Acknowledgements

- [Fooocus](https://github.com/lllyasviel/Fooocus) by lllyasviel
- [DeFooocus](https://github.com/ehristoforu/DeFooocus) by ehristoforu
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (ldm_patched)
- [Prompt Helper](https://github.com/Physton/sd-webui-prompt-all-in-one-app) by Physton
- Stable Diffusion research and open-source ecosystem

---
