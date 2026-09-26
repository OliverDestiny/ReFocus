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
- **Image control tools**: Upscale / Vary, Image Prompt (IP-Adapter, Canny, CPDS, FaceSwap), Inpaint with built-in mask generation (4 models) and external mask upload, Describe (BLIP / WD14 tagger), Metadata loading.
- **Integrated tools**: Prompt Helper (sd-webui-prompt-all-in-one) and rembg background removal.

---

## Project Structure

```txt

ReFocus/
├── launch.py              # Entry point
│
├── ReFocus_version.py     # Version info
│
├── docs/                  # Developer and user documentation
│
├── modules/               # Core backend logic
│   ├── args_manager.py    # CLI arguments
│   ├── webui.py           # Gradio UI
├── extras/                # Extension modules (IP-Adapter, Describe, etc.)
├── comfy/                 # ComfyUI core, byte-identical to upstream
├── node_helpers.py        # required by comfy/hooks.py
│
├── javascript/            # Custom JavaScript for UI interaction
├── css/                   # Custom CSS styles
├── assets/                # Static assets (the favicon)
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

> **Note**: PyTorch is not included in `requirements.txt`. Please install it first according to your CUDA version:
>
> ```bash
> # (for example)
> # For CUDA 12.x
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> # For CUDA 11.8
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> # For CPU-only
> pip install torch torchvision torchaudio
> ```
>
> See [PyTorch official guide](https://pytorch.org/get-started/locally/) for more options.

```bash
git clone https://github.com/OliverDestiny/ReFocus.git
cd ReFocus
pip install -r requirements.txt
python launch.py
```

The UI is served at **http://127.0.0.1:12345/**. Use `python launch.py --port 8080` to change it.

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

ReFocus is a derivative work: it is rebuilt from [DeFooocus](https://github.com/ehristoforu/DeFooocus),
which in turn builds on [Fooocus](https://github.com/lllyasviel/Fooocus). Both are GPL-3.0, as is this
project. Thanks also to the Stable Diffusion research and open-source ecosystem.

### Third-party components

The following third-party code is redistributed inside this repository. **No model weights are
included** — every model a feature needs is downloaded by the user or by the application itself.

| Component | Where | License |
| :--- | :--- | :--- |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — an unmodified copy (vendored core) | `comfy/` | GPL-3.0 |
| [BLIP](https://github.com/salesforce/BLIP) — image captioning | `extras/BLIP/` | BSD-3-Clause (Salesforce) |
| [facexlib](https://github.com/xinntao/facexlib) — face detection and parsing | `extras/facexlib/` | MIT (Xintao Wang) |
| [ComfyUI-WD14-Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger) — WD14 tagging | `extras/wd14tagger.py` | MIT (pythongosssss) |
| [ESRGAN](https://github.com/xinntao/ESRGAN) — upscaler support | `extras/esrgan/` | BSD-3-Clause |
| [sd-webui-prompt-all-in-one-app](https://github.com/Physton/sd-webui-prompt-all-in-one-app) — Prompt Helper | `prompt_helper/` | MIT (Physton) |
| [Fooocus](https://github.com/lllyasviel/Fooocus) — the codebase this UI was built from | — | GPL-3.0 |
| [DeFooocus](https://github.com/ehristoforu/DeFooocus) — direct parent project | — | GPL-3.0 |
| Stable Diffusion model configuration files | `models/configs/` | from the Stable Diffusion repositories |

Each component keeps its own license and copyright; the notices that ship with them (for example
`prompt_helper/LICENSE` and `extras/esrgan/LICENSE-ESRGAN`) are kept in place.

---
