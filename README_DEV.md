# README_DEV.md

Developer documentation for ReFocus. For user documentation, see `manual_en.md` and `manual_cn.md`.

---

## Overview

ReFocus is a Gradio-based UI for Stable Diffusion XL image generation, rebuilt from DeFooocus with significant modifications. This document covers the architecture, key modules, and integration details relevant to developers.

---

## Architecture

### Stack

| Component | Technology |
| :--- | :--- |
| UI Framework | Gradio 6.20.0 |
| Web Server | FastAPI + Uvicorn |
| Diffusion Core | `ldm_patched` (ComfyUI fork) |
| Model Loading | `ldm_patched.modules.sd` |
| Sampling | `ldm_patched.modules.samplers` |
| Image Processing | OpenCV, PIL, NumPy |

### Directory Layout

```txt
ReFocus/
├── launch.py                    # Entry point: FastAPI app + uvicorn
│
├── args_manager.py              # CLI argument definitions
├── webui.py                     # Gradio UI definition (~1000+ lines)
├── shared.py                    # Shared state (gradio_root)
├── ReFocus_version.py           # Version File
│
├── modules/                     # Core backend
│   ├── config.py                # Config loading, model paths, presets
│   ├── core.py                  # Model loading, VAE encode/decode, ksampler
│   ├── default_pipeline.py      # Diffusion pipeline (base + refiner)
│   ├── async_worker.py          # Async generation worker (threaded)
│   ├── inpaint_worker.py        # Inpaint mask processing
│   ├── deps_models_download.py  # Model download utilities
│   └── patch*.py                # Runtime patches (precision, CLIP, attention)
│
├── extras/                      # Optional extension modules
│   ├── ip_adapter.py            # IP-Adapter / Image Prompt
│   ├── interrogate.py           # BLIP captioning (Describe)
│   ├── wd14tagger.py            # WD14 tagger (Describe/Anime)
│   ├── inpaint_mask.py          # Mask generation using rembg (4 models: isnet-general-use, u2net, u2net_human_seg, isnet-anime)
│   └── ...
│
├── ldm_patched/                 # ComfyUI-based diffusion core (GPL v3)
│
├── prompt_helper/               # Prompt Helper sub-application
│   ├── app.py                   # FastAPI sub-app
│   └── static/                  # Frontend assets (Vue app)
└── ...
```

---

## Development Setup

### Environment

```bash
conda create -n refocus python=3.10
conda activate refocus
pip install -r requirements.txt
```

### Running

```bash
python launch.py
```

### CLI Arguments

Key arguments defined in `args_manager.py`:

| Argument | Description |
| :--- | :--- |
| `--preset` | Load a specific UI preset |
| `--disable-preset-selection` | Hide preset dropdown in UI |
| `--language` | Load translation from `language/*.json` |
| `--theme` | Set Gradio theme (light/dark) |
| `--disable-image-log` | Disable writing images to disk |
| `--disable-metadata` | Disable metadata embedding |

See `args_manager.py` for the full list.

---

## Key Modules

### `async_worker.py`

Runs in a separate thread (`threading.Thread`). Handles:

- Parsing UI inputs (via `ctrls` list from `webui.py`)
- Model loading and caching
- Diffusion sampling (with progress callbacks)
- Inpaint / Outpaint processing
- ControlNet / IP-Adapter application

The worker communicates with the UI via `AsyncTask.yields`:

| Yield Flag | Purpose |
| :--- | :--- |
| `preview` | Update progress bar and preview image |
| `results` | Show intermediate results |
| `finish` | Final results and UI reset |

### `webui.py`

Defines the entire Gradio UI. Key sections:

- **Main UI**: `gr.Blocks` with tabs (Generation, Photopea, rembg, Prompt Helper)
- **Input Image Panel**: UOV (Upscale/Vary), Image Prompt, Inpaint/Outpaint, Describe, Metadata
- **Settings Panel**: Steps, Aspect Ratios, Models, LoRAs, Advanced debug tools
- **Parameter Assembly**: `ctrls` list defines the order of parameters passed to `async_worker`

### `config.py`

Manages:

- Model paths (`path_checkpoints`, `path_loras`, etc.)
- Default values (steps, CFG, sampler, etc.)
- Preset loading (`presets/*.json`)
- Model scanning (`model_filenames`, `lora_filenames`)

### `deps_models_download.py`

Centralized model download utility. All external model downloads route through this module.  
New functions:

- `ensure_rembg_models()`: Downloads the 4 core rembg models (`isnet-general-use`, `u2net`, `u2net_human_seg`, `isnet-anime`) to `~/.u2net/`.
- `LCM_LORA_FILENAME` constant defined in `flags.py` for consistent referencing.

### `flags.py`

Defines global constants and enumerations. Recent additions:

- `MASK_MODEL_CHOICES`: List of 4 rembg model names (single source of truth).
- `LCM_LORA_FILENAME`: Filename constant for LCM LoRA (used across download and metadata parsing).
- Removed obsolete `Performance`, `Steps`, `StepsUOV` enums and related selections.

---

## Prompt Helper Integration

The Prompt Helper (`sd-webui-prompt-all-in-one-app`) is mounted as a sub-application inside the main FastAPI server.

### URL Structure

| Path | Purpose |
| :--- | :--- |
| `/prompt-helper/` | Vue frontend entry |
| `/prompt-helper/static/css/*` | CSS assets |
| `/prompt-helper/static/js/*` | JS assets |
| `/prompt-helper/physton_prompt/*` | Backend API endpoints |
| `/sd-webui-prompt-all-in-one-js` | Extension JS |

### Mount Order (Critical)

The order in `launch.py` matters:

1. Static files (`/prompt-helper/static`)
2. `index.html` endpoint (`/prompt-helper`)
3. Extension JS (`/sd-webui-prompt-all-in-one-js`)
4. Prompt Helper backend (`/prompt-helper`)
5. ReFocus Gradio UI (`/`)

> **Note**: Mounting static files before the backend is essential. If the backend mounts first, it will shadow the static route and return 404 for CSS/JS assets.

### Path Fix: Absolute vs Relative

The frontend (`static/index.html`) must use **absolute paths**:

```html
<!-- Correct -->
<link rel="stylesheet" href="/prompt-helper/static/css/main.min.css">

<!-- Incorrect - breaks in remote deployment -->
<link rel="stylesheet" href="./css/main.min.css">
```

Relative paths resolve differently in local vs remote environments, causing CSS/JS loading failures.

### `prompt_helper/app.py` Structure

The sub-application exports `create_prompt_helper_app()`, which returns a FastAPI instance. It handles:

- Static file mounting (for frontend assets)
- API routes (under `/physton_prompt`)
- Optional HTTP basic auth (via environment variables)

---

## Testing & Debugging

### UI Development

Gradio components are defined in `webui.py`. For UI changes, no frontend build step is required—just reload the page.

### Worker Logging

`async_worker.py` prints progress and debugging info. Look for `[ReFocus]` and `[Parameters]` prefixes in console output.

### Model Download Issues

Model download failures are logged in `deps_models_download.py`. The module attempts mirror download first, then falls back to official sources.

### Common Pitfalls

| Issue | Likely Cause |
| :--- | :--- |
| CSS/JS missing | Relative paths in `index.html` or incorrect mount order |
| Gradio UI not rendering | `shared.gradio_root` not set, or `mount_gradio_app` called before UI definition |
| Worker not starting | Patch issues in `modules/patch.py` or missing imports |
| Inpaint mask not working | Mask upload checkbox not enabled, or uploaded mask not properly merged |

---

## Contributing Guidelines

1. **Code style**: Follow existing patterns; no style linters enforced.
2. **UI changes**: Keep it minimal. Avoid adding controls unless necessary.
3. **Backend changes**: Test with both normal and Input Image workflows.
4. **Documentation**: Update `manual_en.md` / `manual_cn.md` for user-facing changes; update this file for developer-facing changes.

---

## License

GNU General Public License v3.0. See `LICENSE` and `NOTICE.md` for details.

---

## References

- [Gradio Documentation](https://www.gradio.app/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Fooocus (upstream)](https://github.com/lllyasviel/Fooocus)
- [DeFooocus (upstream)](https://github.com/ehristoforu/DeFooocus)
- [ComfyUI (ldm_patched)](https://github.com/comfyanonymous/ComfyUI)

---
