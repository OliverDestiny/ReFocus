# README_DEV.md

Developer documentation for ReFocus. For user documentation, see `manual_en.md` and `manual_cn.md` in this folder.

---

## Overview

ReFocus is a Gradio-based UI for Stable Diffusion XL image generation, rebuilt from DeFooocus with significant modifications. This document covers the architecture, key modules, and integration details relevant to developers.

---

## Two product lines and the version policy

The repository carries two independent lines, and knowing which one a change belongs to is the first
question to ask about any work:

- **`main` — the SDXL toolbox, no Anima.** Its philosophy is simple plus powerful: the defaults should
  produce good images without opening Advanced, Advanced should give an advanced user full control, and
  there is **no commitment to tracking new models**. Anyone who needs the newest architecture should
  use ComfyUI; even much larger projects cannot stay current, so this one does not try.
- **`feature/d5-anima` — the base plus Anima.** Two purposes: probe how a new model family fits the
  base, and freeze a personal ComfyUI workflow behind a cleaner, cross-device-usable UI. The practical
  argument is that node graphs are unusable on mobile, SDXL's useful life is long, and UNet models are
  still state of the art for specific content such as art style.

Consequences:

- Anything unrelated to Anima (FreeU, the resolution control, ControlNet, Debug Tools) is written on
  **main** and tagged there. The Anima branch takes it in by merging main, so the two trees stay close.
- `v x.y.z` on main: **x** for a core-level change (the DeFooocus-to-ReFocus rewrite, the move to the
  vendored ComfyUI core), **y** for a feature (FreeU promoted, ControlNet foundation), **z** for a
  bug-fix-only release. `v2.0.0` is the core migration, `v2.0.1` a fix, `v2.1.0` the first feature batch.
- The Anima branch has its own axis, prefixed so `git tag` output is unambiguous: `anima-v1.0.0` is the
  milestone where Anima support lands on the base, then `anima-v1.x` and `anima-v1.x.y` follow the same
  feature/fix meaning.
- Only milestones are tagged. Intermediate commits — including feature branches that are not yet
  verified — carry no tag.
- Acceptance work stays out of the repository: the control experiments live in `.zcode/probes/` and are
  local only. A milestone is tagged after they pass, not before.

---

---

## Architecture

### ControlNet types declare their own parameters

`modules/controlnet_registry.py` is the single place a ControlNet type is declared: its model loader,
its preprocessor, how it is applied (`'conditioning'` through `core.apply_controlnet`, or
`'unet_patch'` through `ip_adapter.patch_model`), and now also two things that used to be hardcoded:

- `conditioning_source` is `'image'` when the slot supplies a reference image, or `'params'` when the
  type describes its condition with numbers instead of an image (a camera-angle ControlNet would be
  the first of those).
- `param_spec` lists the type's own parameters as `ParamSpec` declarations. The UI builds widgets from
  `param_union()` and shows only the ones the selected type declares; the worker fills missing keys
  from the spec defaults.

Each slot's values travel to the worker as one JSON argument, `cn_params_{i}`, so **adding a type never
changes the parameter count**: declare it in the registry and it is done. A collect step in the
Generate (and Polish) chain fills the per-slot states before `get_task` runs. Unreadable JSON is a hard
error rather than a silent fallback to defaults, because a fallback would make a threshold quietly do
nothing.

Both preprocessors and their parameters moved here from the old global Debug Tools sliders: the Canny
thresholds belong to PyraCanny and the face detection confidence to FaceSwap, and they are now per slot
rather than per run.

The five types are ImagePrompt, PyraCanny, CPDS, **Depth** and FaceSwap. Depth needs no parameters of
its own: it pairs Stability's `control-lora-depth-rank128` with MiDaS DPT-Hybrid, the estimator that
Control-LoRA was trained against.

- `extras/preprocessors.MiDaSDepth` holds the model: the architecture comes from `intel-isl/MiDaS`
  through `torch.hub` (fetched once, cached under `~/.cache/torch/hub`, so only the first call in a
  fresh environment needs network), the weights are the local `dpt_hybrid-midas-501f0c75.pt` (about
  500 MB). The checkpoint matches the hub architecture with zero missing and zero unexpected keys.
  - Preprocessing follows MiDaS's own recipe: short side to 384 with both sides snapped to 32, ImageNet
    normalisation, then the prediction is resized back and min-max normalised to a 0..255 grey image.
  - Measured: 13 s for the first call (hub fetch plus CUDA warm-up), 0.05 s per image afterwards, and
    the depth map of a real 512x512 generation came out with std 92.9 (a real gradient, not a constant).
- Verified with the same control experiment the other types use: same reference image and seed, Depth
  weight 0.2 versus 1.0 differs by mean|diff| 31.2/255 over 85 % of pixels, so the depth condition
  really reaches the pipeline (`.zcode/probes/main_batch_probe.py`).

### Widget values must be members of their choices

Gradio accepts an initial `value` that is not in the widget's `choices`, but it rejects it **at request
time** with `Value: X is not in the list of choices` — which means the app starts fine and then every
Generate click fails. Three things feed such values in: a `config.txt` key whose validator is weaker
than the widget's choice list (a renamed or deleted model file, a stale preset name from `--preset`),
and metadata imported from an older image naming a sampler or model that no longer exists.

Therefore: `webui.first_valid(value, choices, default)` coerces initial values into the choice list and
prints what it substituted, and `meta_parser.load_parameter_button_click` checks membership for
`sampler`, `scheduler`, `refiner_swap_method`, `base_model`, `refiner_model` and the LoRA names (keeping
the widget's current value when a name is gone instead of passing it on), and clamps the resolution
into the sliders' range. When a config key feeds a widget, its validator must be at least as strict as
that widget's `choices`, or the value has to be normalized before use — the metadata scheme does the
latter, because a renamed key has to keep accepting the old spelling.

### Where the shared UI parameters live

The right column has three tabs and the split is deliberate: **Settings** for what a normal session
touches, **Models** for the model pickers, and **Advanced** for the parameters that have no normal
counterpart. There is no developer-mode gate on any of it — the old "Advanced mode" checkbox only
hid controls that are now simply in the Advanced tab.

- **Settings is ordered by how often a normal session touches a control**, top to bottom: Steps (with
  its three preset buttons in the same row, `equal_height` so they stand at the slider's height), the
  output size (two sliders with number boxes, a swap button, then the presets as buttons), the
  Negative Prompt, the Preset dropdown, Sampling, Seed, Image Number, FreeU, Guidance & Sharpness,
  Output Format, and the output/metadata switches last.
  - The Negative Prompt is a `gr.Accordion` that starts closed — a heading with no checkbox, so the
    textbox inside keeps its own state and the tab stays short.
  - Sampling and Guidance/Sharpness each sit behind a checkbox that reveals them.
  - FreeU's checkbox *is* `freeu_enabled`, so ticking it enables FreeU and reveals its four
    coefficients at once. Its defaults are 1.3 / 1.4 / 0.9 / 0.2, the values upstream ComfyUI's
    `FreeU_V2` node ships (the FreeU v2 recommendation); FreeU itself stays off by default.
  - The output size presets come from `config.available_aspect_ratios` and render four to a row with
    the size only (`1024×1536`); the ratio suffix does not fit. The `.preset_row` CSS drops the
    theme's per-button min-width, which is what otherwise wraps them into two fat columns. Step is 8.
    `aspect_ratios_selection` used to be a single radio list and the two `overwrite_width` /
    `overwrite_height` sliders existed only because there was no way to type a size.
  - The shipped presets keep a short side of at least 1024 (SDXL's native scale) plus the 1024×1024
    square: 1152×2048, 1080×1920, 1024×1536, 1536×2048, 1024×1024, 2048×1536, 1536×1024, 1920×1080.
- **Advanced** is the Debug Tools, Control and Inpaint groups: the ADM scalers, adaptive CFG, the
  refiner swap method, the ControlNet debug switches, the mixing flags, ControlNet softness and the
  inpaint parameters. Six `overwrite_*` controls were deleted from here, because Steps is already an
  exact integer slider, Refiner Switch At expresses the same switch as a fraction, the Vary and
  Upscale tabs have their own Denoise Strength, and the width/height overwrites were replaced by the
  resolution control.
- **Every control is defined exactly once.** The output flags and metadata options used to be defined
  a second time in Debug Tools after they moved to Settings; the later definition won silently,
  because the contract (`ctrls`) is assembled at the end of the module and reads the last binding —
  the copies in Settings did nothing. `.zcode/probes/ui_probe.py` checks that no contract control
  shares a label with a component outside the contract.

### CLI arguments

`modules/args_manager.py` owns ReFocus's own flags and **wins over the core for any name both
define**: it strips its arguments from `sys.argv` before `comfy.cli_args` parses, then writes its
values (including defaults) back into the shared namespace. `--port` is the visible case — ReFocus
defaults to **12345**, the core's own default is 8188, and `GRADIO_SERVER_PORT` is not consulted.
`--disable-metadata` is the other shared name. Everything else (`--listen`, `--temp-path`,
`--disable-smart-memory`, `--fast`, …) still comes from the core.

### Custom CSS/JS and `js=` callbacks

`modules/webui.get_custom_head()` reads `css/style.css` and the five scripts under `javascript/`,
**relative to the repository root** (the parent of `modules/`), and appends them to `<head>`. A wrong
path here fails silently — each file is skipped and the page simply loses that script — so every miss
is printed at startup and `.zcode/probes/ui_probe.py` asserts the whole set is present. When these
paths pointed at `modules/` the entire bundle was dropped: the page still rendered (Gradio's own JS),
but every custom feature was dead, including the Input Image checkbox, whose event carried
`js=switch_js` and aborted in the browser because `viewer_to_bottom()` did not exist.

For that reason the state-changing part of an event never carries `js`: a throwing js callback aborts
the whole event (no server round trip at all), so `input_image_checkbox.change(...)` updates the panel
visibility on its own and the scrolling runs in a following `.then(fn=lambda: None, js=...)`.

### Stack

| Component | Technology |
| :--- | :--- |
| UI Framework | Gradio 6.20.0 |
| Web Server | FastAPI + Uvicorn |
| Diffusion Core | `comfy/` — ComfyUI v0.33.4, vendored unmodified |
| Model Loading | `comfy.sd` |
| Sampling | `comfy.samplers` |
| Image Processing | OpenCV, PIL, NumPy |

### Directory Layout

Two rules decide where a file lives: the vendored core stays at the root so replacing it is a wholesale
copy (`comfy/` plus the `node_helpers.py` it imports by bare name), and everything of ReFocus's own
that is not an entry point lives under `modules/`. `launch.py` and `ReFocus_version.py` are the two
remaining top-level Python files; `args_manager.py` and `webui.py` moved into `modules/` with the rest,
and the documents moved into `docs/`.


```txt
ReFocus/
├── launch.py                    # Entry point: FastAPI app + uvicorn
│
├── ReFocus_version.py           # Version File
│
├── docs/                        # Documentation (this file, the manuals, NOTICE)
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
├── comfy/                       # ComfyUI core, byte-identical to upstream v0.33.4 (GPL v3)
├── node_helpers.py              # required by comfy/hooks.py
│
├── prompt_helper/               # Prompt Helper sub-application
│   ├── app.py                   # FastAPI sub-app
│   └── static/                  # Frontend assets (Vue app)
└── ...
```

### Vendored core (`comfy/`)

`comfy/` is a **byte-identical, unmodified copy of ComfyUI v0.33.4**. It replaces the older
`ldm_patched/` fork, which had been renamed and patched and could not be updated.

Two rules follow from that, and both matter:

1. **Do not edit anything under `comfy/`.** ReFocus' own behaviour lives in `modules/`, and the
   injection layer is confined to `modules/patch.py` — currently three patches:
   `SDXL.encode_adm`, a thin wrapper around `ControlNet.forward`, and a load-time logger. Every
   replacement is asserted to have taken effect in `patch_all()`, so an upstream change makes it
   fail loudly at startup rather than silently stop working.
2. **There is no commitment to track upstream.** The point of keeping the copy unmodified is that
   adopting a newer core is a wholesale jump: copy that tag's `comfy/` over this one, re-check
   `modules/comfy_ops.py` (the functions ported from `comfy_extras/` and `nodes.py`, which are
   not vendored because they drag in `comfy_api`, `folder_paths` and a large number of unrelated
   architectures), and re-run the acceptance checks. Keeping a diffable copy is what makes that
   mechanical instead of a re-derivation.

`node_helpers.py` sits next to `comfy/` at the repository root and is **not** part of ReFocus: it is a
top-level module that `comfy/hooks.py` imports by bare name (`from node_helpers import
conditioning_set_values`), and `comfy.hooks` is imported at module level by `model_patcher`, `samplers`
and `sd`, so it loads on every run. It cannot move into `modules/` without editing `comfy/`, and moving
`comfy/` itself would require putting `modules/` on `sys.path` — which would shadow the stdlib `html`
(with `modules/html.py`) and the pinned `rembg` package (with `modules/rembg.py`). Both stay at the root.

Two pip packages are required by the core and are pinned exactly in `requirements.txt`:
`comfy-kitchen` (imported unconditionally by `comfy/ldm/modules/attention.py` and
`comfy/text_encoders/llama.py`, and used for Anima's RoPE) and `comfy-aimdo` (the dynamic VRAM
offloader; imported unconditionally by six files, and dormant because `aimdo_enabled` defaults
to False — it is kept because it is the mechanism that would let larger models fit in limited
VRAM). Both are versioned independently of ComfyUI and have had breaking changes within the same
minor line, so do not float them.

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

Key arguments defined in `modules/args_manager.py`:

| Argument | Description |
| :--- | :--- |
| `--preset` | Load a specific UI preset |
| `--disable-preset-selection` | Hide preset dropdown in UI |
| `--language` | Load translation from `language/*.json` |
| `--theme` | Set Gradio theme (light/dark) |
| `--disable-image-log` | Disable writing images to disk |
| `--disable-metadata` | Disable metadata embedding |

See `modules/args_manager.py` for the full list.

---

## Key Modules

### `async_worker.py`

Runs in a separate thread (`threading.Thread`). Handles:

- Parsing UI inputs (via `ctrls` list from `modules/webui.py`)
- Model loading and caching
- Diffusion sampling (with progress callbacks)
- Inpaint processing
- ControlNet / IP-Adapter application

The worker communicates with the UI via `AsyncTask.yields`:

| Yield Flag | Purpose |
| :--- | :--- |
| `preview` | Update progress bar and preview image |
| `results` | Show intermediate results |
| `finish` | Final results and UI reset |

### `modules/webui.py`

Defines the entire Gradio UI. Key sections:

- **Main UI**: `gr.Blocks` with tabs (Generation, Photopea, rembg, Prompt Helper)
- **Input Image Panel**: UOV (Upscale/Vary), Image Prompt, Inpaint, Describe, Metadata
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

Gradio components are defined in `modules/webui.py`. For UI changes, no frontend build step is required—just reload the page.

### Worker Logging

`async_worker.py` prints progress and debugging info. Look for `[ReFocus]` and `[Parameters]` prefixes in console output.

### Model Download Issues

Model download failures are logged in `deps_models_download.py`. The module attempts mirror download first, then falls back to official sources.

### Common Pitfalls

| Issue | Likely Cause |
| :--- | :--- |
| CSS/JS missing | Relative paths in `index.html` or incorrect mount order |
| Gradio UI not rendering | `gradio_root` not set, or `mount_gradio_app` called before UI definition |
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
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — the `comfy/` core

---
