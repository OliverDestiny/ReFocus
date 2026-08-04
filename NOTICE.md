# Acknowledgments & Licenses

## ReFocus

- **License**: GNU General Public License v3.0
- **Note**: This project is a derivative work of [Fooocus](https://github.com/lllyasviel/Fooocus) and [DeFooocus](https://github.com/ehristoforu/DeFooocus), both licensed under GPL v3.0. As a derivative work, ReFocus is also distributed under the GPL v3.0 license.

---

## Upstream Projects

### Fooocus / DeFooocus

- **Original Project**: [Fooocus](https://github.com/lllyasviel/Fooocus) by lllyasviel
- **Derived Project**: [DeFooocus](https://github.com/ehristoforu/DeFooocus) by ehristoforu
- **License**: GNU General Public License v3.0
- **Modifications**: This project modifies and extends the original codebase, including but not limited to: Gradio 6.20 migration, UI overhaul, UOV (Upscale/Vary) refactor, Inpaint mask upload integration, ControlNet fixes, and LCM auto-switching logic.
- **Copyright Notice**: All original copyright notices of the upstream projects are retained and respected.

### ldm_patched (ComfyUI fork)

- **Source**: Derived from [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- **License**: GNU General Public License v3.0
- **Usage**: Provides the core diffusion model loading, sampling, and VAE infrastructure used by ReFocus.

### Prompt Helper (sd-webui-prompt-all-in-one-app)

- **Original Project**: [sd-webui-prompt-all-in-one-app](https://github.com/Physton/sd-webui-prompt-all-in-one-app) by Physton
- **License**: MIT License
- **Copyright**: Copyright (c) 2023 Physton
- **Integration**: Mounted as a sub-application via FastAPI at `/prompt-helper/`.

---

## Additional Dependencies

ReFocus uses many open-source libraries that are not directly modified. Key ones include:

- [Gradio](https://github.com/gradio-app/gradio) — Apache 2.0
- [PyTorch](https://github.com/pytorch/pytorch) — BSD-style
- [Transformers](https://github.com/huggingface/transformers) — Apache 2.0
- [OpenCV](https://github.com/opencv/opencv) — Apache 2.0
- [rembg](https://github.com/danielgatis/rembg) — MIT
- [BLIP](https://github.com/salesforce/BLIP) — BSD-3-Clause

Full license details for all dependencies are available in the `requirements.txt` file and their respective repositories.
