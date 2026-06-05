# My Creative Diffusion UI (Based on DeFooocus)

A modern, simplified, and fully local image‑generation interface designed for **Stable Diffusion (SD)** and **Stable Diffusion XL (SDXL)** models.  
This project began as a personal rebuild of DeFooocus, but it is now evolving into a clean, maintainable, and extensible creative engine.

The goal is simple:

> **Power without complexity.**  
> A beautiful, minimal UI that produces high‑quality images without node graphs, clutter, or unnecessary parameters.

---

## ✨ Key Features

### 🎨 Simple, Creative UI

- Minimal interface inspired by Fooocus  
- Focus on prompts and creativity, not technical knobs  
- Clean layout for SD and SDXL workflows  

### ⚙️ Modernized Backend

- Refactored codebase for clarity and maintainability  
- Dead APIs removed  
- Simplified model loading  
- Cleaner logging and metadata handling  

### 🚀 SD & SDXL Support

- Works with standard SD 1.5 models  
- Works with SDXL base + refiner  
- Supports LoRAs, embeddings, and VAE overrides  

### 🧩 Extensible Architecture

- Modular structure for future model adapters  
- Designed to support new model families later (Qwen‑Image, Z‑Image, Anima, etc.)  
- Clear separation between UI, pipeline, and model logic  

### 🖼️ Local‑Only Operation

- No cloud APIs  
- No external dependencies  
- Full privacy and offline capability  

---

## 📁 Project Structure (Simplified)

```
core/
    loader/        # Model loading and configuration
    sampler/       # Sampling algorithms
    pipeline/      # SD/SDXL inference pipeline
ui/
    components/    # UI elements
    server/        # Gradio or custom backend
models/
    checkpoints/   # SD/SDXL models (ignored by Git)
    loras/
    vae/
utils/
    logging/
    config/

```

This structure is intentionally clean and future‑proof.

---

## 🛠️ Installation

> **Note:** This project is under active reconstruction.  
> Instructions will evolve as the codebase stabilizes.

### Requirements

- Python 3.10  
- NVIDIA GPU recommended (6GB+ VRAM)  
- Windows or Linux  

### Basic Setup

```bash
git clone https://github.com/YOUR_USERNAME/defooocus-rebuild.git
cd defooocus-rebuild
pip install -r requirements.txt
python launch.py
```

Place your SD/SDXL models into:

models/checkpoints/

---

## 🧭 Roadmap

### Phase 1 — SD/SDXL Foundation

- Clean and reorganize code  
- Modernize UI  
- Stabilize inference pipeline  

### Phase 2 — Quality of Life

- Better presets  
- Improved metadata  
- Faster upscaling options  

### Phase 3 — Model Adapter Layer

- Unified interface for future models  
- Support for Qwen‑Image, Z‑Image, Anima, etc.  

### Phase 4 — Personal Features

- Style library  
- Prompt presets  
- Batch generation manager  

---

## 📜 License

This project is a personal rebuild and does not include model files.  
Users must provide their own SD/SDXL models.

---

## 🙏 Acknowledgements

This project draws inspiration from:

- Fooocus by llyasviel  
- DeFooocus community forks  
- Stable Diffusion research and open‑source ecosystem  
