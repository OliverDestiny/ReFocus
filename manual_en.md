# ReFocus User Manual

## Quick Start

1. Enter your prompt in the **Prompt** box on the left.
2. Click **Generate**.
3. Wait for the results to appear in the gallery on the right.

That's it. All advanced features are optional. You only need a prompt to get started.

---

## Tools

Three standalone tools are available as tabs at the top of the main interface.

### Photopea

**Photopea** is an online professional image editor (similar to Photoshop). You can use it without leaving ReFocus for:

- Cropping and resizing
- Layers and masks
- Color correction and retouching

> An internet connection is required for this feature.

### rembg

The **rembg** tab provides one-click background removal:

1. Upload an image
2. Select a removal model from the dropdown
3. Click "Remove Background"

The available models are the same four as in "Automatic Mask Generation":

- **isnet-general-use**: General-purpose model, suitable for most scenes (default)
- **u2net**: Classic salient object detection model, fast
- **u2net_human_seg**: Specialized for human segmentation, ideal for portraits
- **isnet-anime**: Optimized for anime/illustration style images

### Prompt Helper

The **Prompt Helper** tab is based on `sd-webui-prompt-all-in-one-app`, providing a visual tag-based prompt building experience:

- Browse common prompt tags by category, click to add them to the builder
- Search and filter specific prompt styles
- Translation and format optimization between Chinese and English

The tool runs in a separate panel. You can combine prompts in the builder, then use the built-in "Copy to Clipboard" button to paste into the main prompt box.

---

## Controlling Image Generation

For more control, check **Input Image** — four tool panels will expand below.

### Upscale or Variation (UOV)

Upload an image, then you can:

- **Vary**: Generate a similar but slightly different version
  - Subtle → Minor adjustments
  - Strong → More dramatic changes
  - Custom → Manually control the variation strength (0–1, higher = more change)

- **Upscale**: Enlarge the image while adding details
  - Select a scale factor (0.25x – 4x)
  - Enable "Fast Mode" to upscale without diffusion sampling (faster, fewer details)
  - Check "Ignore Prompt" to use an empty prompt, letting the model explore freely — great for creative inspiration

### Image Prompt

Upload reference images to influence the generation. Four control types are available:

- **ImagePrompt**: Extracts style or content from the reference
- **PyraCanny**: Uses edges to lock composition — good for preserving pose or layout
- **CPDS**: Preserves spatial structure and depth
- **FaceSwap**: Maintains facial consistency

Each reference image can be adjusted independently:

- **Stop At**: Controls when the control network stops influencing the sampling process (higher = longer influence)
- **Weight**: Controls the strength of the control network's influence

### Inpaint

Draw white areas (mask) on the image to let the model regenerate those regions.

- **Inpaint**: Erase and redraw a specific area

### Automatic Mask Generation

Check **"Show Mask Generation"** to let the system automatically analyze the image and generate a mask. Four models are available:

- **isnet-general-use**: General-purpose model, suitable for most scenes (default)
- **u2net**: Classic salient object detection model, fast
- **u2net_human_seg**: Specialized for human segmentation, ideal for portraits
- **isnet-anime**: Optimized for anime/illustration style images

Select a model and click **"Generate mask from image"**. The mask will be overlaid on the image. You can still manually adjust the mask (paint or erase) before performing the inpaint operation.

### Describe (Prompt Reverse Engineering)

Upload an image, and the system will analyze it and generate a descriptive prompt automatically. Two modes are available:

- Photograph mode
- Art/Anime mode

### Metadata

If you have a previously generated ReFocus image with embedded parameters, you can:

- Upload it to view the complete parameter record
- Click "Apply Metadata" to load all parameters back into the UI for easy reproduction

---

## Advanced Settings (Right Panel)

Check the **Advanced** checkbox on the main interface to expand the settings panel on the right.

### Basic Parameters

- **Steps**: Higher values generally produce better details. 1–10 steps automatically switch to LCM ultra-fast mode.
- **Aspect Ratio**: A list of preset aspect ratios.
- **Image Number**: Number of images to generate per batch.
- **Negative Prompt**: Describe what you do not want to appear in the image.
- **Seed**: Fixed seeds reproduce the same result; random seeds produce different results each time.

### Models and LoRA

- Select a base model (SDXL only) and a refiner model.
- Refiner Switch: Controls at which step to switch to the refiner.
- Up to 5 LoRAs can be loaded simultaneously, each with individually adjustable weights.

### Advanced Debug Tools

> ⚠️ **Warning**
>
> The following parameters are intended for advanced users and debugging scenarios. **If you do not understand what a parameter does, do not change it.** In most cases, default values produce excellent results. Incorrect adjustments may degrade image quality or cause unexpected outputs.

- **Guidance Scale (CFG Scale)**: Higher values make the result more faithful to the prompt. The default is usually sufficient.
- **Image Sharpness**: Controls image clarity. Excessive adjustment may cause unnatural artifacts.
- **Output Format**: Select the file format for saving images (png, jpg, webp).
- **Sampler / Scheduler**: Manually select the sampling algorithm. Keep the defaults unless you understand the differences.
- **ControlNet Softness**: Adjusts how "soft" the ControlNet's influence is. The default works for most cases.
- **Canny Threshold**: Adjusts edge detection sensitivity. Only needed if edge control is not accurate.
- **FreeU**: Experimental optimization that can improve background details. Effectiveness varies by model.

> If your goal is simply to create a good-looking image, **do not touch the parameters above**. They exist to solve specific technical issues, not for everyday use.

---

## Shortcuts and Tips

- **After clicking Generate**: Stop completely terminates the process; Skip skips the current image and continues to the next.
- **Prompt Box**: You can paste JSON metadata directly into the prompt box. The system will automatically detect it and display the "Load Parameters" button.
- **Presets**: Select a preset in the settings panel to quickly switch entire parameter configurations.
- **History Log**: All generated images are saved in the `outputs/` folder, along with an HTML index page for easy browsing.
