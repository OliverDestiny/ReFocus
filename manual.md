# DeFooocus UI 使用手册

> 本文档适用于 DeFooocus-rebuild 版本，基于 Gradio 6.20 构建。

## 一、界面布局概览

DeFooocus 的 UI 采用**左右两栏布局**：

| 区域 | 比例 | 内容 |
|------|------|------|
| **左栏（主区域）** | 2/3 | 生成主界面、图像输入面板 |
| **右栏（设置面板）** | 1/3 | 所有参数设置（默认隐藏，需勾选“Advanced”展开） |

## 二、左栏 —— 主区域

### 2.1 顶部 Tabs

| Tab 名称 | 功能说明 |
| ---------- | ---------- |
| **Generation** | 主生成界面，包含预览、进度条和结果画廊 |
| **Photopea** | 嵌入 Photopea 在线图像编辑器，可进行高级图片处理 |
| **rembg** | 移除图片背景（基于 rembg 库） |
| **Prompt Helper** | 嵌入提示词辅助工具（本地服务，端口 17860） |

---

### 2.2 Generation Tab

#### 预览与画廊

| 控件 | 类型 | 说明 |
| ------ | ------ | ------ |
| **Preview** | Image | 生成过程中的实时预览（生成时自动显示） |
| **Finished Images** | Gallery | 单次生成完成后的结果展示（生成时自动显示） |
| **Progress Bar** | HTML | 显示生成进度百分比和当前步骤文字 |
| **Gallery** | Gallery | 主画廊，显示所有已生成的历史图片 |

#### 提示词输入区

| 控件 | 类型 | 说明 |
| ------ | ------ | ------ |
| **Prompt** | Textbox | 正向提示词输入框，支持多行。可在此粘贴包含参数的 JSON 元数据 |
| **Generate** | Button | 点击开始生成图像 |
| **Load Parameters** | Button | 从提示词框中解析 JSON 参数并加载到 UI（当检测到 JSON 时自动出现） |
| **Skip** | Button | 跳过当前正在生成的图像（继续生成下一张） |
| **Stop** | Button | 完全停止生成过程 |

#### 控制开关

| 控件 | 类型 | 说明 |
|------|------|------|
| **Input Image** | Checkbox | 启用图像输入面板（勾选后下方面板展开） |
| **Advanced** | Checkbox | 显示/隐藏右侧高级设置面板 |

---

### 2.3 Input Image Panel（勾选“Input Image”后显示）

此面板包含 4 个子选项卡：

#### 2.3.1 Upscale or Variation（UOV）

| 控件 | 类型 | 说明 |
|------|------|------|
| **Upscale or Variation** | Image | 上传待处理图片 |
| **Method** | Radio | 选择处理方式：Disabled（禁用）、Vary (Subtle)（轻微变化）、Vary (Strong)（强烈变化）、Upscale (1.5x)、Upscale (2x)、Upscale (Fast 2x) |

> **UOV 说明**：`Vary` 基于原图进行重绘；`Upscale` 使用 AI 模型放大图片；`Fast 2x` 直接放大不经过扩散采样。

#### 2.3.2 Image Prompt（图像提示）

| 控件 | 类型 | 说明 |
| ------ | ------ | ------ |
| **Image** (1-4) | Image | 上传参考图像（最多 4 张） |
| **Type** | Radio | 控制类型：ImagePrompt（风格迁移）、PyraCanny（边缘控制）、CPDS（深度/结构控制）、FaceSwap（人脸替换） |
| **Advanced** | Checkbox | 展开高级参数：Stop At（停止步数）、Weight（权重） |

> **各类型说明**：
>
> - **ImagePrompt**：IP-Adapter 风格迁移，提取参考图的风格/内容
> - **PyraCanny**：Canny 边缘检测控制，严格锁定构图
> - **CPDS**：深度/结构控制，保留空间布局
> - **FaceSwap**：人脸识别 + IP-Adapter，保持面部一致性

#### 2.3.3 Inpaint or Outpaint（修复/扩图）

| 控件 | 类型 | 说明 |
| ------ | ------ | ------ |
| **Inpaint Canvas** | ImageEditor | 上传图片并绘制修复区域（白色画笔绘制 mask） |
| **Method** | Dropdown | Inpaint or Outpaint (default)、Improve Detail（提升细节）、Modify Content（修改内容） |
| **Inpaint Additional Prompt** | Textbox | 额外提示词，用于描述修复内容（仅在特定模式下可见） |
| **Outpaint Direction** | CheckboxGroup | 扩图方向：Left、Right、Top、Bottom |
| **Additional Prompt Quick List** | Dataset | 快速填充示例提示词（点击自动填入） |
| **Mask Upload** | Image | 上传外部 mask 图片（需勾选“Enable Mask Upload”） |
| **Mask generation model** | Dropdown | 自动生成 Mask 的模型：u2net、sam 等 |
| **Generate mask from image** | Button | 基于上传图片自动生成 Mask |

#### 2.3.4 Describe（反推提示词）

| 控件 | 类型 | 说明 |
| ------ | ------ | ------ |
| **Describe Image** | Image | 上传要分析的图片 |
| **Content Type** | Radio | Photograph（照片风格）或 Art/Anime（艺术/动漫风格） |
| **Describe this Image into Prompt** | Button | 点击后自动反推提示词并填入 Prompt 框 |

#### 2.3.5 Metadata（元数据）

| 控件 | 类型 | 说明 |
| ------ | ------ | ------ |
| **Metadata Image** | Image | 上传 DeFooocus/Fooocus 生成的图片 |
| **Metadata JSON** | JSON | 显示图片中嵌入的元数据 |
| **Apply Metadata** | Button | 将元数据参数加载到 UI 控件中 |

## 三、右栏 —— 设置面板（需勾选“Advanced”）

### 3.1 Settings Tab

#### 步数控制

| 控件 | 类型 | 说明 |
| ------ | ------ | ------ |
| **Steps** | Slider | 采样步数，范围 1-50。当设为 1-10 时自动切换为 LCM 模式 |
| **45 (Quality)** | Button | 快速设为 45 步 |
| **25 (Speed)** | Button | 快速设为 25 步 |
| **10 (Extreme)** | Button | 快速设为 10 步（触发 LCM） |
| **Preset** | Dropdown | 加载预设配置（需 `--disable-preset-selection` 未启用） |

#### 纵横比与采样

| 控件 | 类型 | 说明 |
| ------ | ------ | ------ |
| **Aspect Ratios** | Radio | 预定义的宽高比列表（如 1024×1024、1152×896 等） |
| **Sampling** | Checkbox | 启用后展开采样器/调度器手动选择 |
| **Sampler** | Dropdown | 采样器：euler、dpmpp_2m_sde_gpu、lcm 等 |
| **Scheduler** | Dropdown | 调度器：normal、karras、exponential、sgm_uniform、lcm 等 |

#### 图像数量与提示词

| 控件 | 类型 | 说明 |
| ------ | ------ | ------ |
| **Image Number** | Slider | 单次生成图片数量（1-32） |
| **Negative Prompt** | Textbox | 反向提示词 |
| **Translate Prompts** | Checkbox | 自动将提示词翻译为英文（需联网） |
| **Randomize seed** | Checkbox | 随机种子；取消后可使用下方 Seed 框指定固定种子 |
| **Seed** | Textbox | 固定种子值（取消随机化后可见） |
| **History Log** | HTML | 链接到本地历史记录 HTML 页面 |

---

### 3.2 Models Tab

| 控件 | 类型 | 说明 |
| ------ | ------ | ------ |
| **Base Model** | Dropdown | 选择基础模型（仅 SDXL） |
| **Refiner Model** | Dropdown | 选择精炼器模型（SDXL 或 SD 1.5），可选 None |
| **Refiner Switch At** | Slider | 精炼器切换步数比例（0.1-1.0），如 0.8 表示在 80% 步数时切换 |
| **LoRA 1-5** | Dropdown + Slider | 选择 LoRA 文件及权重（-2 到 2） |
| **Refresh All Files** | Button | 刷新模型文件列表（检测新增/删除的模型） |

---

### 3.3 Advanced Tab

#### 3.3.1 Debug Tools

| 控件 | 类型 | 说明 |
| ------ | ------ | ------ |
| **Guidance Scale** | Slider | CFG 值（1-30），越高风格越鲜明 |
| **Image Sharpness** | Slider | 锐度（0-30），越高纹理越锐利 |
| **Output Format** | Radio | png、jpg、webp |
| **Advanced mode** | Checkbox | 显示/隐藏下方高级调试工具 |

以下控件在 Advanced mode 勾选后显示：

| 控件 | 类型 | 说明 |
| ------ | ------ | ------ |
| **Positive ADM Guidance Scaler** | Slider | 正向 ADM 缩放（0.1-3.0） |
| **Negative ADM Guidance Scaler** | Slider | 负向 ADM 缩放（0.1-3.0） |
| **ADM Guidance End At Step** | Slider | ADM 引导结束位置（0-1） |
| **Refiner swap method** | Dropdown | joint / separate / vae |
| **CFG Mimicking from TSNR** | Slider | TSNR CFG 模拟值（1-30） |
| **Generate Image Grid for Each Batch** | Checkbox | 将多张生成结果拼合为网格图（实验性） |
| **Forced Overwrite of Sampling Step** | Slider | 强制覆盖步数（-1 禁用，>0 覆盖） |
| **Forced Overwrite of Refiner Switch Step** | Slider | 强制覆盖精炼器切换步数 |
| **Forced Overwrite of Generating Width** | Slider | 强制覆盖生成宽度（-1 禁用） |
| **Forced Overwrite of Generating Height** | Slider | 强制覆盖生成高度（-1 禁用） |
| **Overwrite Denoising Strength (Vary)** | Slider | 覆盖 Vary 的降噪强度（-1 禁用） |
| **Overwrite Denoising Strength (Upscale)** | Slider | 覆盖 Upscale 的降噪强度（-1 禁用） |
| **Disable Preview** | Checkbox | 禁用生成过程中的实时预览 |
| **Disable Intermediate Results** | Checkbox | 禁用中间结果展示，仅显示最终画廊 |
| **Black Out NSFW** | Checkbox | 检测到 NSFW 内容时输出黑色图片 |
| **Save Metadata to Images** | Checkbox | 将生成参数嵌入图片元数据 |
| **Metadata Scheme** | Radio | fooocus (json) / a1111 (plain text) |

#### 3.3.2 Control

| 控件 | 类型 | 说明 |
| ------ | ------ | ------ |
| **Debug Preprocessors** | Checkbox | 显示 ControlNet 预处理器的输出结果 |
| **Skip Preprocessors** | Checkbox | 跳过预处理（输入已是边缘图/深度图等） |
| **Mixing Image Prompt and Vary/Upscale** | Checkbox | 在 Vary/Upscale 时同时使用 Image Prompt |
| **Mixing Image Prompt and Inpaint** | Checkbox | 在 Inpaint 时同时使用 Image Prompt |
| **Softness of ControlNet** | Slider | ControlNet 平滑度（0-1），类似于 A1111 的 Control Mode |
| **Canny Low Threshold** | Slider | Canny 低阈值（1-255） |
| **Canny High Threshold** | Slider | Canny 高阈值（1-255） |

#### 3.3.3 Inpaint

| 控件 | 类型 | 说明 |
| ------ | ------ | ------ |
| **Debug Inpaint Preprocessing** | Checkbox | 显示 Inpaint 预处理结果（mask 处理可视化） |
| **Disable initial latent in inpaint** | Checkbox | 禁用 Inpaint 的初始潜空间（使用纯噪声） |
| **Inpaint Engine** | Dropdown | 修复引擎版本：None / v1 / v2.5 / v2.6 |
| **Inpaint Denoising Strength** | Slider | 修复降噪强度（0-1），同 A1111 的 denoising strength |
| **Inpaint Respective Field** | Slider | 修复区域范围（0-1），0=Only Masked，1=Whole Image |
| **Mask Erode or Dilate** | Slider | Mask 腐蚀/膨胀（-64 到 64），正值扩大白色区域 |
| **Enable Mask Upload** | Checkbox | 启用外部 Mask 上传 |
| **Invert Mask** | Checkbox | 反转 Mask（白变黑，黑变白） |

#### 3.3.4 FreeU

| 控件 | 类型 | 说明 |
|------|------|------|
| **Enabled** | Checkbox | 启用 FreeU |
| **B1 / B2** | Slider | FreeU 后端参数（0-2） |
| **S1 / S2** | Slider | FreeU 前端参数（0-4） |

## 四、快速上手指南

### 基本流程

1. 在 **Prompt** 框中输入提示词
2. 点击 **Generate** 生成图像

### 使用图像控制

1. 勾选 **Input Image**
2. 切换到对应 Tab（Upscale/Variation、Image Prompt、Inpaint/Outpaint）
3. 上传图片并配置参数
4. 点击 **Generate**

### LCM 极速模式

- 将 **Steps** 滑块设为 1-10，系统自动启用 LCM
- 或点击 **10 (Extreme)** 快捷按钮
- 生成步数将自动匹配 LCM 采样器

### 加载预设

- 在 **Settings** 中选择 **Preset** 下拉菜单
- 或从 **Metadata** Tab 上传图片加载历史参数
- 或在 Prompt 框中粘贴 JSON 元数据，点击 **Load Parameters**

---

*最后更新：2026-07-27*
