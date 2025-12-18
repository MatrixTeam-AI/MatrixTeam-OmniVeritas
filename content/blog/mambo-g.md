---
title: 'MAMBO-G: Magnitude-Aware Mitigation for Boosted Guidance'
description: 'Introducing a novel strategy to mitigate instability in boosted guidance for diffusion models, improving both quality and speed.'
pubDate: 'Dec 17 2025'
pinned: true
heroImage: '/MatrixTeam-OmniVeritas/mambo-images/header.png'
tags:
  - diffusion-models
  - generative-ai
  - research
  - mambo-g
---

# MAMBO-G: Magnitude-Aware Mitigation for Boosted Guidance

**Authors:** Shangwenzhu Zhileishu Ruilifeng.

## Abstract

Classifier-Free Guidance (CFG) is a crucial technique in modern text-to-image and text-to-video generation, significantly improving the alignment between generated content and text prompts. However, as model scales increase (e.g., SD3, Lumina, WAN2.1), strong guidance often leads to instability, manifesting as oversaturated colors, unnatural structures, and artifacts.

We introduce **MAMBO-G** (Magnitude-Aware Mitigation for Boosted Guidance), a novel adaptive guidance strategy designed to address these challenges. By analyzing the dynamics of guidance in high-dimensional latent spaces, we identify that initial guidance updates often point in a generic direction, leading to "overshoot" when strong guidance is applied early. MAMBO-G mitigates this by dynamically adjusting the guidance scale based on a "Magnitude-Aware Ratio," ensuring stability without compromising text alignment.

### Visual Comparison

<div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 10px;">
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/fig1_baseline.png" alt="Baseline CFG" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>Baseline CFG</strong><br>(Oversaturated)</p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/fig1_ours.png" alt="MAMBO-G (Ours)" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>MAMBO-G</strong><br>(Ours)</p>
  </div>
</div>

<p style="text-align: center; font-style: italic; margin-top: 0;">Figure 1: Comparison between standard CFG (left) and MAMBO-G (right). Both results are generated using 10 sampling steps with a guidance scale of 4. Standard CFG leads to severe oversaturation and artifacts, while MAMBO-G maintains structural integrity and visual quality.</p>

### Instability at High Guidance
Strong guidance is essential for ensuring the generated content adheres to the text prompt. However, as model scales increase, simply boosting the guidance scale often backfires. The generation process collapses, yielding images with **oversaturated colors**, **unnatural high-contrast artifacts**, and **structural disintegration**. This instability is not just a random error but a systematic failure mode in modern high-dimensional diffusion and flow-matching models.

### The "Overshoot" Phenomenon: A Geometric Perspective
Why does strong guidance fail? Our analysis traces the root cause to the initialization phase (Zero-SNR, $t=1$).

1.  **Generic Direction at Initialization:** At the very start ($t=1$), the input is pure Gaussian noise, statistically independent of the target data. Consequently, the model's guidance update vector ($\Delta \mathbf{v}$) depends *solely* on the text prompt, ignoring the specific structure of the initial noise.
2.  **The Conflict:** Empirically, we observe that the guidance direction is nearly identical (Cosine Similarity $\approx 1.0$) across different random seeds at $t=1$. This means the guidance pushes *every* sample in the exact same "generic" direction.
3.  **Manifold Deviation:** In high-dimensional latent spaces, applying a large guidance scale to this generic vector forces the generation trajectory to move aggressively along a path that likely does not align with the specific noise instance's optimal path to the data manifold. This results in a severe **"overshoot,"** driving the state into invalid regions of the latent space from which the model cannot recover.

**MAMBO-G addresses this by detecting when the guidance is acting in this "blind," generic manner and temporarily dampening it.**

### Visual Comparison 2

<div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 10px;">
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/fig2_baseline.png" alt="Baseline CFG (Detailed)" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>Baseline CFG</strong></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/fig2_ours.png" alt="MAMBO-G (Detailed)" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>MAMBO-G</strong></p>
  </div>
</div>

<p style="text-align: center; font-style: italic; margin-top: 0;">Figure 2: Further comparison of detail preservation. Standard CFG (left) often distorts fine details due to overshoot, whereas MAMBO-G (right) preserves intricate textures and lighting effects.</p>
 
## The MAMBO-G Solution

MAMBO-G proposes a magnitude-aware adaptive guidance strategy. The core innovation lies in defining a **Magnitude-Aware Ratio ($r_t$)**:

$$
r_t(\mathbf{x}_t, t) = \frac{\|\mathbf{v}_{\text{cond}}(\mathbf{x}_t, t) - \mathbf{v}_{\text{uncond}}(\mathbf{x}_t, t)\|}{\|\mathbf{v}_{\text{uncond}}(\mathbf{x}_t, t)\|}
$$

We interpret this ratio as a **Coefficient of Variation (CV)** for the diffusion process. It quantifies the relative magnitude of the guidance update (representing the "variation") with respect to the unconditional prediction (representing the "base" or "mean"). A high coefficient implies that the guidance force is overwhelming the intrinsic denoising direction, signaling a potential risk of overshoot and instability.

Based on this ratio, MAMBO-G applies an adaptive damping factor to the guidance scale:

$$
w(r_t) = 1 + w_{\max} \cdot \exp(-\alpha r_t)
$$

This ensures that guidance is suppressed when the risk is high (typically at the very beginning of sampling) and restored as the image structure becomes clearer.

## Key Features & Results

-   **Enhanced Stability:** Effectively suppresses overshoot in early sampling stages, reducing oversaturation and structural collapse.
-   **Accelerated Inference:** Achieves high-quality generation in fewer steps. Experiments show comparable quality to standard CFG with significantly fewer steps (e.g., 3x speedup on SD3.5, 4x on Lumina).
-   **Plug-and-Play Compatibility:** MAMBO-G does not require model retraining. It can be easily integrated into existing DiT (Diffusion Transformer) architectures and samplers.

## Our result on Qwen-Image

### Original Qwen-Image Pipeline

#### Code Example

```python
from diffusers import QwenImagePipeline
import torch

model_name = "Qwen/Qwen-Image"

device = "cuda"
pipe = QwenImagePipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
pipe = pipe.to(device)
prompt = "a comic potrait of a female necromancer with big and cute eyes, fine - face, realistic shaded perfect face, fine details. night setting. very anime style. realistic shaded lighting poster by ilya kuvshinov katsuhiro, magali villeneuve, artgerm, jeremy lipkin and michael garmash, rob rey and kentaro miura style, trending on art station"
negative_prompt = " "
width, height = 1328, 1328
num_inference_steps = 10
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=num_inference_steps,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(1),
    cfg_type="original",
).images[0]
image.save(f"qwenimage_result.png")

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=num_inference_steps,
    true_cfg_scale=10,
    generator=torch.Generator(device="cuda").manual_seed(1),
    cfg_type="mambo_g",
).images[0]
image.save(f"qwenimage_result_mambo.png")
```

#### Results

<div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 10px;">
  <div style="flex: 1; text-align: center;"> 
    <img src="/MatrixTeam-OmniVeritas/mambo-images/qwenimage_result_mambo.png" alt="MAMBO-G 10 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>MAMBO-G 10-Steps</strong></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/qwenimage_result.png" alt="Original 10 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>Baseline 10-Steps</strong></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/qwenimage_result_30.png" alt="Original 30 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>Baseline 30-Steps</strong></p>
  </div>
</div> 

### Qwen-Image Controlnet Pipeline

#### Code Example

```python

from diffusers import QwenImageControlNetPipeline, QwenImageControlNetModel
import torch
from diffusers.utils import load_image

base_model = "Qwen/Qwen-Image"
controlnet_model = "Qwen/Qwen-Image-ControlNet-Union"

controlnet = QwenImageControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)

pipe = QwenImageControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

control_image = load_image("./mambo-images/depth.png")
prompt = "A swanky, minimalist living room with a huge floor-to-ceiling window letting in loads of natural light. A beige couch with white cushions sits on a wooden floor, with a matching coffee table in front. The walls are a soft, warm beige, decorated with two framed botanical prints. A potted plant chills in the corner near the window. Sunlight pours through the leaves outside, casting cool shadows on the floor."
controlnet_conditioning_scale = 1.0
steps = 10

image = pipe(
    prompt=prompt,
    negative_prompt=" ",
    control_image=control_image,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    width=control_image.size[0],
    height=control_image.size[1],
    num_inference_steps=steps,
    true_cfg_scale=4,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
image.save(f"qwenimage_cn_union_result.png")

image = pipe(
    prompt=prompt,
    negative_prompt=" ",
    control_image=control_image,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    width=control_image.size[0],
    height=control_image.size[1],
    num_inference_steps=steps,
    true_cfg_scale=10.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
    cfg_type="mambo_g",
    cfg_kwargs={"debug_ratio": True}
).images[0]
image.save(f"qwenimage_cn_union_result_mambo_g.png")
```

#### Results

<div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 10px;">
  <div style="flex: 1; text-align: center;"> 
    <img src="/MatrixTeam-OmniVeritas/mambo-images/qwenimage_cn_union_result_mambo_g.png" alt="MAMBO-G 10 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>MAMBO-G 10-Steps</strong></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/qwenimage_cn_union_result.png" alt="Original 10 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>Baseline 10-Steps</strong></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/qwenimage_cn_union_result_30.png" alt="Original 30 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>Baseline 30-Steps</strong></p>
  </div>
</div> 

### Qwen-Image Edit Pipeline

#### Code Example

```python

import os
from PIL import Image
import torch

from diffusers import QwenImageEditPipeline

pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)
image = Image.open("./mambo-images/input1.jpg").convert("RGB")
prompt = "Obtain the front view"
inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 5,
}

inputs_mambo_g = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 10.0,
    "negative_prompt": " ",
    "num_inference_steps": 5,
    "cfg_type": "mambo_g"
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("image_edit.png")
    output = pipeline(**inputs_mambo_g)
    output_image = output.images[0]
    output_image.save("image_edit_mambo_g.png")
```
#### Results

<div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 10px;">
  <div style="flex: 1; text-align: center;"> 
    <img src="/MatrixTeam-OmniVeritas/mambo-images/image_edit_mambo_g.png" alt="MAMBO-G 5 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>MAMBO-G 5-Steps</strong></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/image_edit.png" alt="Original 5 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>Baseline 5-Steps</strong></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/image_edit_30.png" alt="Original 15 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>Baseline 15-Steps</strong></p>
  </div>
</div> 

### Qwen-Image Image-to-Image Pipeline

#### Code Example

```python
import torch
from diffusers import QwenImageImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image

pipe = QwenImageImg2ImgPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
url = "./mambo-images/sketch-mountains-input.jpg"
init_image = Image.open(url).resize((1024, 1024))
prompt = "A stunning fantasy landscape, trending on ArtStation. Majestic, sharp mountains dominate the background, under a luminous twilight sky filled with scattered, golden starlight. An ethereal, glowing blue river winds through the middle, separating a rugged, warm-toned desert on the left from a vibrant, emerald-green valley on the right, dotted with dark coniferous trees. A mystical purple castle, with towering spires, rises subtly on the right horizon. The atmosphere is imbued with a soft, magical mist and subtle, radiating energy ripples across the sky and landscape. Rendered as a highly detailed digital painting, focusing on atmospheric perspective, dramatic volumetric lighting, and rich, painterly textures. Epic, immersive, concept art style, sharp focus, 8K, cinematic wide shot."
num_inference_steps=10
images = pipe(prompt=prompt, negative_prompt=" ", image=init_image, strength=0.95, true_cfg_scale=4.0, num_inference_steps=num_inference_steps, generator=torch.Generator(device="cuda").manual_seed(42),).images[0]
images.save("qwenimage_img2img.png")
images = pipe(prompt=prompt, negative_prompt=" ", image=init_image, strength=0.95, true_cfg_scale=10.0, cfg_type="mambo_g", num_inference_steps=num_inference_steps, generator=torch.Generator(device="cuda").manual_seed(42)).images[0]
images.save("qwenimage_img2img_mambo_g.png")
```

#### Results

<div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 10px;">
  <div style="flex: 1; text-align: center;"> 
    <img src="/MatrixTeam-OmniVeritas/mambo-images/qwenimage_img2img_mambo_g.png" alt="MAMBO-G 5 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>MAMBO-G 10-Steps</strong></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/qwenimage_img2img.png" alt="Original 5 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>Baseline 10-Steps</strong></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/qwenimage_img2img_30.png" alt="Original 15 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>Baseline 30-Steps</strong></p>
  </div>
</div> 

### Qwen-Image Edit Plus Pipeline

#### Code Example
```python
import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open("./mambo-images/edit2509_1.jpg")
image2 = Image.open("./mambo-images/edit2509_2.jpg")
prompt = "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square."
inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.manual_seed(42),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 10,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
inputs_mambo = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.manual_seed(42),
    "true_cfg_scale": 10.0,
    "negative_prompt": " ",
    "num_inference_steps": 10,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
    "cfg_type": "mambo_g"
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit_plus.png")
    output = pipeline(**inputs_mambo)
    output_image = output.images[0]
    output_image.save("output_image_edit_plus_mambo.png")
```

#### Results

<div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 10px;">
  <div style="flex: 1; text-align: center;"> 
    <img src="/MatrixTeam-OmniVeritas/mambo-images/output_image_edit_plus_mambo.png" alt="MAMBO-G 5 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>MAMBO-G 10-Steps</strong></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/output_image_edit_plus.png" alt="Original 5 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>Baseline 10-Steps</strong></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/output_image_edit_plus_30.png" alt="Original 15 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>Baseline 30-Steps</strong></p>
  </div>
</div> 


## Conclusion

MAMBO-G offers a theoretically grounded and practically effective solution to the instability problems of Classifier-Free Guidance in high-dimensional generative models. By making guidance magnitude-aware, it unlocks the full potential of large-scale models, enabling faster and higher-quality generation.

