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

## Our result on Qwen-Image (via Modular Diffusers & Guider)

### Original Qwen-Image Pipeline

#### Code Example

```python
import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.qwenimage import TEXT2IMAGE_BLOCKS
from diffusers.guiders import MagnitudeAwareGuidance, ClassifierFreeGuidance
blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)

modular_repo_id = "YiYiXu/QwenImage-modular"
pipeline = blocks.init_pipeline(modular_repo_id)
pipeline.load_components(torch_dtype=torch.bfloat16)
pipeline.to("cuda")
prompt = "a comic potrait of a female necromancer with big and cute eyes, fine - face, realistic shaded perfect face, fine details. night setting. very anime style. realistic shaded lighting poster by ilya kuvshinov katsuhiro, magali villeneuve, artgerm, jeremy lipkin and michael garmash, rob rey and kentaro miura style, trending on art station"
width, height = 1328, 1328
num_inference_steps = 10 
# num_inference_steps = 30
seed = 1
guider = ClassifierFreeGuidance(guidance_scale=4.0)
pipeline.update_components(guider=guider)
image = pipeline(prompt=prompt, width=width, height=height, output="images", num_inference_steps=num_inference_steps, generator=torch.Generator("cuda").manual_seed(seed))[0]
image.save(f"t2v_original_{num_inference_steps}_steps.png")
guider = MagnitudeAwareGuidance(guidance_scale=10.0, alpha=8.0, guidance_rescale=1.0)
pipeline.update_components(guider=guider)
image = pipeline(prompt=prompt, width=width, height=height, output="images", num_inference_steps=num_inference_steps, generator=torch.Generator("cuda").manual_seed(seed))[0]
image.save(f"t2v_mambo_{num_inference_steps}_steps.png")
```

#### Results

<div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 10px;">
  <div style="flex: 1; text-align: center;"> 
    <img src="/MatrixTeam-OmniVeritas/mambo-images/t2v_mambo_10_steps.png" alt="MAMBO-G 10 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>MAMBO-G 10-Steps</strong></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/t2v_original_10_steps.png" alt="Original 10 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>Baseline 10-Steps</strong></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/t2v_mambo_30_steps.png" alt="MAMBO-G 30 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>MAMBO-G 30-Steps</strong></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/MatrixTeam-OmniVeritas/mambo-images/t2v_original_30_steps.png" alt="Original 30 Steps" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>Baseline 30-Steps</strong></p>
  </div>
</div> 

## Conclusion

MAMBO-G offers a theoretically grounded and practically effective solution to the instability problems of Classifier-Free Guidance in high-dimensional generative models. By making guidance magnitude-aware, it unlocks the full potential of large-scale models, enabling faster and higher-quality generation.

