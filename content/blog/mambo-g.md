---
title: 'MAMBO-G: Magnitude-Aware Mitigation for Boosted Guidance'
description: 'Introducing a novel strategy to mitigate instability in boosted guidance for diffusion models, improving both quality and speed.'
pubDate: 'Dec 17 2025'
pinned: true
heroImage: '/Matrix-omini/mambo-images/header.png'
tags:
  - diffusion-models
  - generative-ai
  - research
  - mambo-g
---

# MAMBO-G: Magnitude-Aware Mitigation for Boosted Guidance

**Authors:** Shangwenzhu et al.

## Abstract

Classifier-Free Guidance (CFG) is a crucial technique in modern text-to-image and text-to-video generation, significantly improving the alignment between generated content and text prompts. However, as model scales increase (e.g., SD3, Lumina, WAN2.1), strong guidance often leads to instability, manifesting as oversaturated colors, unnatural structures, and artifacts.

We introduce **MAMBO-G** (Magnitude-Aware Mitigation for Boosted Guidance), a novel adaptive guidance strategy designed to address these challenges. By analyzing the dynamics of guidance in high-dimensional latent spaces, we identify that initial guidance updates often point in a generic direction, leading to "overshoot" when strong guidance is applied early. MAMBO-G mitigates this by dynamically adjusting the guidance scale based on a "Magnitude-Aware Ratio," ensuring stability without compromising text alignment.

### Visual Comparison

<div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 10px;">
  <div style="flex: 1; text-align: center;">
    <img src="/Matrix-omini/mambo-images/fig1_baseline.png" alt="Baseline CFG" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>Baseline CFG</strong><br>(Oversaturated)</p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/Matrix-omini/mambo-images/fig1_ours.png" alt="MAMBO-G (Ours)" style="width: 100%; border-radius: 8px;">
    <p style="margin-top: 5px;"><strong>MAMBO-G</strong><br>(Ours)</p>
  </div>
</div>

<p style="text-align: center; color: #666; font-style: italic; margin-top: 0;">Figure 1: Comparison between standard CFG (left) and MAMBO-G (right). Both results are generated using 10 sampling steps with a guidance scale of 4. Standard CFG leads to severe oversaturation and artifacts, while MAMBO-G maintains structural integrity and visual quality.</p>

## Key Challenges

Recent large-scale diffusion and flow-matching models face specific challenges with CFG:

1.  **Instability at High Guidance:** Strong guidance, while necessary for prompt adherence, often collapses the generation process into artifacts or oversaturated images.
2.  **The "Overshoot" Phenomenon:** In high-dimensional spaces, the initial noise ($t=1$) is statistically independent of the target data. The guidance update at this stage tends to be a generic direction determined solely by the text prompt. Applying large CFG scales here forces the generation trajectory to deviate significantly from the data manifold.

![Interval Analysis](/Matrix-omini/mambo-images/fig2_interval.png)
*Figure 2: Analysis of guidance interval impact. The plot illustrates how guidance strength varies across different timesteps, highlighting the critical early stage where overshoot is most likely to occur.*

## The MAMBO-G Solution

MAMBO-G proposes a magnitude-aware adaptive guidance strategy. The core innovation lies in defining a **Magnitude-Aware Ratio ($r_t$)**:

$$
r_t(\mathbf{x}_t, t) = \frac{\|\mathbf{v}_{\text{cond}}(\mathbf{x}_t, t) - \mathbf{v}_{\text{uncond}}(\mathbf{x}_t, t)\|}{\|\mathbf{v}_{\text{uncond}}(\mathbf{x}_t, t)\|}
$$

This ratio measures the relative strength of the "external guidance force" against the "internal denoising force" (using the **unconditional prediction as the base**). A high ratio indicates a potential risk of overshoot.

Based on this ratio, MAMBO-G applies an adaptive damping factor to the guidance scale:

$$
w(r_t) = 1 + w_{\max} \cdot \exp(-\alpha r_t)
$$

This ensures that guidance is suppressed when the risk is high (typically at the very beginning of sampling) and restored as the image structure becomes clearer.

## Key Features & Results

-   **Enhanced Stability:** Effectively suppresses overshoot in early sampling stages, reducing oversaturation and structural collapse.
-   **Accelerated Inference:** Achieves high-quality generation in fewer steps. Experiments show comparable quality to standard CFG with significantly fewer steps (e.g., 3x speedup on SD3.5, 4x on Lumina).
-   **Plug-and-Play Compatibility:** MAMBO-G does not require model retraining. It can be easily integrated into existing DiT (Diffusion Transformer) architectures and samplers.

## Conclusion

MAMBO-G offers a theoretically grounded and practically effective solution to the instability problems of Classifier-Free Guidance in high-dimensional generative models. By making guidance magnitude-aware, it unlocks the full potential of large-scale models, enabling faster and higher-quality generation.

---

*This project is supported by National Natural Science Foundation of China and National Key Research Projects.*
