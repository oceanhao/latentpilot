<div align="center">

<h1><span style="background: linear-gradient(to right, #007BA7, #99B5D2); -webkit-background-clip: text; color: transparent;font-style: italic;"> LatentPilot </span>: Scene-Aware Vision-and-Language Navigation by Dreaming Ahead with Latent Visual Reasoning </h1>

<div>
    <a href='https://abdd.top' target='_blank'>Haihong Hao<sup>🍕</sup></a>;
    Lei Chen<sup>🍕</sup></a>;
    <a href='https://mingfei.info' target='_blank'>Mingfei Han<sup>🌭</sup></a>;
    <a href='https://scholar.google.com/citations?hl=en&user=RLAgwBkAAAAJ' target='_blank'>Changlin Li<sup>🍔</sup></a>;
    <a href='https://marsaki.github.io/' target='_blank'>Dong An<sup>⭐</sup></a>;
        <a href='https://yuqiang-yang.github.io/' target='_blank'>Yuqiang Yang<sup>🌈</sup></a>;
    <a href='https://www.zhihui.li/' target='_blank'>Zhihui Li<sup>🍕</sup></a>;
    <a href='https://www.xiaojun.ai/' target='_blank'>Xiaojun Chang<sup>🍕</sup></a>
</div>
<sup>🍕</sup>University of Science and Technology of China
<sup>🌭</sup>MBZUAI
<sup>🍔</sup>Stanford University
<sup>⭐</sup>Amap, Alibaba Group
<sup>🌈</sup>Shanghai AI Laboratory
<br>


<!-- <p align="center">
    <img src="assets/icon.png" width="30%"><br>
</p>
</div> -->

<!-- <div>
    <a href='https://arxiv.org/abs/2505.16663' target='_blank'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</div> -->


## 🍹 Abstract

Existing vision-and-language navigation (VLN) models primarily reason over past and current visual observations, while largely ignoring the future visual dynamics induced by actions. As a result, they often lack an effective understanding of the causal relationship between actions and how the visual world changes, limiting robust decision-making. Humans, in contrast, can "imagine" the near future by leveraging action-dynamics causality, which improves both environmental understanding and navigation choices. Inspired by this capability, we propose LatentPilot, a new paradigm that exploits future observations during training as a valuable data source to learn action-conditioned visual dynamics, while requiring no access to future frames at inference. Concretely, we propose a flywheel-style training mechanism that iteratively collects on-policy trajectories and retrains the model to better match the agent's behavior distribution, with an expert takeover triggered when the agent deviates excessively. LatentPilot further learns visual latent tokens without explicit supervision; these latent tokens attend globally in a continuous latent space and are carried across steps, serving as both the current output and the next input, thereby enabling the agent to "dream ahead" and reason about how actions will affect subsequent observations. Experiments on R2R-CE, RxR-CE, and R2R-PE benchmarks achieve new SOTA results, and real-robot tests across diverse environments demonstrate LatentPilot's superior understanding of environment-action dynamics in scene.

## 🍻 TODOs

- [ ] Release inference code.
- [ ] Release models weights.
- [ ] Release data preparation scripts.

<!-- ## 🥂 Acknowledgements

We extend our gratitude to MatterPort 3D for their valuable contributions to the open-source platform and community. -->

<!-- 
## 🍺 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{hao2025conav,
  title={CoNav: Collaborative Cross-Modal Reasoning for Embodied Navigation},
  author={Hao, Haihong and Han, Mingfei and Li, Changlin and Li, Zhihui and Chang, Xiaojun},
  journal={arXiv preprint arXiv:2505.16663},
  year={2025}
}
``` -->
<div style="width: 100%; display: flex; justify-content: center; align-items: center; margin: 20px 0;">
    <span style="font-size: 1.2em; font-weight: bold; color: #444;">
        Thank you (.❛ ᴗ ❛.)
    </span>
</div>

<!-- ## 📖 Example 

![Alt text](assets/example_5.gif) -->
