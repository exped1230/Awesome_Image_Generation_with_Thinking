# 💡🎞️ Awesome_Image_Generation_with_Thinking

<div align="center">
  <img src="logo.png" alt="Logo" width="300">
  <h1 align="center">Image Generation with Thinking.</h1>
  
[![Awesome](https://awesome.re/badge.svg)](https://github.com/XiaoYee/Awesome_Efficient_LRM_Reasoning) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/zhaochen0110/Awesome_Think_With_Images?color=green) 

</div>

Welcome to the Awesome-Image-Generation-with-Thinking repository! This repository represents a comprehensive collection of research focused on empowering models to think during image generation. We delve into how these sophisticated models are evolving beyond mere pattern recognition, acquiring capabilities for intricate reasoning, nuanced understanding, and dynamic interaction by processing and interpreting visual information in cognitive-inspired ways.

This collection is for researchers, developers, and enthusiasts eager to explore the forefront of:
*   **Prompt-Based Innovation:** How LVLMs can guide visual understanding and generation.
*   **Supervised Fine-Tuning:** Training models with rich, contextual visual data.
*   **Reinforcement Learning:** Enabling agents to learn through visual interaction and feedback.


---

## 🔔 News

- [2025-06] We created this repository to maintain a paper list on Awesome-Think-With-Images. Contributions are welcome!

- [2025-05] We are excited to release **[OpenThinkIMG](https://github.com/OpenThinkIMG/OpenThinkIMG)**, the first dedicated end-to-end open-source framework designed to empower LVLMs to truly **think with images**! For ease of use, we've configured a Docker environment. We warmly invite the community to explore, use, and contribute.

---

## 📜 Table of Contents

*   [✍️ Editing-based explicitly reflection](#-editing-based-explicitly-reflection)
*   [🗒️ Reasoning with prompt](#-reasoning-with-prompt)
*   [🏆 RL for self-evolution](#-RL-for-self-evolution)
*   [📚 Benchmarks](#-benchmarks)

---


## 📖 Survey

- [Delving into RL for image generation with CoT: A study on DPO vs. GRPO](https://arxiv.org/abs/2505.17017) (May, 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/ZiyuGuo99/Image-Generation-CoT)](https://github.com/ZiyuGuo99/Image-Generation-CoT)


## 🚀 Editing-based explicitly reflection

Unlocking visual intelligence through the art and science of prompting. These methods explore how carefully crafted textual or visual cues can guide LVLMs to perform complex reasoning tasks with images, often without explicit task-specific training.

- [Visual programming: Compositional visual reasoning without training](https://arxiv.org/abs/2211.11559) (CVPR, 2023) <br>
[![GitHub stars](https://img.shields.io/github/stars/allenai/visprog)](https://github.com/allenai/visprog)
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://prior.allenai.org/projects/visprog)


- [ViperGPT: Visual inference via python execution for reasoning](https://arxiv.org/abs/2303.08128) (ICCV, 2023) <br>
[![GitHub stars](https://img.shields.io/github/stars/cvlab-columbia/viper)](https://github.com/cvlab-columbia/viper)


- [From reflection to perfection: Scaling inference-time optimization for text-to-image diffusion models via reflection tuning](https://arxiv.org/abs/2504.16080) (Apr., 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/Diffusion-CoT/ReflectionFlow)](https://github.com/Diffusion-CoT/ReflectionFlow)
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://diffusion-cot.github.io/reflection2perfection/)
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://huggingface.co/collections/diffusion-cot/reflectionflow-release-6803e14352b1b13a16aeda44)


- [GoT: Unleashing reasoning capability of multimodal large language model for visual generation and editing](https://arxiv.org/abs/2503.10639) (Mar., 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/rongyaofang/GoT)](https://github.com/rongyaofang/GoT)
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://github.com/rongyaofang/GoT#released-datasets)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://github.com/rongyaofang/GoT#released-model-got-framework)


- [Visual planning: Let's think only with images](https://arxiv.org/abs/2505.11409) (Mar., 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/yix8/VisualPlanning)](https://github.com/yix8/VisualPlanning)


## 🏆 RL for Self-Evolution

Harnessing the power of Reinforcement Learning to teach models how to reason with images through trial, error, and reward. These approaches enable agents to learn complex visual behaviors, tool interactions, and even intrinsic motivation for exploration.

- [Can we generate images with CoT? Let’s verify and reinforce image generation step by step](https://arxiv.org/abs/2501.13926) (Jan., 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/ZiyuGuo99/Image-Generation-CoT)](https://github.com/ZiyuGuo99/Image-Generation-CoT)
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://huggingface.co/datasets/ZiyuG/Image-Generation-CoT)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://huggingface.co/ZiyuG/Image-Generation-CoT)


- [ImageGen-CoT: Enhancing text-to-image in-context learning with chain-of-thought reasoning](https://arxiv.org/abs/2503.19312) (Jan., 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/ZiyuGuo99/Image-Generation-CoT)](https://github.com/ZiyuGuo99/Image-Generation-CoT)
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://imagegen-cot.github.io)


- [SimpleAR: Pushing the frontier of autoregressive visual generation through pretraining, SFT, and RL](https://arxiv.org/abs/2504.11455) (Apr., 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/wdrink/SimpleAR)](https://github.com/wdrink/SimpleAR)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://huggingface.co/collections/Daniel0724/simplear-6805053f5b4b9961ac025136)


- [T2I-R1: Reinforcing image generation with collaborative semantic-level and token-level CoT](https://arxiv.org/abs/2505.00703) (May, 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/CaraJ7/T2I-R1)](https://github.com/CaraJ7/T2I-R1)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://huggingface.co/CaraJ/T2I-R1)


- [Flow-GRPO: Training flow matching models via online RL](https://arxiv.org/abs/2505.05470) (May, 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/yifan123/flow_grpo)](https://github.com/yifan123/flow_grpo)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://github.com/yifan123/flow_grpo?tab=readme-ov-file#-model)


- [DanceGRPO: Unleashing GRPO on visual generation](https://arxiv.org/abs/2505.07818) (May, 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/XueZeyue/DanceGRPO)](https://github.com/XueZeyue/DanceGRPO)
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://dancegrpo.github.io)


- [GoT-R1: Unleashing reasoning capability of MLLM for visual generation with reinforcement learning](https://arxiv.org/abs/2505.17022) (May, 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/gogoduan/GoT-R1)](https://github.com/gogoduan/GoT-R1)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://github.com/gogoduan/GoT-R1#released-model-got-r1)


- [Co-Reinforcement learning for unified multimodal understanding and generation](https://arxiv.org/abs/2505.17534) (May, 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/mm-vl/ULM-R1)](https://github.com/mm-vl/ULM-R1)
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://huggingface.co/collections/mm-vl/corl-67e0f23d6ecbdc3a9fb747e9)


- [ReasonGen-R1: CoT for autoregressive image generation model through SFT and RL](https://arxiv.org/abs/2505.24875) (May, 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/Franklin-Zhang0/ReasonGen-R1)](https://github.com/Franklin-Zhang0/ReasonGen-R1)
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://reasongen-r1.github.io)
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://huggingface.co/collections/Franklin0/reasongen-r1-6836ed61fc4f6db543c0d368)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://huggingface.co/collections/Franklin0/reasongen-r1-6836ed61fc4f6db543c0d368)

---


## 🎓 Reasoning with prompt

Tailoring pre-trained models for visual reasoning through targeted fine-tuning on specialized datasets. This approach leverages instruction-following data and demonstrations of reasoning steps to enhance model capabilities.

- [Generating images with multimodal language models](https://arxiv.org/abs/2305.17216) ![](https://img.shields.io/badge/abs-2023.05-red)

---


## 📚 Benchmarks
*Essential resources for understanding the broader landscape and evaluating progress in visual reasoning.*

- [A Cognitive Evaluation Benchmark of Image Reasoning and Description for Large Vision-Language Models](https://arxiv.org/abs/2402.18409) ![](https://img.shields.io/badge/abs-2024.02-red)


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=exped1230/Awesome_Image_Generation_with_Thinking&type=Date)](https://www.star-history.com/#exped1230/Awesome_Image_Generation_with_Thinking&Date)



