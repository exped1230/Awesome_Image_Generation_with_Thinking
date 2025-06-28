# 💡🎞️ Awesome_Image_Generation_with_Thinking

<div align="center">
  <img src="logo.png" alt="Logo" width="300">
  <h1 align="center">Image Generation with Thinking.</h1>
  
[![Awesome](https://awesome.re/badge.svg)](https://github.com/XiaoYee/Awesome_Efficient_LRM_Reasoning) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/exped1230/Awesome_Image_Generation_with_Thinking?color=green) 

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

- [Multi-modal generation via cross-modal in-context learning](https://arxiv.org/abs/2405.18304) (May, 2024) <br>
[![GitHub stars](https://img.shields.io/github/stars/VIROBO-15/MGCC)](https://github.com/VIROBO-15/MGCC)


- [Emu: Generative pretraining in multimodality](https://arxiv.org/abs/2307.05222) (ICLR, 2024) <br>
[![GitHub stars](https://img.shields.io/github/stars/baaivision/Emu)](https://github.com/baaivision/Emu)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://github.com/baaivision/Emu/tree/main/Emu1#model-weights)


- [DreamLLM: Synergistic multimodal comprehension and creation](https://arxiv.org/abs/2309.11499) (ICLR, 2024) <br>
[![GitHub stars](https://img.shields.io/github/stars/RunpeiDong/DreamLLM)](https://github.com/RunpeiDong/DreamLLM)
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://dreamllm.github.io)


- [Making LLaMA see and draw with seed tokenizer](https://arxiv.org/abs/2310.01218) (ICLR, 2024) <br>
[![GitHub stars](https://img.shields.io/github/stars/AILab-CVC/SEED)](https://github.com/AILab-CVC/SEED)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://github.com/AILab-CVC/SEED/blob/main/SEED-1.md)


- [MiniGPT-5: Interleaved vision-and-language generation via generative vokens](https://arxiv.org/abs/2310.02239) (Mar., 2024) <br>
[![GitHub stars](https://img.shields.io/github/stars/eric-ai-lab/MiniGPT-5)](https://github.com/eric-ai-lab/MiniGPT-5)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://huggingface.co/KZ-ucsc/MiniGPT5)


- [Generative multimodal models are in-context learners](https://arxiv.org/abs/2312.13286) (CVPR, 2024) <br>
[![GitHub stars](https://img.shields.io/github/stars/baaivision/Emu)](https://github.com/baaivision/Emu)
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://baaivision.github.io/emu2/)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://huggingface.co/BAAI/Emu2)


- [Unified-IO 2: Scaling autoregressive multimodal models with vision, language, audio, and action](https://arxiv.org/abs/2312.17172) (CVPR, 2024) <br>
[![GitHub stars](https://img.shields.io/github/stars/allenai/unified-io-2)](https://github.com/allenai/unified-io-2)
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://unified-io-2.allenai.org)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://github.com/allenai/unified-io-2#checkpoints)


- [SEED-X: Multimodal models with unified multi-granularity comprehension and generation](https://arxiv.org/abs/2404.14396) (May, 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/AILab-CVC/SEED-X)](https://github.com/AILab-CVC/SEED-X)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://huggingface.co/AILab-CVC/SEED-X-17B)


- [Chameleon: Mixed-Modal early-fusion foundation models](https://arxiv.org/abs/2404.14396) (May, 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/facebookresearch/chameleon)](https://github.com/facebookresearch/chameleon)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://huggingface.co/collections/facebook/chameleon-668da9663f80d483b4c61f58)


- [Transfusion: Predict the next token and diffuse images with one multi-modal model](https://arxiv.org/abs/2408.11039) (Aug., 2024) <br>
[![GitHub stars](https://img.shields.io/github/stars/lucidrains/transfusion-pytorch)](https://github.com/lucidrains/transfusion-pytorch)


- [Show-o: One single transformer to unify multimodal understanding and generation](https://arxiv.org/abs/2408.12528) (ICLR, 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/showlab/Show-o)](https://github.com/showlab/Show-o)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://huggingface.co/showlab)


- [VILA-U: A unified foundation model integrating visual understanding and generation](https://arxiv.org/abs/2409.04429) (Mar., 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/mit-han-lab/vila-u)](https://github.com/mit-han-lab/vila-u)
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://hanlab.mit.edu/projects/vila-u)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://huggingface.co/collections/mit-han-lab/vila-u-7b-6716f7dd5331e4bdf944ffa6)


- [Emu3: Next-token prediction is all you need](https://arxiv.org/abs/2409.18869) (Sep., 2024) <br>
[![GitHub stars](https://img.shields.io/github/stars/baaivision/Emu3)](https://github.com/baaivision/Emu3)
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://emu.baai.ac.cn/about)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://huggingface.co/collections/BAAI/emu3-66f4e64f70850ff358a2e60f)


- [Janus: Decoupling visual encoding for unified multimodal understanding and generation](https://arxiv.org/abs/2410.13848) (Oct., 2024) <br>
[![GitHub stars](https://img.shields.io/github/stars/deepseek-ai/Janus)](https://github.com/deepseek-ai/Janus)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://github.com/deepseek-ai/Janus#2-model-download)


- [JanusFlow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation](https://arxiv.org/abs/2411.07975) (Mar., 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/deepseek-ai/Janus)](https://github.com/deepseek-ai/Janus)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://github.com/deepseek-ai/Janus#2-model-download)


- [TokenFlow: Unified image tokenizer for multimodal understanding and generation](https://arxiv.org/abs/2412.03069) (CVPR, 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/ByteFlow-AI/TokenFlow)](https://github.com/ByteFlow-AI/TokenFlow)
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://byteflow-ai.github.io/TokenFlow/)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://huggingface.co/ByteFlow-AI)


- [MetaMorph: Multimodal understanding and generation via instruction tuning](https://arxiv.org/abs/2412.14164) (Dec., 2024) <br>
[![GitHub stars](https://img.shields.io/github/stars/facebookresearch/metamorph)](https://github.com/facebookresearch/metamorph/)
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://tsb0601.github.io/metamorph/)


- [LMFusion: Adapting pretrained language models for multimodal generation](https://arxiv.org/abs/2412.15188) (Feb., 2025) <br>


- [Janus-Pro: Unified multimodal understanding and generation with data and model scaling](https://arxiv.org/abs/2501.17811) (Jan., 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/deepseek-ai/Janus)](https://github.com/deepseek-ai/Janus)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://github.com/deepseek-ai/Janus?tab=readme-ov-file#2-model-download)


- [MINT: Multi-modal chain of thought in unified generative models for enhanced image generation](https://arxiv.org/abs/2503.01298) (Mar., 2025) <br>


- [Transfer between modalities with metaqueries](https://arxiv.org/abs/2504.06256) (Apr., 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/facebookresearch/metaquery)](https://github.com/facebookresearch/metaquery)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://xichenpan.com/metaquery/)
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://huggingface.co/collections/xcpan/metaquery-instruction-tuning-data-685b0f16d81ce54bcb7ea3a8)


- [BLIP3-o: A family of fully open unified multimodal models-architecture, training and dataset](https://arxiv.org/abs/2505.09568) (May, 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/JiuhaiChen/BLIP3o)](https://github.com/JiuhaiChen/BLIP3o)
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://huggingface.co/BLIP3o/BLIP3o-Model)
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://huggingface.co/BLIP3o/datasets)


- [Emerging properties in unified multimodal pretraining](https://arxiv.org/abs/2505.14683) (May, 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/bytedance-seed/BAGEL)](https://github.com/bytedance-seed/BAGEL)
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://bagel-ai.org)
[![Model](https://img.shields.io/badge/Model-Available-orange?style=flat-square)](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)


- [Thinking with generated images](https://arxiv.org/abs/2505.22525) (May, 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/GAIR-NLP/thinking-with-generated-images)](https://github.com/GAIR-NLP/thinking-with-generated-images)
[![Model](https://img.shields.io/badge/Model1-Available-orange?style=flat-square)](https://huggingface.co/GAIR/twgi-critique-anole-7b)
[![Model](https://img.shields.io/badge/Model2-Available-orange?style=flat-square)](https://huggingface.co/GAIR/twgi-subgoal-anole-7b)


- [Show-o2: Improved native unified multimodal models](https://arxiv.org/abs/2506.15564) (Jun., 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/showlab/Show-o)](https://github.com/showlab/Show-o)
[![Model](https://img.shields.io/badge/Model1-Available-orange?style=flat-square)](https://github.com/showlab/Show-o/tree/main/show-o2#pre-trained-model-weigths)


- [ShareGPT-4o-Image: Aligning multimodal models with GPT-4o-level image generation](https://arxiv.org/abs/2506.18095) (Jun., 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/FreedomIntelligence/ShareGPT-4o-Image)](https://github.com/FreedomIntelligence/ShareGPT-4o-Image)
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image)
[![Model](https://img.shields.io/badge/Model1-Available-orange?style=flat-square)](https://huggingface.co/FreedomIntelligence/Janus-4o-7B)


- [OmniGen2: Exploration to advanced multimodal generation](https://arxiv.org/abs/2506.18871) (Jun., 2025) <br>
[![GitHub stars](https://img.shields.io/github/stars/VectorSpaceLab/OmniGen2)](https://github.com/VectorSpaceLab/OmniGen2)
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://huggingface.co/datasets/OmniGen2/OmniContext)
[![Model](https://img.shields.io/badge/Model1-Available-orange?style=flat-square)](https://huggingface.co/OmniGen2/OmniGen2)

---


## 📚 Benchmarks
*Essential resources for understanding the broader landscape and evaluating progress in visual reasoning.*

- [ELLA: Equip diffusion models with LLM for enhanced semantic alignment](https://arxiv.org/abs/2403.05135) (Mar., 2024) <br>
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://ella-diffusion.github.io)
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://github.com/TencentQQGYLab/ELLA?tab=readme-ov-file#-dpg-bench)


- [T2I-CompBench: A comprehensive benchmark for open-world compositional text-to-image generation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f8ad010cdd9143dbb0e9308c093aff24-Abstract-Datasets_and_Benchmarks.html) (NeurIPS, 2023) <br>
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://karine-h.github.io/T2I-CompBench/)
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://connecthkuhk-my.sharepoint.com/personal/huangky_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhuangky%5Fconnect%5Fhku%5Fhk%2FDocuments%2FT2I%2DCompBench&ga=1)


- [GenEval: An object-focused framework for evaluating text-to-image alignment](https://arxiv.org/abs/2310.11513) (NeurIPS, 2023) <br>
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://github.com/djghosh13/geneval#image-generation)


- [Commonsense-T2I challenge: Can text-to-image generation models understand commonsense?](https://openreview.net/forum?id=MI52iXSSNy) (COLM, 2024) <br>
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://zeyofu.github.io/CommonsenseT2I/)
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://huggingface.co/datasets/CommonsenseT2I/CommonsensenT2I)


- [WISE: A world knowledge-informed semantic evaluation for text-to-image generation](https://arxiv.org/abs/2503.07265) (May, 2025) <br>
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://github.com/PKU-YuanGroup/WISE#wise-eval)


- [TIIF-Bench: How does your T2I model follow your instructions?](https://arxiv.org/abs/2506.02161) (Jun., 2025) <br>
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://a113n-w3i.github.io/TIIF_Bench/)
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://huggingface.co/datasets/A113NW3I/TIIF-Bench-Data)


- [OneIG-Bench: Omni-dimensional nuanced evaluation for image generation](https://arxiv.org/abs/2506.07977) (Jun., 2025) <br>
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=flat-square)](https://oneig-bench.github.io)
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?style=flat-square)](https://huggingface.co/datasets/OneIG-Bench/OneIG-Bench)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=exped1230/Awesome_Image_Generation_with_Thinking&type=Date)](https://www.star-history.com/#exped1230/Awesome_Image_Generation_with_Thinking&Date)



