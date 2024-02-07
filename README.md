# FLATTEN: optical FLow-guided ATTENtion for consistent text-to-video editing
[![arXiv](https://img.shields.io/badge/arXiv-2310.05922-b31b1b.svg)](https://arxiv.org/abs/2310.05922)
[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://flatten-video-editing.github.io/) 
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fyrcong%2Fflatten%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)

Pytorch Implementation of "FLATTEN: optical FLow-guided ATTENtion for consistent text-to-video editing".

🎊🎊🎊 We are proud to announce that our paper has been accepted at ICLR 2024!

https://github.com/yrcong/FLATTEN_video_editing/assets/47991543/baa18b99-512e-4e05-a751-30cb1a6f9aa2

## 📖Abstract
🚩**Text-to-Video** 🚩**Training-free** 🚩**Plug-and-Play**<br>

Text-to-video editing aims to edit the visual appearance of a source video conditional on textual prompts. A major challenge in this task is to ensure that all frames in the edited video are visually consistent. In this work, for the first time, we introduce optical flow into the attention module in the diffusion model's U-Net to address the inconsistency issue for text-to-video editing. Our method, FLATTEN, enforces the patches on the same flow path across different frames to attend to each other in the attention module, thus improving the visual consistency in the edited videos. Additionally, our method is training-free and can be seamlessly integrated into any diffusion-based text-to-video editing methods and improve their visual consistency.

## Requirements
First you can download Stable Diffusion 2.1 [here](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).

Install the following packages:
- PyTorch == 2.1
- accelerate == 0.24.1
- diffusers == 0.19.0
- transformers == 4.35.0
- xformers == 0.0.23

## Video Editing
For text-to-video edting, a source video and a textual prompt should be given. An example is provided:
```
sh cat1.sh
```

## BibTex
```
@article{cong2023flatten,
  title={FLATTEN: optical FLow-guided ATTENtion for consistent text-to-video editing},
  author={Cong, Yuren and Xu, Mengmeng and Simon, Christian and Chen, Shoufa and Ren, Jiawei and Xie, Yanping and Perez-Rua, Juan-Manuel and Rosenhahn, Bodo and Xiang, Tao and He, Sen},
  journal={arXiv preprint arXiv:2310.05922},
  year={2023}
}
```
