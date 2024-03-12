# FLATTEN: optical FLow-guided ATTENtion for consistent text-to-video editing
[![arXiv](https://img.shields.io/badge/arXiv-2310.05922-b31b1b.svg)](https://arxiv.org/abs/2310.05922)
[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://flatten-video-editing.github.io/) 
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fyrcong%2Fflatten%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)

**Pytorch Implementation of "FLATTEN: optical FLow-guided ATTENtion for consistent text-to-video editing".**

🎊🎊🎊 We are proud to announce that our paper has been accepted at **ICLR 2024**! If you are interested in FLATTEN, please give us a star😬
![teaser-ezgif com-resize](https://github.com/yrcong/flatten/assets/47991543/4f92f2bd-e4e9-4710-82b3-6efd36c27f46)

Thanks to @[**logtd**](https://github.com/logtd) for integrating FLATTEN into ComfyUI and the great sampled videos! **Here is the [Link](https://github.com/logtd/ComfyUI-FLATTEN?tab=readme-ov-file)!**

https://github.com/yrcong/flatten/assets/47991543/1ad49092-9133-42d0-984f-38c6427bde34


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

## Usage
For text-to-video edting, a source video and a textual prompt should be given. You can run the script to get the teaser video easily:
```
sh cat.sh
```
or with the command:
```
python inference.py \
--prompt "A Tiger, high quality" \
--neg_prompt "a cat with big eyes, deformed" \
--guidance_scale 20 \
--video_path "data/puff.mp4" \
--output_path "outputs/" \
--video_length 32 \
--width 512 \
--height 512 \
--old_qk 0 \
--frame_rate 2 \
```

## Editing tricks
-  You can use a negative prompt (NP) when there is a big gap between the edit target and the source (1st row).
-  You can increase the scale of classifier-free guidance to enhance the semantic alignment (2nd row).

<table class="center">
<tr>
  <td width=30% align="center"><img src="data/source.gif" raw=true></td>
  <td width=30% align="center"><img src="data/tiger_empty.gif" raw=true></td>
	<td width=30% align="center"><img src="data/tiger_neg.gif" raw=true></td>
</tr>
<tr>
  <td width=30% align="center">Source video</td>
  <td width=30% align="center">NP: " "</td>
  <td width=30% align="center">NP: "A cat with big eyes, deformed."</td>
</tr>
<tr>
  <td width=30% align="center"><img src="data/guidance10.gif" raw=true></td>
  <td width=30% align="center"><img src="data/guidance17.5.gif" raw=true></td>
	<td width=30% align="center"><img src="data/guidance20.gif" raw=true></td>
</tr>
<tr>
  <td width=30% align="center">Classifier-free guidance: 10</td>
  <td width=30% align="center">Classifier-free guidance: 17.5</td>
  <td width=30% align="center">Classifier-free guidance: 25</td>
</tr>
</table>


## BibTex
```
@article{cong2023flatten,
  title={FLATTEN: optical FLow-guided ATTENtion for consistent text-to-video editing},
  author={Cong, Yuren and Xu, Mengmeng and Simon, Christian and Chen, Shoufa and Ren, Jiawei and Xie, Yanping and Perez-Rua, Juan-Manuel and Rosenhahn, Bodo and Xiang, Tao and He, Sen},
  journal={arXiv preprint arXiv:2310.05922},
  year={2023}
}
```
