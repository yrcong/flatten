import os
import argparse
import torch
import torchvision
from einops import rearrange
from diffusers import DDIMScheduler, AutoencoderKL, DDIMInverseScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from models.pipeline_flatten import FlattenPipeline
from models.util import save_videos_grid, read_video, sample_trajectories
from models.unet import UNet3DConditionModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Textual prompt for video editing")
    parser.add_argument("--neg_prompt", type=str, required=True, help="Negative prompt for guidance")
    parser.add_argument("--guidance_scale", default=10.0, type=float, help="Guidance scale")
    parser.add_argument("--video_path", type=str, required=True, help="Path to a source video")
    parser.add_argument("--sd_path", type=str, default="checkpoints/stable-diffusion-2-1-base", help="Path of Stable Diffusion")
    parser.add_argument("--output_path", type=str, default="./outputs", help="Directory of output")
    parser.add_argument("--video_length", type=int, default=15, help="Length of output video")
    parser.add_argument("--old_qk", type=int, default=0, help="Whether to use old queries and keys for flow-guided attention")
    parser.add_argument("--height", type=int, default=512, help="Height of synthesized video, and should be a multiple of 32")
    parser.add_argument("--width", type=int, default=512, help="Width of synthesized video, and should be a multiple of 32")
    parser.add_argument("--sample_steps", type=int, default=50, help="Steps for feature injection")
    parser.add_argument("--inject_step", type=int, default=40, help="Steps for feature injection")
    parser.add_argument("--seed", type=int, default=66, help="Random seed of generator")
    parser.add_argument("--frame_rate", type=int, default=None, help="The frame rate of loading input video. Default rate is computed according to video length.")
    parser.add_argument("--fps", type=int, default=15, help="FPS of the output video")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)
    device = "cuda"
    # Height and width should be 512
    args.height = (args.height // 32) * 32
    args.width = (args.width // 32) * 32

    tokenizer = CLIPTokenizer.from_pretrained(args.sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_path, subfolder="text_encoder").to(dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder="vae").to(dtype=torch.float16)
    unet = UNet3DConditionModel.from_pretrained_2d(args.sd_path, subfolder="unet").to(dtype=torch.float16)
    scheduler=DDIMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
    inverse=DDIMInverseScheduler.from_pretrained(args.sd_path, subfolder="scheduler")

    pipe = FlattenPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=scheduler, inverse_scheduler=inverse)
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # read the source video
    video = read_video(video_path=args.video_path, video_length=args.video_length,
                       width=args.width, height=args.height, frame_rate=args.frame_rate)
    original_pixels = rearrange(video, "(b f) c h w -> b c f h w", b=1)
    save_videos_grid(original_pixels, os.path.join(args.output_path, "source_video.mp4"), rescale=True)

    t2i_transform = torchvision.transforms.ToPILImage()
    real_frames = []
    for i, frame in enumerate(video):
        real_frames.append(t2i_transform(((frame+1)/2*255).to(torch.uint8)))

    # compute optical flows and sample trajectories
    trajectories = sample_trajectories(os.path.join(args.output_path, "source_video.mp4"), device)
    torch.cuda.empty_cache()

    for k in trajectories.keys():
        trajectories[k] = trajectories[k].to(device)
    sample = pipe(args.prompt, video_length=args.video_length, frames=real_frames,
                num_inference_steps=args.sample_steps, generator=generator, guidance_scale=args.guidance_scale,
                negative_prompt=args.neg_prompt, width=args.width, height=args.height,
                trajs=trajectories, output_dir="tmp/", inject_step=args.inject_step, old_qk=args.old_qk).videos
    temp_video_name = args.prompt+"_"+args.neg_prompt+"_"+str(args.guidance_scale)
    save_videos_grid(sample, f"{args.output_path}/{temp_video_name}.mp4", fps=args.fps)
