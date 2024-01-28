import os
import imageio
import numpy as np
from typing import Union
import decord
decord.bridge.set_bridge('torch')
import torch
import torchvision
import PIL
from typing import List
from tqdm import tqdm
from einops import rearrange
import torchvision.transforms.functional as F
import random

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def save_videos_grid_pil(videos: List[PIL.Image.Image], path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def read_video(video_path, video_length, width=512, height=512, frame_rate=None):
    vr = decord.VideoReader(video_path, width=width, height=height)
    if frame_rate is None:
        frame_rate = max(1, len(vr) // video_length)
    sample_index = list(range(0, len(vr), frame_rate))[:video_length]
    video = vr.get_batch(sample_index)
    video = rearrange(video, "f h w c -> f c h w")
    video = (video / 127.5 - 1.0)
    return video


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


"""optical flow and trajectories sampling"""
def preprocess(img1_batch, img2_batch, transforms):
    img1_batch = F.resize(img1_batch, size=[512, 512], antialias=False)
    img2_batch = F.resize(img2_batch, size=[512, 512], antialias=False)
    return transforms(img1_batch, img2_batch)

def keys_with_same_value(dictionary):
    result = {}
    for key, value in dictionary.items():
        if value not in result:
            result[value] = [key]
        else:
            result[value].append(key)

    conflict_points = {}
    for k in result.keys():
        if len(result[k]) > 1:
            conflict_points[k] = result[k]
    return conflict_points

def find_duplicates(input_list):
    seen = set()
    duplicates = set()

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)

def neighbors_index(point, window_size, H, W):
    """return the spatial neighbor indices"""
    t, x, y = point
    neighbors = []
    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            if i == 0 and j == 0:
                continue
            if x + i < 0 or x + i >= H or y + j < 0 or y + j >= W:
                continue
            neighbors.append((t, x + i, y + j))
    return neighbors


@torch.no_grad()
def sample_trajectories(video_path, device):
    from torchvision.models.optical_flow import Raft_Large_Weights
    from torchvision.models.optical_flow import raft_large

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")

    clips = list(range(len(frames)))

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    finished_trajectories = []

    current_frames, next_frames = preprocess(frames[clips[:-1]], frames[clips[1:]], transforms)
    list_of_flows = model(current_frames.to(device), next_frames.to(device))
    predicted_flows = list_of_flows[-1]

    predicted_flows = predicted_flows/512

    resolutions = [64, 32, 16, 8]
    res = {}
    window_sizes = {64: 2,
                    32: 1,
                    16: 1,
                    8: 1}

    for resolution in resolutions:
        print("="*30)
        trajectories = {}
        predicted_flow_resolu = torch.round(resolution*torch.nn.functional.interpolate(predicted_flows, scale_factor=(resolution/512, resolution/512)))

        T = predicted_flow_resolu.shape[0]+1
        H = predicted_flow_resolu.shape[2]
        W = predicted_flow_resolu.shape[3]

        is_activated = torch.zeros([T, H, W], dtype=torch.bool)

        for t in range(T-1):
            flow = predicted_flow_resolu[t]
            for h in range(H):
                for w in range(W):

                    if not is_activated[t, h, w]:
                        is_activated[t, h, w] = True
                        # this point has not been traversed, start new trajectory
                        x = h + int(flow[1, h, w])
                        y = w + int(flow[0, h, w])
                        if x >= 0 and x < H and y >= 0 and y < W:
                            # trajectories.append([(t, h, w), (t+1, x, y)])
                            trajectories[(t, h, w)]= (t+1, x, y)

        conflict_points = keys_with_same_value(trajectories)
        for k in conflict_points:
            index_to_pop = random.randint(0, len(conflict_points[k]) - 1)
            conflict_points[k].pop(index_to_pop)
            for point in conflict_points[k]:
                if point[0] != T-1:
                    trajectories[point]= (-1, -1, -1) # stupid padding with (-1, -1, -1)

        active_traj = []
        all_traj = []
        for t in range(T):
            pixel_set = {(t, x//H, x%H):0 for x in range(H*W)}
            new_active_traj = []
            for traj in active_traj:
                if traj[-1] in trajectories:
                    v = trajectories[traj[-1]]
                    new_active_traj.append(traj + [v])
                    pixel_set[v] = 1
                else:
                    all_traj.append(traj)
            active_traj = new_active_traj
            active_traj+=[[pixel] for pixel in pixel_set if pixel_set[pixel] == 0]
        all_traj += active_traj

        useful_traj = [i for i in all_traj if len(i)>1]
        for idx in range(len(useful_traj)):
            if useful_traj[idx][-1] == (-1, -1, -1):
                useful_traj[idx] = useful_traj[idx][:-1]
        print("how many points in all trajectories for resolution{}?".format(resolution), sum([len(i) for i in useful_traj]))
        print("how many points in the video for resolution{}?".format(resolution), T*H*W)

        # validate if there are no duplicates in the trajectories
        trajs = []
        for traj in useful_traj:
            trajs = trajs + traj
        assert len(find_duplicates(trajs)) == 0, "There should not be duplicates in the useful trajectories."

        # check if non-appearing points + appearing points = all the points in the video
        all_points = set([(t, x, y) for t in range(T) for x in range(H) for y in range(W)])
        left_points = all_points- set(trajs)
        print("How many points not in the trajectories for resolution{}?".format(resolution), len(left_points))
        for p in list(left_points):
            useful_traj.append([p])
        print("how many points in all trajectories for resolution{} after pending?".format(resolution), sum([len(i) for i in useful_traj]))


        longest_length = max([len(i) for i in useful_traj])
        sequence_length = (window_sizes[resolution]*2+1)**2 + longest_length - 1

        seqs = []
        masks = []

        # create a dictionary to facilitate checking the trajectories to which each point belongs.
        point_to_traj = {}
        for traj in useful_traj:
            for p in traj:
                point_to_traj[p] = traj

        for t in range(T):
            for x in range(H):
                for y in range(W):
                    neighbours = neighbors_index((t,x,y), window_sizes[resolution], H, W)
                    sequence = [(t,x,y)]+neighbours + [(0,0,0) for i in range((window_sizes[resolution]*2+1)**2-1-len(neighbours))]
                    sequence_mask = torch.zeros(sequence_length, dtype=torch.bool)
                    sequence_mask[:len(neighbours)+1] = True

                    traj = point_to_traj[(t,x,y)].copy()
                    traj.remove((t,x,y))
                    sequence = sequence + traj + [(0,0,0) for k in range(longest_length-1-len(traj))]
                    sequence_mask[(window_sizes[resolution]*2+1)**2: (window_sizes[resolution]*2+1)**2 + len(traj)] = True

                    seqs.append(sequence)
                    masks.append(sequence_mask)

        seqs = torch.tensor(seqs)
        masks = torch.stack(masks)
        res["traj{}".format(resolution)] = seqs
        res["mask{}".format(resolution)] = masks
    return res

