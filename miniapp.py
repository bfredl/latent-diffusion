import os
from types import SimpleNamespace
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
import numpy as np
import torch
from PIL import Image
import time

from torchvision.utils import make_grid
from einops import rearrange

model_path_e = hf_hub_download(repo_id="multimodalart/compvis-latent-diffusion-text2img-large", filename="txt2img-f8-large.ckpt")

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def load_model_from_config(config, ckpt, verbose=False, is_cuda=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=("cuda" if is_cuda else None))
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model = model.half()
    if is_cuda:
        model = model.cuda()
    model.eval()
    return model

config = OmegaConf.load("latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml")
model = load_model_from_config(config,model_path_e, is_cuda=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")
model = model.to(device)

def run(prompt, steps, width, height, images, scale):
    opt = SimpleNamespace(
        prompt = prompt,
        outdir='outputs',
        ddim_steps = int(steps),
        ddim_eta = 0,
        n_iter = 1,
        W=int(width),
        H=int(height),
        n_samples=int(images),
        scale=scale,
        plms=True
    )

    if opt.plms:
        opt.ddim_eta = 0
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    all_samples_images=list()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with model.ema_scope():
                uc = None
                if opt.scale > 0:
                    uc = model.get_learned_conditioning(opt.n_samples * [""])
                for n in range(opt.n_iter):
                    c = model.get_learned_conditioning(opt.n_samples * [prompt])
                    shape = [4, opt.H//8, opt.W//8]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image_vector = Image.fromarray(x_sample.astype(np.uint8))
                        all_samples_images.append(image_vector)
                        #Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                        base_count += 1
                    all_samples.append(x_samples_ddim)

    
    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=2)
    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}-{int(time.time())}.png'))
    return(Image.fromarray(grid.astype(np.uint8)),all_samples_images,None)

if False:
    run('A Swedish chef going "borka borka"', 50, 256, 256, 4, 6.0)
