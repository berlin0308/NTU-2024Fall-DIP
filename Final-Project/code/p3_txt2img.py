import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from transformers import AutoFeatureExtractor

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def gen_personalized_images(opt, prompt, prompt_output_path, model, sid, pid):
    
    prompt = prompt.replace("<new1>", "*")
    prompt = prompt.replace("<new2>", "*")
    print(f"\n\nProcessed Prompt: {prompt}\n\n")
    
    sampler = DPMSolverSampler(model)

    # os.makedirs(opt.outdir, exist_ok=True)
    # prompt_output_path = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
        
    assert prompt is not None
    data = [batch_size * [prompt]]


    base_count = 0
    grid_count = len(os.listdir(prompt_output_path)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        
                        c = model.get_learned_conditioning(prompts)
                        c = 0.8 * c


                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(prompt_output_path, f"source{sid}_prompt{pid}_{base_count:02}.png"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(prompt_output_path, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{prompt_output_path} \n"
          f" \nEnjoy.")


if __name__ == "__main__":



    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="p3_output"
    )
    parser.add_argument(
        "--skip_grid",
        default=False,
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.1,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5, # 25 images for each turn
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=15,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stable_diffusion/ldm/models/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--embedding_manager_ckpt", 
        type=str, 
        default="", 
        help="Initialize embedding manager from a checkpoint"
    )
    opt = parser.parse_args()


    from datetime import datetime
    print(datetime.now())

    json_file = {
        "0":{
            "src_image": "dog",
            "token_name": "<new1>",
            "prompt": ["A <new1> posing proudly on a hilltop with Mount Fuji in the background.", 
                "A <new1> perched on a park bench with the Colosseum looming behind."],
            "prompt_4_clip_eval": ["A dog posing proudly on a hilltop with Mount Fuji in the background.", 
                "A dog perched on a park bench with the Colosseum looming behind."],
            "baseline": [[79, 34], [74, 35]]
        },
        "1":{
            "src_image": "David Revoy",
            "token_name": "<new2>",
            "prompt": ["The streets of Paris in the style of <new2>.", 
                "Manhattan skyline in the style of <new2>."],
            "prompt_4_clip_eval": ["The streets of Paris in the style of David Revoy.", 
                "Manhattan skyline in the style of David Revoy."],
            "baseline": [[70, 30], [70, 30]]
        }
    }

    # json_file = {
    #     "0":{
    #         "src_image": "iot-device",
    #         "token_name": "*",
    #         "prompt": ["A photo of a * in the greenhouse.", "A photo of a * with plants as the background."
    #                     , "A photo of a quadcopter * .", "A photo of a quadcopter * in a greenhouse."
    #                     , "A photo of a quadcopter in a greenhouse."]
    #     }
    # }

    output_path_root = opt.outdir
    os.makedirs(output_path_root, exist_ok=True)

    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    if opt.embedding_manager_ckpt != "":
        print("Set embeddings")
        config.model.params.personalization_config.params.embedding_manager_ckpt = opt.embedding_manager_ckpt
        config.model.params.personalization_config.params.placeholder_strings = ["*"]
        config.model.params.personalization_config.params.initializer_words = ["copter"]
    else:
        print("Not use the embeddings!\n")

    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)


    for image_source_id in [0,1]:
        output_path = os.path.join(output_path_root, str(image_source_id))
        os.makedirs(output_path, exist_ok=True)

        prompts = json_file[str(image_source_id)]["prompt"]

        for i, prompt in enumerate(prompts):
            prompt_output_path = os.path.join(output_path, str(i))
            os.makedirs(prompt_output_path, exist_ok=True)

            print(f"\nPrompt: {prompt}")
            gen_personalized_images(opt=opt, prompt=prompt, prompt_output_path=prompt_output_path
                                    , model=model, sid=image_source_id, pid=i)
            print(f"\nImage Saved: {prompt_output_path}")


    print(datetime.now())

    """
    python3 p3_txt2img.py --embedding_manager_ckpt p3_merged_\*_#_embeds.pt --config configs/stable-diffusion/v1-inference.yaml --outdir p3_output/output_images_s15 --scale 15


    python3 p3_txt2img.py --embedding_manager_ckpt DIP_data/logs/iot-device2024-11-06T18-02-24_txt-inv-iotdevice/checkpoints/embeddings_gs-6099.pt --config DIP_data/v1-inference.yaml --outdir DIP_data/output_images_s1 --scale 1    
    python3 p3_txt2img.py --embedding_manager_ckpt DIP_data/logs/uav_crop_blur2024-11-07T21-37-26_txt-inv-uav-crop-blur-drone/checkpoints/embeddings_gs-5999.pt --config DIP_data/v1-inference.yaml --outdir DIP_data/output_images_s5 --scale 5    
    """
