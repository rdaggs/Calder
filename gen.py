from PIL import Image, ImageEnhance
import os
import torch
import numpy as np
import argparse
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image
from IPython.display import display, HTML, Image as IPyImage

device = "mps"
print(f"Using device: {device}")    

STACK_DIR = "stack"
ORDER_FILE = "order.txt"
GIFS_DIR = "outputs/dir"
os.makedirs(GIFS_DIR,exist_ok=True)

# GENERATE GIF 
def generate_gif(n, src, tgt, pipe, M, SEED, PROMPT, GUIDANCE, NUM_STEPS, DURATION):
    """
    Generate transition gif from src -> tgt, saved as {n}.gif in GIFS_DIR.

    Parameters
    ----------
    n   : int         — output filename (e.g. 3 -> '3.gif')
    src : PIL.Image   — starting frame (pass last frame of previous gif to chain)
    tgt : PIL.Image   — ending anchor image

    Returns
    -------
    frames : list of PIL.Image   — all frames including src and tgt
    """
    gen    = torch.Generator(device=device).manual_seed(SEED)
    frames = [src]
    prev   = src

    for k in range(1, M + 1):
        lam = k / (M + 1)   # 0 → 1 (excluding endpoints)
        beta = 0.85 * (lam ** 2.5) # used to be 2
        init = Image.blend(prev, tgt, beta)
        # strength = 0.55 + 0.40 * lam   # tops out ~0.95
        strength = 0.35 + 0.25 * lam   # tops out ~0.6


        img = pipe(
            prompt              = PROMPT,
            image               = init,
            strength            = float(strength),
            guidance_scale      = float(GUIDANCE),
            num_inference_steps = int(NUM_STEPS),
            generator           = gen,
        ).images[0]

        img = ImageEnhance.Color(img).enhance(0.95)
        r, g, b = img.split()
        b = b.point(lambda i: i * 1.02)
        r = r.point(lambda i: i * 0.98)
        img = Image.merge("RGB", (r, g, b))

        frames.append(img)
        prev = img

    frames.append(tgt)

    gif_path = os.path.join(GIFS_DIR, f"{n}.gif")
    frames[0].save(
        gif_path,
        save_all      = True,
        append_images = frames[1:],
        duration      = DURATION,
        loop          = 0,
    )

    if device == "mps":
        torch.mps.empty_cache()

    print(f"Saved {gif_path}  ({len(frames)} frames)")
    return frames


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--intermediate_frames", type=int, default=30, help="intermediate steps per image")
    ap.add_argument("--num_steps", type=int, default=8, help="diffusion steps")
    ap.add_argument("--guidance", type=float, default=2.0, help= "adherence to prompt")
    ap.add_argument("--duration", type=int, default=60, help="ms per gif frame")
    ap.add_argument("--prompt", type=str, default="dreamy but structurally coherent texture, consistent lighting, realistic texture", help="prompt")
    args = ap.parse_args()

    M = args.intermediate_frames
    NUM_STEPS = args.num_steps
    GUIDANCE = args.guidance
    DURATION = args.duration
    SEED = 1234
    SIZE = (512,512)
    PROMPT = args.prompt


    #===============================LOAD MODEL===============================#
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype = torch.float16,
        variant     = "fp16",
    ).to(device)
    pipe.safety_checker = None

    print("Model loaded.")

    #===============================LOAD ANCHORS==============================#
    with open(ORDER_FILE) as f:
        order_lines = [l.strip() for l in f if l.strip()]

    image_paths = []
    for line in order_lines:
        folder, filename = line.split(" - ", 1)
        image_paths.append(os.path.join(STACK_DIR, folder, filename))

    anchors = [load_image(p).convert("RGB").resize(SIZE) for p in image_paths]
    print(f"Loaded {len(anchors)} anchor images from order.txt")

    #===============================GENERATE GIF==============================#
    existing = [int(f[:-4]) for f in os.listdir(GIFS_DIR) if f.endswith(".gif") and f[:-4].isdigit()]
    start_i  = max(existing) + 1 if existing else 1

    if start_i > 1:
        prev_gif = Image.open(os.path.join(GIFS_DIR, f"{start_i - 1}.gif"))
        prev_gif.seek(prev_gif.n_frames - 1)
        chain_src = prev_gif.convert("RGB").resize(SIZE)
        print(f"Resuming from {start_i}.gif (last frame of {start_i - 1}.gif loaded)")
    else:
        chain_src = anchors[0]


    for i, tgt in enumerate(anchors[1:] + [anchors[0]], start=1):
        if i < start_i:
            continue
        print(f"Generating {i}.gif ...")
        frames = generate_gif(n=i, src=chain_src, tgt=tgt, pipe=pipe, M=M, SEED=SEED, PROMPT=PROMPT, GUIDANCE=GUIDANCE, NUM_STEPS=NUM_STEPS, DURATION=DURATION)
        # all_frames[i] = frames
        chain_src = frames[-1]

    print("\nAll gifs generated.")







if __name__ == "__main__":
    main()
