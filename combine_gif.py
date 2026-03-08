import argparse, datetime, os
from PIL import Image

def combine_gifs(gifs_dir, output_path, duration):
    """
    Combine all .gif files in gifs_dir into one master gif.
    
    Parameters
    ----------
    gifs_dir    : str — folder containing numbered .gif files
    output_path : str — where to save the combined gif
    duration    : int — ms per frame in the output gif

    Returns
    -------
    output_path : str — path to the saved master gif
    """
    gif_files = sorted(
        [f for f in os.listdir(gifs_dir) if f.endswith(".gif")],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0
    )

    if not gif_files:
        print(f"No gifs found in {gifs_dir!r}")
        return None

    print(f"Combining {len(gif_files)} gifs into {output_path!r} ...")

    all_frames = []

    for i, fname in enumerate(gif_files):
        path = os.path.join(gifs_dir, fname)
        gif  = Image.open(path)

        # Extract all frames from this gif
        frames = []
        try:
            while True:
                frames.append(gif.copy().convert("RGB"))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        # For intermediate gifs, skip first and last frame to avoid duplicates
        # (first frame = last frame of previous gif, last frame = first frame of next gif)
        if i == 0:
            # First gif: keep all frames except the last (it's duplicated in gif 2)
            all_frames.extend(frames[:-1])
        elif i == len(gif_files) - 1:
            # Last gif: skip first frame (duplicated from previous), keep the rest
            all_frames.extend(frames[1:])
        else:
            # Middle gifs: skip both first and last
            all_frames.extend(frames[1:-1])

        print(f"  {fname}: {len(frames)} frames ({len(frames) if i == 0 else len(frames) - 1 if i == len(gif_files) - 1 else len(frames) - 2} kept)")

    # Save combined gif
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_frames[0].save(
        output_path,
        save_all      = True,
        append_images = all_frames[1:],
        duration      = duration,
        loop          = 0,
    )

    print(f"\nSaved master gif with {len(all_frames)} total frames to {output_path!r}")
    return output_path



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gifs_dir",type=str,default='outputs/gifs')
    ap.add_argument("--output_path",type=str,default=f"outputs/master_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.gif")
    ap.add_argument("--duration",type=int,default=60,help="ms per frame")
    args = ap.parse_args()

    gifs_dir, output_path, duration = args.gifs_dir, args.output_path, args.duration

    combine_gifs(gifs_dir=gifs_dir,output_path=output_path,duration=duration)

if __name__ == "__main__":
    main()