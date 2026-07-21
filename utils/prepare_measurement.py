import argparse
from pathlib import Path

import torch
from munch import munchify
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from util import get_img_list, set_seed
from functions.degradation import get_degradation


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_path",
        type=Path,
        required=True,
        help="Directory containing prepared images.",
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Degradation type (e.g. sr_avgpool, gaussian_blur).",
    )

    parser.add_argument(
        "--deg_scale",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=768,
    )

    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    
    parser.add_argument(
        "--save_preview",
        action="store_true",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    dataset_root = args.img_path.parent

    if args.task.startswith("sr_"):
        measurement_name = f"{args.task}_x{args.deg_scale}"
    else:
        measurement_name = args.task

    measurement_dir = (
        dataset_root
        / "measurement"
        / measurement_name
    )
    preview_dir = (
        dataset_root
        /"measurement_preview"
        / measurement_name
    )

    measurement_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    preview_dir.mkdir(
        parents = True,
        exist_ok = True
    )

    deg_config = munchify(
        {
            "channels": 3,
            "image_size": args.img_size,
            "deg_scale": args.deg_scale,
        }
    )

    operator = get_degradation(
        args.task,
        deg_config,
        torch.device(args.device),
    )

    tf = transforms.ToTensor()

    image_list = list(get_img_list(args.img_path))

    pbar = tqdm(image_list, desc="Generating measurements")

    for idx, path in enumerate(pbar):

        img = tf(Image.open(path).convert("RGB"))
        img = img.unsqueeze(0).to(args.device)
        img = img * 2 - 1

        y = operator.A(img)
        y = y + args.noise_std * torch.randn_like(y)

        torch.save(
            {
                "measurement": y.cpu(),
                "task": args.task,
                "deg_scale": args.deg_scale,
                "noise_std": args.noise_std,
                "seed": args.seed,
                "source": path.name,
            },
            measurement_dir / f"{idx:05d}.pt",
        )
        if args.save_preview:
            save_image(
                operator.At(y).reshape(img.shape),
                preview_dir / f"{idx:05d}.png",
                normalize = True,
            )

    print(f"Saved {len(image_list)} measurements to")
    print(measurement_dir)


if __name__ == "__main__":
    main()