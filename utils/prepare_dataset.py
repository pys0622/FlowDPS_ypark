import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def get_images(root: Path):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def file_sha256(path: Path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def select_random(files, count, seed):
    if len(files) < count:
        raise ValueError(
            f"Requested {count} images, but found only {len(files)}."
        )

    rng = random.Random(seed)
    selected = rng.sample(files, count)

    # 랜덤 선택 결과는 고정하되 처리 순서는 항상 일정하게 유지
    return sorted(selected)


def select_first(files, count):
    if len(files) < count:
        raise ValueError(
            f"Requested {count} images, but found only {len(files)}."
        )

    return files[:count]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=768)

    parser.add_argument(
        "--selection",
        choices=["first", "random", "balanced_afhq"],
        default="random",
    )
    parser.add_argument(
        "--ffhq_validation_only",
        action="store_true",
    )  

    parser.add_argument(
        "--afhq_catndog_only",
        action="store_true",
        help="Use only images under AFHQ cat and dog directories.",
    )

    parser.add_argument(
        "--resize_mode",
        choices=["flowdps", "stretch"],
        default="flowdps",
    )

    args = parser.parse_args()

    all_files = get_images(args.input)

    if not all_files:
        raise RuntimeError(f"No images found under {args.input}")

    if args.ffhq_validation_only:
        filtered = []

        for path in all_files:
            try:
                image_id = int(path.stem)
            except ValueError:
                continue

            if 60000 <= image_id <= 69999:
                filtered.append(path)

        all_files = filtered
    
    if args.afhq_catndog_only:
        filtered = []

        for path in all_files:
            try:
                class_name = path.relative_to(args.input).parts[0]
            except ValueError:
                continue
            
            if class_name in {"cat", "dog"}:
                filtered.append(path)

        all_files = filtered

    if args.selection == "random":
        selected = select_random(
            all_files,
            args.num_samples,
            args.seed,
        )
    elif args.selection == "first":
        selected = select_first(
            all_files,
            args.num_samples,
        )
    else:
        raise ValueError(f"Unknown selection mode: {args.selection}")

    if args.resize_mode == "flowdps":
        transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
        ])
    else:
        transform = transforms.Resize(
            (args.img_size, args.img_size)
        )

    output_image_dir = args.output / "images"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []

    for index, source_path in enumerate(tqdm(selected, desc="Preparing")):
        output_name = f"{index:05d}.png"
        output_path = output_image_dir / output_name

        with Image.open(source_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size

            processed = transform(image)
            processed.save(output_path, format="PNG")

        manifest_entries.append({
            "index": index,
            "filename": output_name,
            "source_path": str(source_path.resolve()),
            "source_relative_path": str(
                source_path.relative_to(args.input)
            ),
            "source_sha256": file_sha256(source_path),
            "original_width": original_width,
            "original_height": original_height,
            "output_width": processed.width,
            "output_height": processed.height,
        })

    manifest = {
        "input_root": str(args.input.resolve()),
        "output_root": str(args.output.resolve()),
        "num_samples": len(selected),
        "seed": args.seed,
        "selection": args.selection,
        "resize_mode": args.resize_mode,
        "img_size": args.img_size,
        "samples": manifest_entries,
    }

    args.output.mkdir(parents=True, exist_ok=True)

    with (args.output / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2)

    print(f"Prepared {len(selected)} images.")
    print(f"Images:   {output_image_dir}")
    print(f"Manifest: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()