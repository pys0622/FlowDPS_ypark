import argparse
import json
from pathlib import Path
from typing import Any

import sys
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


AFHQ_PROMPTS = {
    "cat": "a photo of a closed face of a cat",
    "dog": "a photo of a closed face of a dog",
}

FFHQ_PROMPT = "a photo of a closed face"


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    samples = manifest.get("samples")

    if not isinstance(samples, list):
        raise ValueError(
            f"Manifest must contain a list under 'samples': {path}"
        )

    if not samples:
        raise ValueError(f"Manifest contains no samples: {path}")

    return manifest


def get_relative_path(sample: dict[str, Any]) -> Path:
    relative_path = sample.get("source_relative_path")

    if relative_path:
        return Path(relative_path)

    source_path = sample.get("source_path")

    if source_path:
        return Path(source_path)

    raise ValueError(
        "Manifest sample must contain either "
        "'source_relative_path' or 'source_path'."
    )


def prepare_afhq(manifest: dict[str, Any]) -> list[str]:
    prompts = []

    for index, sample in enumerate(manifest["samples"]):
        relative_path = get_relative_path(sample)

        path_parts = {
            part.lower()
            for part in relative_path.parts
        }

        matched_classes = path_parts.intersection(AFHQ_PROMPTS)

        if len(matched_classes) != 1:
            raise ValueError(
                f"Could not determine AFHQ class for sample {index}: "
                f"{relative_path}. Expected exactly one of "
                f"{sorted(AFHQ_PROMPTS)} in the path."
            )

        class_name = matched_classes.pop()
        prompts.append(AFHQ_PROMPTS[class_name])

    return prompts


def prepare_ffhq(manifest: dict[str, Any]) -> list[str]:
    return [FFHQ_PROMPT] * len(manifest["samples"])



#Methods copied from SeeSR project
def load_tag_model(
    seesr_root: Path,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load the RAM model with SeeSR's DAPE condition weights.
    """

    seesr_root = seesr_root.resolve()

    if not seesr_root.is_dir():
        raise FileNotFoundError(
            f"SeeSR repository does not exist: {seesr_root}"
        )

    ram_path = (
        seesr_root
        / "preset"
        / "models"
        / "ram_swin_large_14m.pth"
    )

    dape_path = (
        seesr_root
        / "preset"
        / "models"
        / "DAPE.pth"
    )

    if not ram_path.is_file():
        raise FileNotFoundError(
            f"RAM checkpoint does not exist: {ram_path}"
        )

    if not dape_path.is_file():
        raise FileNotFoundError(
            f"DAPE checkpoint does not exist: {dape_path}"
        )

    seesr_root_str = str(seesr_root)

    if seesr_root_str not in sys.path:
        sys.path.insert(0, seesr_root_str)

    from ram.models.ram_lora import ram

    model = ram(
        pretrained=str(ram_path),
        pretrained_condition=str(dape_path),
        image_size=384,
        vit="swin_l",
    )

    model.eval()
    model.to(device)

    return model

def get_validation_prompt(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
) -> str:
    """
    Extract a DAPE tag string from a measurement preview image.

    This follows SeeSR's test_seesr.py preprocessing:
        PIL image
        -> ToTensor
        -> batch dimension
        -> Resize(384, 384)
        -> ImageNet normalization
        -> inference_ram

    Only the tag string is returned. No quality prefix is added here.
    """
    from ram import inference_ram as inference

    tensor_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    ram_transform = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image_tensor = tensor_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor = ram_transform(image_tensor)

    with torch.inference_mode():
        result = inference(image_tensor, model)

    if not isinstance(result, (list, tuple)):
        raise TypeError(
            "Unexpected RAM inference result type: "
            f"{type(result).__name__}"
        )

    if not result:
        raise RuntimeError(
            "RAM inference returned an empty result."
        )

    prompt = str(result[0]).strip()

    if not prompt:
        return ""
        # raise RuntimeError(
        #     "RAM inference returned an empty prompt."
        # )

    return prompt


def prepare_div2k(
    manifest: dict[str, Any],
    measurement_preview_dir: Path,
    seesr_root: Path,
) -> list[str]:
    """
    Generate one DAPE prompt per DIV2K sample.

    Preview images are matched to manifest samples by the sample's
    'filename' field.
    """
    measurement_preview_dir = measurement_preview_dir.resolve()

    if not measurement_preview_dir.is_dir():
        raise FileNotFoundError(
            "Measurement preview directory does not exist: "
            f"{measurement_preview_dir}"
        )

    samples = manifest["samples"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"DAPE device: {device}")
    print(f"SeeSR root: {seesr_root.resolve()}")
    print(
        "Measurement preview directory: "
        f"{measurement_preview_dir}"
    )

    model = load_tag_model(
        seesr_root=seesr_root,
        device=device,
    )

    prompts: list[str] = []

    for index, sample in enumerate(
        tqdm(
            samples,
            desc="Extracting DAPE prompts",
        )
    ):
        filename = sample.get("filename")

        if not filename:
            raise ValueError(
                f"Manifest sample {index} does not contain 'filename'."
            )

        image_path = measurement_preview_dir / filename

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Measurement preview does not exist for "
                f"sample {index}: {image_path}"
            )

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")

            prompt = get_validation_prompt(
                image=rgb_image,
                model=model,
                device=device,
            )


        if prompt == "":
            print(f"[Warning] Empty DAPE prompt: {image_path}")
            prompt = "high quality"

        prompts.append(prompt)

    return prompts


def write_prompts(prompts: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for prompt in prompts:
            file.write(prompt + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate FlowDPS prompt files from a prepared dataset manifest."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str.upper,
        choices=["AFHQ", "FFHQ", "DIV2K"],
        required=True,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    
    parser.add_argument(
    "--measurement_preview_dir",
    type=Path,
    default=None,
    help="Measurement preview directory generated by prepare_measurement.py",
    )

    parser.add_argument(
        "--seesr_root",
        type=Path,
        default=None,
        help="Root directory of the SeeSR repository.",
    )
    

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )


    args = parser.parse_args()
    manifest = load_manifest(args.manifest)

    if args.dataset == "AFHQ":
        prompts = prepare_afhq(manifest)

    elif args.dataset == "FFHQ":
        prompts = prepare_ffhq(manifest)

    elif args.dataset == "DIV2K":
        if args.measurement_preview_dir is None:
            raise ValueError(
                "--measurement_preview_dir is required for DIV2K."
            )

        if args.seesr_root is None:
            raise ValueError(
                "--seesr_root is required for DIV2K."
            )

        prompts = prepare_div2k(
            manifest,
            args.measurement_preview_dir,
            args.seesr_root,
        )

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if len(prompts) != len(manifest["samples"]):
        raise RuntimeError(
            "Internal error: prompt count does not match manifest count."
        )

    if any(not prompt.strip() for prompt in prompts):
        raise RuntimeError("Generated prompt list contains an empty prompt.")

    write_prompts(prompts, args.output)

    print(f"Dataset:  {args.dataset}")
    print(f"Manifest: {args.manifest}")
    print(f"Output:   {args.output}")
    print(f"Prompts:  {len(prompts)}")

    for index, prompt in enumerate(prompts[:5]):
        filename = manifest["samples"][index].get(
            "filename",
            f"sample-{index}",
        )
        print(f"[{index:05d}] {filename}: {prompt}")


if __name__ == "__main__":
    main()