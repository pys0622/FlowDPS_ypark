import argparse
import json
from pathlib import Path
from typing import Any


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


def load_div2k_descriptions(path: Path) -> dict[str, str]:
    """
    Load DAPE descriptions from a text file.

    Supported line formats:

        00001.png: a mountain, trees, and a lake
        00001: a mountain, trees, and a lake
        a mountain, trees, and a lake

    Lines containing filenames or indices are matched by image stem.
    Lines without a key are treated as ordered descriptions.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"DIV2K description file does not exist: {path}"
        )

    descriptions: dict[str, str] = {}
    ordered_descriptions: list[str] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line:
                continue

            if ": " in line:
                key, description = line.split(": ", maxsplit=1)
                key = Path(key.strip()).stem
                description = description.strip()

                if not key:
                    raise ValueError(
                        f"Empty key at {path}:{line_number}"
                    )

                if key in descriptions:
                    raise ValueError(
                        f"Duplicate DIV2K description key '{key}' "
                        f"at {path}:{line_number}"
                    )

                descriptions[key] = description
            else:
                ordered_descriptions.append(line)

    if descriptions and ordered_descriptions:
        raise ValueError(
            "DIV2K description file mixes keyed and ordered lines. "
            "Use either 'filename: description' for every line or "
            "one plain description per line."
        )

    if descriptions:
        return descriptions

    return {
        str(index): description
        for index, description in enumerate(ordered_descriptions)
    }


def format_div2k_prompt(description: str) -> str:
    description = description.strip().rstrip(".")

    if not description:
        return "a high quality photo"

    lowered = description.lower()

    if lowered.startswith("a high quality photo"):
        return description

    return f"a high quality photo of {description}"


def prepare_div2k(
    manifest: dict[str, Any],
    description_file: Path,
) -> list[str]:
    descriptions = load_div2k_descriptions(description_file)
    samples = manifest["samples"]

    keyed_by_filename = any(
        Path(sample.get("filename", "")).stem in descriptions
        for sample in samples
    )

    prompts = []

    if keyed_by_filename:
        missing = []

        for index, sample in enumerate(samples):
            filename = sample.get("filename")

            if not filename:
                raise ValueError(
                    f"Sample {index} does not contain 'filename'."
                )

            stem = Path(filename).stem
            description = descriptions.get(stem)

            if description is None:
                missing.append(stem)
                continue

            prompts.append(format_div2k_prompt(description))

        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(
                f"Missing DAPE descriptions for {len(missing)} samples. "
                f"First missing keys: {preview}"
            )

        return prompts

    if len(descriptions) != len(samples):
        raise ValueError(
            "Ordered DIV2K description count does not match manifest: "
            f"{len(descriptions)} descriptions for {len(samples)} samples."
        )

    for index in range(len(samples)):
        prompts.append(
            format_div2k_prompt(descriptions[str(index)])
        )

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
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--description_file",
        type=Path,
        default=None,
        help=(
            "DAPE description file for DIV2K. Required when "
            "--dataset DIV2K is selected."
        ),
    )

    args = parser.parse_args()
    manifest = load_manifest(args.manifest)

    if args.dataset == "AFHQ":
        prompts = prepare_afhq(manifest)

    elif args.dataset == "FFHQ":
        prompts = prepare_ffhq(manifest)

    elif args.dataset == "DIV2K":
        if args.description_file is None:
            raise ValueError(
                "--description_file is required for DIV2K."
            )

        prompts = prepare_div2k(
            manifest,
            args.description_file,
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