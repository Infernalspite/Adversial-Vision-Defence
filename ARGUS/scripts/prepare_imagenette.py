from pathlib import Path
import argparse
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "sources"
    )
    parser.add_argument("--total", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()

    random.seed(args.seed)

    images = sorted(
        p for p in args.input.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    if len(images) < args.total:
        raise RuntimeError(
            f"Found only {len(images)} images, "
            f"but requested {args.total}."
        )

    random.shuffle(images)
    images = images[:args.total]

    baseline_count = args.total // 3
    validation_count = args.total // 3
    test_count = args.total - baseline_count - validation_count

    splits = {
        "baseline": images[:baseline_count],
        "validation": images[
            baseline_count:baseline_count + validation_count
        ],
        "test": images[
            baseline_count + validation_count:
        ],
    }

    for split_name, split_images in splits.items():
        output_dir = args.output / split_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for index, source in enumerate(split_images):
            destination = (
                output_dir
                / f"{index:04d}_{source.stem}{source.suffix.lower()}"
            )
            shutil.copy2(source, destination)

        print(f"{split_name}: {len(split_images)} images")

    print()
    print("Dataset preparation complete.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()