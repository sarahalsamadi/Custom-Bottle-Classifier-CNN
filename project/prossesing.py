import argparse
import hashlib
import random
from pathlib import Path
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def sha1_file(p: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def safe_open_rgb(p: Path):
    """Try opening an image safely. Returns PIL Image RGB or None if corrupted/unreadable."""
    try:
        im = Image.open(p)
        im = ImageOps.exif_transpose(im)  # Fix rotation based on phone EXIF
        im = im.convert("RGB")
        return im
    except Exception:
        return None

def center_square_crop(im: Image.Image) -> Image.Image:
    """Center-crop the image to a square (min(width, height))."""
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return im.crop((left, top, left + side, top + side))

def process_image(src: Path, dst: Path, size: int, quality: int) -> bool:
    """Open -> EXIF transpose -> RGB -> center crop -> resize -> save as JPEG."""
    im = safe_open_rgb(src)
    if im is None:
        return False

    im = center_square_crop(im)
    im = im.resize((size, size), Image.Resampling.LANCZOS)

    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, format="JPEG", quality=quality, optimize=True)
    return True

def split_train_val(items: list[Path], train_ratio: float):
    """
    Split list into train/val only:
    train = first N
    val   = remainder
    """
    n = len(items)
    n_train = int(n * train_ratio)
    train_files = items[:n_train]
    val_files = items[n_train:]
    return train_files, val_files

def main():
    ap = argparse.ArgumentParser(
        description="Prepare classification dataset (TRAIN/VAL only): validate + optional dedup + process + split"
    )
    ap.add_argument("--input", default="captured_images",
                    help="Input folder with class subfolders (e.g., captured_images/HAYAWIYA)")
    ap.add_argument("--output", default="dataset",
                    help="Output dataset folder (will create train/ and val/ inside)")
    ap.add_argument("--classes", nargs="+", default=["HAYAWIYA", "SHAMLAN"],
                    help="Class folder names under input")
    ap.add_argument("--img-size", type=int, default=224,
                    help="Output image size (square), e.g., 224")
    ap.add_argument("--jpg-quality", type=int, default=92,
                    help="JPEG quality 1-95 recommended")
    ap.add_argument("--train", type=float, default=0.8,
                    help="Train ratio (val will be 1-train). Example: 0.8 => train 80%%, val 20%%")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for shuffling")
    ap.add_argument("--dedup", action="store_true",
                    help="Enable duplicate removal via SHA1 (exact duplicates)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Do not write files, only show counts/report")
    args = ap.parse_args()

    if not (0.0 < args.train < 1.0):
        raise SystemExit(f"--train must be between 0 and 1 (exclusive). Got: {args.train}")

    in_dir = Path(args.input)
    out_dir = Path(args.output)

    # Safety: avoid overwriting existing output
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"Output folder already exists and is not empty: {out_dir}. "
            "Remove it or choose another --output."
        )

    random.seed(args.seed)

    stats = {}
    global_seen_hashes = set()  # if --dedup enabled

    for cls in args.classes:
        cls_dir = in_dir / cls
        if not cls_dir.exists():
            raise SystemExit(f"Missing class folder: {cls_dir}")

        files = [p for p in cls_dir.rglob("*") if is_image(p)]
        random.shuffle(files)

        good = []
        corrupt = 0
        dup = 0

        for p in files:
            im = safe_open_rgb(p)
            if im is None:
                corrupt += 1
                continue

            if args.dedup:
                h = sha1_file(p)
                if h in global_seen_hashes:
                    dup += 1
                    continue
                global_seen_hashes.add(h)

            good.append(p)

        # split train/val only
        train_files, val_files = split_train_val(good, args.train)

        stats[cls] = {
            "found": len(files),
            "good": len(good),
            "corrupt_skipped": corrupt,
            "duplicates_skipped": dup,
            "train": len(train_files),
            "val": len(val_files),
        }

        if args.dry_run:
            continue

        def export(split_name: str, split_files: list[Path]):
            for i, src in enumerate(split_files, start=1):
                name = f"{cls}_{split_name}_{i:05d}.jpg"
                dst = out_dir / split_name / cls / name
                process_image(src, dst, args.img_size, args.jpg_quality)

        export("train", train_files)
        export("val", val_files)

    # report
    print("\n=== Dataset preparation report (TRAIN/VAL only) ===")
    print(f"Input : {in_dir.resolve()}")
    print(f"Output: {out_dir.resolve()}")
    print(f"Classes: {args.classes}")
    print(f"Processing: center-square-crop -> resize {args.img_size}x{args.img_size} -> save JPEG (quality={args.jpg_quality})")
    print(f"Split: train={args.train:.2f}, val={1.0-args.train:.2f}, seed={args.seed}, dedup={args.dedup}")

    for cls, s in stats.items():
        print(f"\n[{cls}]")
        for k, v in s.items():
            print(f"  {k}: {v}")

    if args.dry_run:
        print("\n(dry-run) No files were written.")

if __name__ == "__main__":
    main()
