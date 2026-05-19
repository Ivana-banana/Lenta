from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def extract_media(pptx_path: str | Path, out_dir: str | Path) -> int:
    pptx_path = Path(pptx_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(pptx_path) as zf:
        for name in zf.namelist():
            if not name.startswith("ppt/media/"):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg"}:
                continue
            out = out_dir / f"{pptx_path.stem}_{Path(name).stem}{suffix}"
            out.write_bytes(zf.read(name))
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pptx", default="../ГМ для ТК.pptx")
    parser.add_argument("--out", default="configs/template_images")
    args = parser.parse_args()
    count = extract_media(args.pptx, args.out)
    print(f"Extracted {count} images to {args.out}")


if __name__ == "__main__":
    main()

