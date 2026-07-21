from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from pathlib import Path


SOURCE_FONTS = (b"Helvetica Neue", b"Trebuchet MS", b"Calibri")
TARGET_FONT = b"Microsoft YaHei"


def replace_fonts_in_pptx(pptx_path: Path) -> int:
    replacements = 0
    with tempfile.NamedTemporaryFile(
        prefix=f"{pptx_path.stem}_fontfix_",
        suffix=".pptx",
        dir=pptx_path.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)

    try:
        with zipfile.ZipFile(pptx_path, "r") as source, zipfile.ZipFile(
            temp_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as target:
            for item in source.infolist():
                payload = source.read(item.filename)
                if item.filename.endswith((".xml", ".rels")):
                    for source_font in SOURCE_FONTS:
                        count = payload.count(source_font)
                        if count:
                            payload = payload.replace(source_font, TARGET_FONT)
                            replacements += count
                target.writestr(item, payload)
        os.replace(temp_path, pptx_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return replacements


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace non-CJK presentation fonts with Microsoft YaHei."
    )
    parser.add_argument("pptx", type=Path)
    args = parser.parse_args()
    pptx_path = args.pptx.resolve()
    if not pptx_path.is_file():
        raise FileNotFoundError(pptx_path)
    count = replace_fonts_in_pptx(pptx_path)
    print(f"font_replacements={count} pptx={pptx_path}")


if __name__ == "__main__":
    main()
