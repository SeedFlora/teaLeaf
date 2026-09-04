"""Generate launcher icons at every density from the supplied source artwork.

The source is a 1254x1254 PNG with no alpha channel, so its rounded corners are baked in
as opaque pixels rather than transparency. Shipping that directly would render as a square
tile with dark corners under any launcher mask.

Two outputs solve that. The legacy square/round mipmaps get the artwork with its corners
made transparent, and the adaptive icon gets a separate foreground and background layer:
Android composites those under whatever mask the launcher applies, so the icon takes the
device's own shape instead of fighting it.

The adaptive foreground is inset because Android reserves the outer ~18% of an adaptive
layer for parallax and masking; artwork drawn to the edge gets clipped on many launchers.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

# Launcher icon sizes per density bucket.
DENSITIES = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}
# Adaptive layers are 108dp; only the central 72dp is guaranteed visible.
ADAPTIVE = {"mdpi": 108, "hdpi": 162, "xhdpi": 216, "xxhdpi": 324, "xxxhdpi": 432}
SAFE_FRACTION = 0.72


def corner_alpha(img: Image.Image, radius_frac: float = 0.22) -> Image.Image:
    """Replace the baked-in rounded corners with real transparency."""
    img = img.convert("RGBA")
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [(0, 0), (w - 1, h - 1)], radius=int(min(w, h) * radius_frac), fill=255
    )
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def circular(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse([(0, 0), (w - 1, h - 1)], fill=255)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def dominant_background(img: Image.Image) -> tuple[int, int, int]:
    """Sample the artwork's interior for a background colour that will not clash."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    px = [rgb.getpixel((int(w * fx), int(h * fy)))
          for fx in (0.5, 0.35, 0.65) for fy in (0.12, 0.5, 0.88)]
    r = sum(p[0] for p in px) // len(px)
    g = sum(p[1] for p in px) // len(px)
    b = sum(p[2] for p in px) // len(px)
    # Lighten toward white so the leaf artwork stays legible on top.
    return (min(255, (r + 255 * 3) // 4), min(255, (g + 255 * 3) // 4), min(255, (b + 255 * 3) // 4))


def main() -> int:
    src_path = Path(sys.argv[1])
    res = Path(sys.argv[2])

    src = Image.open(src_path)
    print(f"source: {src.size[0]}x{src.size[1]} mode={src.mode}")

    squared = corner_alpha(src)
    bg = dominant_background(src)
    print(f"adaptive background colour: rgb{bg}")

    for density, size in DENSITIES.items():
        d = res / f"mipmap-{density}"
        d.mkdir(parents=True, exist_ok=True)
        squared.resize((size, size), Image.LANCZOS).save(d / "ic_launcher.png")
        circular(src.convert("RGBA")).resize((size, size), Image.LANCZOS).save(
            d / "ic_launcher_round.png")

    for density, size in ADAPTIVE.items():
        d = res / f"mipmap-{density}"
        d.mkdir(parents=True, exist_ok=True)
        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        inner = int(size * SAFE_FRACTION)
        art = src.convert("RGBA").resize((inner, inner), Image.LANCZOS)
        off = (size - inner) // 2
        layer.paste(art, (off, off), art)
        layer.save(d / "ic_launcher_foreground.png")

    values = res / "values"
    values.mkdir(parents=True, exist_ok=True)
    (values / "ic_launcher_background.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<resources>\n'
        f'    <color name="ic_launcher_background">#{bg[0]:02X}{bg[1]:02X}{bg[2]:02X}</color>\n'
        "</resources>\n", encoding="utf-8")

    anydpi = res / "mipmap-anydpi-v26"
    anydpi.mkdir(parents=True, exist_ok=True)
    adaptive_xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">\n'
        '    <background android:drawable="@color/ic_launcher_background" />\n'
        '    <foreground android:drawable="@mipmap/ic_launcher_foreground" />\n'
        '    <monochrome android:drawable="@mipmap/ic_launcher_foreground" />\n'
        "</adaptive-icon>\n"
    )
    (anydpi / "ic_launcher.xml").write_text(adaptive_xml, encoding="utf-8")
    (anydpi / "ic_launcher_round.xml").write_text(adaptive_xml, encoding="utf-8")

    total = sum(1 for _ in res.rglob("ic_launcher*"))
    print(f"generated {total} launcher icon resources under {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
