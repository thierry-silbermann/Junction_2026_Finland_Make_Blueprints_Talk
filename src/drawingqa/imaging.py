from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from .data import IMAGE_EXTENSIONS
from .schemas import PipelineConfig


@dataclass(frozen=True)
class Page:
    number: int
    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class View:
    id: str
    page: int
    path: Path
    # Crop position normalized to the full rendered page.
    region: tuple[float, float, float, float]

    def to_page(self, bbox):
        x1, y1, x2, y2 = self.region
        a, b, c, d = bbox
        return (x1 + a * (x2-x1), y1 + b * (y2-y1),
                x1 + c * (x2-x1), y1 + d * (y2-y1))


def save_image(image: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".png")
    os.close(descriptor)
    try:
        image.save(name, "PNG")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def render_pages(source: Path, directory: Path, fingerprint: str,
                 settings: PipelineConfig, selected_page: int | None = None) -> list[Page]:
    """Render complete selected pages; never silently truncate a document."""
    directory.mkdir(parents=True, exist_ok=True)
    pages = []

    def write(number, image):
        if image.width * image.height > settings.max_source_pixels:
            raise ValueError("Page exceeds max_source_pixels; reduce PDF DPI or use a smaller source")
        image = rgb(image)
        target = directory / f"{fingerprint[:20]}-p{number}-dpi{settings.pdf_dpi}.png"
        if not target.exists():
            save_image(image, target)
        pages.append(Page(number, target, image.width, image.height))

    def selection(total):
        if total < 1:
            raise ValueError("Document has no pages")
        if selected_page is not None:
            if selected_page < 1 or selected_page > total:
                raise ValueError(f"Requested page {selected_page}; document has {total}")
            return [selected_page]
        if total > settings.max_pages:
            raise ValueError(f"Document has {total} pages; select page or raise max_pages={settings.max_pages}")
        return list(range(1, total + 1))

    if source.suffix.lower() == ".pdf":
        import fitz
        with fitz.open(source) as document:
            if document.needs_pass:
                raise ValueError("Encrypted PDF requires an unlocked copy")
            for number in selection(document.page_count):
                page = document[number-1]
                factor = settings.pdf_dpi / 72
                if math.ceil(page.rect.width*factor)*math.ceil(page.rect.height*factor) > settings.max_source_pixels:
                    raise ValueError("PDF page exceeds pixel limit before rasterization; lower pdf_dpi")
                pixels = page.get_pixmap(matrix=fitz.Matrix(factor, factor), colorspace=fitz.csRGB, alpha=False)
                write(number, Image.frombytes("RGB", [pixels.width, pixels.height], pixels.samples))
    elif source.suffix.lower() in IMAGE_EXTENSIONS:
        with Image.open(source) as image:
            for number in selection(getattr(image, "n_frames", 1)):
                image.seek(number - 1)
                if image.width * image.height > settings.max_source_pixels:
                    raise ValueError("Image exceeds max_source_pixels")
                write(number, image.copy())
    else:
        raise ValueError(f"Unsupported raster format: {source.suffix}")
    return pages


def _positions(length, size, overlap):
    if length <= size:
        return [0]
    stride = max(1, int(size*(1-overlap)))
    positions = list(range(0, length-size+1, stride))
    if positions[-1] != length-size:
        positions.append(length-size)
    return positions


def make_views(pages: list[Page], directory: Path, settings: PipelineConfig,
               category: str) -> list[View]:
    directory.mkdir(parents=True, exist_ok=True)
    planned = []
    for page in pages:
        planned.append((page, "overview", (0, 0, page.width, page.height)))
        # Room/floor counts require whole-panel context. Tiling could split one room.
        if settings.views == "tiles" and category in {"doors", "windows", "toilets", "sinks", "showers_bathtubs"}:
            if page.width > settings.tile_size or page.height > settings.tile_size:
                for y in _positions(page.height, settings.tile_size, settings.tile_overlap):
                    for x in _positions(page.width, settings.tile_size, settings.tile_overlap):
                        planned.append((page, f"x{x}-y{y}", (x, y, min(x+settings.tile_size,page.width), min(y+settings.tile_size,page.height))))
    if len(planned) > settings.max_views:
        raise ValueError(f"Need {len(planned)} views; max_views={settings.max_views}. Select page or adjust tiling; no partial count was sent")
    result = []
    for page, tag, rect in planned:
        key = f"{page.path.stem}-{tag}-{settings.max_image_side}-{settings.tile_size}"
        target = directory / (key + ".png")
        if not target.exists():
            with Image.open(page.path) as image:
                crop = image.crop(rect)
                crop.thumbnail((settings.max_image_side, settings.max_image_side), Image.Resampling.LANCZOS)
                save_image(crop, target)
        region = (rect[0]/page.width, rect[1]/page.height, rect[2]/page.width, rect[3]/page.height)
        result.append(View(f"p{page.number}-{tag}", page.number, target, region))
    return result
