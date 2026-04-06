"""Image preprocessing pipeline — pure Python.

Provides:
- Image class backed by list-of-lists of (r, g, b) tuples
- Color-space conversions (RGB ↔ HSV, RGB → grayscale)
- ImageAugmenter with composable transforms
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple, Optional, Callable

Pixel = Tuple[int, int, int]  # (r, g, b)


class Image:
    """Simulated image stored as a 2-D list of (r, g, b) tuples."""

    def __init__(self, data: List[List[Pixel]]):
        self._data = data

    # ---- factory ----------------------------------------------------------
    @staticmethod
    def from_list(data: List[List[Pixel]]) -> "Image":
        return Image(data)

    @staticmethod
    def blank(width: int, height: int, fill: Pixel = (0, 0, 0)) -> "Image":
        row = [fill] * width
        return Image([list(row) for _ in range(height)])

    # ---- properties --------------------------------------------------------
    @property
    def width(self) -> int:
        if not self._data:
            return 0
        return len(self._data[0])

    @property
    def height(self) -> int:
        return len(self._data)

    # ---- pixel access ------------------------------------------------------
    def get_pixel(self, x: int, y: int) -> Pixel:
        return self._data[y][x]

    def set_pixel(self, x: int, y: int, value: Pixel) -> None:
        self._data[y][x] = value

    # ---- resize (bilinear interpolation) -----------------------------------
    def resize(self, new_width: int, new_height: int) -> "Image":
        if new_width <= 0 or new_height <= 0:
            raise ValueError("Target dimensions must be positive")
        src_h = self.height
        src_w = self.width
        out = Image.blank(new_width, new_height)
        for y2 in range(new_height):
            for x2 in range(new_width):
                # map destination to source coordinates
                sx = x2 * (src_w - 1) / max(new_width - 1, 1)
                sy = y2 * (src_h - 1) / max(new_height - 1, 1)
                x0 = int(math.floor(sx))
                y0 = int(math.floor(sy))
                x1 = min(x0 + 1, src_w - 1)
                y1 = min(y0 + 1, src_h - 1)
                fx = sx - x0
                fy = sy - y0
                p00 = self.get_pixel(x0, y0)
                p10 = self.get_pixel(x1, y0)
                p01 = self.get_pixel(x0, y1)
                p11 = self.get_pixel(x1, y1)
                r = int(
                    p00[0] * (1 - fx) * (1 - fy)
                    + p10[0] * fx * (1 - fy)
                    + p01[0] * (1 - fx) * fy
                    + p11[0] * fx * fy
                )
                g = int(
                    p00[1] * (1 - fx) * (1 - fy)
                    + p10[1] * fx * (1 - fy)
                    + p01[1] * (1 - fx) * fy
                    + p11[1] * fx * fy
                )
                b = int(
                    p00[2] * (1 - fx) * (1 - fy)
                    + p10[2] * fx * (1 - fy)
                    + p01[2] * (1 - fx) * fy
                    + p11[2] * fx * fy
                )
                out.set_pixel(x2, y2, (_clamp(r), _clamp(g), _clamp(b)))
        return out

    # ---- crop --------------------------------------------------------------
    def crop(self, x: int, y: int, w: int, h: int) -> "Image":
        out = Image.blank(w, h)
        for dy in range(h):
            for dx in range(w):
                out.set_pixel(dx, dy, self.get_pixel(x + dx, y + dy))
        return out

    # ---- to_grayscale (returns 2-D float list) -----------------------------
    def to_grayscale(self) -> List[List[float]]:
        return [
            [0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2] for p in row]
            for row in self._data
        ]

    # ---- normalize (0-255 → 0.0-1.0) -------------------------------------
    def normalize(self) -> List[List[Tuple[float, float, float]]]:
        return [
            [(p[0] / 255.0, p[1] / 255.0, p[2] / 255.0) for p in row]
            for row in self._data
        ]

    # ---- data access -------------------------------------------------------
    @property
    def data(self) -> List[List[Pixel]]:
        return self._data

    def copy(self) -> "Image":
        return Image([row[:] for row in self._data])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Image):
            return NotImplemented
        return self._data == other._data


# ---- helpers --------------------------------------------------------------

def _clamp(v: int, lo: int = 0, hi: int = 255) -> int:
    return max(lo, min(hi, v))


# ---- colour-space conversions ---------------------------------------------

def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    rn = r / 255.0
    gn = g / 255.0
    bn = b / 255.0
    cmax = max(rn, gn, bn)
    cmin = min(rn, gn, bn)
    delta = cmax - cmin
    # hue
    if delta == 0:
        h = 0.0
    elif cmax == rn:
        h = 60.0 * (((gn - bn) / delta) % 6)
    elif cmax == gn:
        h = 60.0 * (((bn - rn) / delta) + 2)
    else:
        h = 60.0 * (((rn - gn) / delta) + 4)
    # saturation
    s = 0.0 if cmax == 0 else delta / cmax
    v = cmax
    return (h, s, v)


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    c = v * s
    h2 = h / 60.0
    x = c * (1 - abs(h2 % 2 - 1))
    m = v - c
    if h2 < 1:
        r1, g1, b1 = c, x, 0
    elif h2 < 2:
        r1, g1, b1 = x, c, 0
    elif h2 < 3:
        r1, g1, b1 = 0, c, x
    elif h2 < 4:
        r1, g1, b1 = 0, x, c
    elif h2 < 5:
        r1, g1, b1 = x, 0, c
    else:
        r1, g1, b1 = c, 0, x
    return (
        _clamp(int((r1 + m) * 255)),
        _clamp(int((g1 + m) * 255)),
        _clamp(int((b1 + m) * 255)),
    )


def rgb_to_grayscale(r: int, g: int, b: int) -> float:
    return 0.299 * r + 0.587 * g + 0.114 * b


# ---- augmenter ------------------------------------------------------------

class ImageAugmenter:
    """Composable image augmentation for training data."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    # individual transforms — each returns a *new* Image -------

    def random_flip(self, img: Image, p: float = 0.5) -> Image:
        if self._rng.random() < p:
            out = img.copy()
            for y in range(out.height):
                row = out._data[y]
                out._data[y] = row[::-1]
            return out
        return img

    def random_rotate(self, img: Image, max_degrees: float = 15.0) -> Image:
        """Rotate by a random angle in [-max_degrees, max_degrees].

        Uses a simple nearest-neighbour rotation for simulation.
        """
        angle = self._rng.uniform(-max_degrees, max_degrees)
        rad = math.radians(angle)
        cx = img.width / 2.0
        cy = img.height / 2.0
        out = img.copy()
        for y2 in range(img.height):
            for x2 in range(img.width):
                dx = x2 - cx
                dy = y2 - cy
                sx = dx * math.cos(rad) - dy * math.sin(rad) + cx
                sy = dx * math.sin(rad) + dy * math.cos(rad) + cy
                ix = int(round(sx))
                iy = int(round(sy))
                if 0 <= ix < img.width and 0 <= iy < img.height:
                    out.set_pixel(x2, y2, img.get_pixel(ix, iy))
                else:
                    out.set_pixel(x2, y2, (0, 0, 0))
        return out

    def random_brightness(self, img: Image, delta: float = 30.0) -> Image:
        factor = 1.0 + self._rng.uniform(-delta / 255.0, delta / 255.0)
        out = img.copy()
        for y in range(img.height):
            for x in range(img.width):
                r, g, b = img.get_pixel(x, y)
                out.set_pixel(x, y, (_clamp(int(r * factor)),
                                       _clamp(int(g * factor)),
                                       _clamp(int(b * factor))))
        return out

    def random_noise(self, img: Image, sigma: float = 10.0) -> Image:
        out = img.copy()
        for y in range(img.height):
            for x in range(img.width):
                r, g, b = img.get_pixel(x, y)
                nr = self._rng.gauss(0, sigma)
                ng = self._rng.gauss(0, sigma)
                nb = self._rng.gauss(0, sigma)
                out.set_pixel(x, y, (_clamp(int(r + nr)),
                                       _clamp(int(g + ng)),
                                       _clamp(int(b + nb))))
        return out

    def random_crop(self, img: Image, min_ratio: float = 0.7) -> Image:
        ratio = self._rng.uniform(min_ratio, 1.0)
        cw = max(1, int(img.width * ratio))
        ch = max(1, int(img.height * ratio))
        sx = self._rng.randint(0, max(0, img.width - cw))
        sy = self._rng.randint(0, max(0, img.height - ch))
        return img.crop(sx, sy, cw, ch)

    # compose ----------------------------------------------------------------

    def compose(self, img: Image, transforms: List[Callable[[Image], Image]]) -> Image:
        cur = img
        for t in transforms:
            cur = t(cur)
        return cur
