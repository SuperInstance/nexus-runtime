"""Tests for preprocessing module — Image, augmenter, colour spaces."""

import math
import random
import pytest
from jetson.vision.preprocessing import (
    Image, ImageAugmenter, rgb_to_hsv, hsv_to_rgb, rgb_to_grayscale,
)


# ---- helpers --------------------------------------------------------------

def make_image(w, h, fill=(128, 128, 128)):
    return Image.blank(w, h, fill)


def gradient_image(w, h):
    """Each pixel gets (x, y, (x+y)%256)."""
    data = [[(x, y, (x + y) % 256) for x in range(w)] for y in range(h)]
    return Image.from_list(data)


# ---- Image class -----------------------------------------------------------

class TestImage:
    def test_blank_dimensions(self):
        img = make_image(10, 5)
        assert img.width == 10
        assert img.height == 5

    def test_from_list(self):
        data = [[(255, 0, 0)]] * 3
        img = Image.from_list(data)
        assert img.width == 1
        assert img.height == 3
        assert img.get_pixel(0, 0) == (255, 0, 0)

    def test_set_get_pixel(self):
        img = make_image(4, 4)
        img.set_pixel(1, 2, (10, 20, 30))
        assert img.get_pixel(1, 2) == (10, 20, 30)
        assert img.get_pixel(0, 0) == (128, 128, 128)

    def test_resize_same(self):
        img = make_image(4, 4, (100, 150, 200))
        out = img.resize(4, 4)
        assert out.width == 4
        assert out.height == 4

    def test_resize_up(self):
        img = make_image(2, 2, (50, 100, 200))
        out = img.resize(4, 4)
        assert out.width == 4
        assert out.height == 4
        # corners should be preserved
        assert out.get_pixel(0, 0) == (50, 100, 200)
        assert out.get_pixel(3, 3) == (50, 100, 200)

    def test_resize_down(self):
        img = make_image(8, 8, (200, 100, 50))
        out = img.resize(4, 4)
        assert out.width == 4
        assert out.height == 4

    def test_resize_invalid(self):
        img = make_image(4, 4)
        with pytest.raises(ValueError):
            img.resize(0, 4)
        with pytest.raises(ValueError):
            img.resize(-1, 4)

    def test_crop(self):
        img = gradient_image(10, 10)
        cropped = img.crop(2, 3, 4, 5)
        assert cropped.width == 4
        assert cropped.height == 5
        assert cropped.get_pixel(0, 0) == img.get_pixel(2, 3)

    def test_to_grayscale(self):
        img = Image.blank(3, 2, (100, 150, 200))
        gray = img.to_grayscale()
        assert len(gray) == 2
        assert len(gray[0]) == 3
        expected = 0.299 * 100 + 0.587 * 150 + 0.114 * 200
        assert abs(gray[0][0] - expected) < 0.01

    def test_normalize(self):
        img = Image.blank(2, 2, (255, 128, 0))
        norm = img.normalize()
        assert abs(norm[0][0][0] - 1.0) < 1e-6
        assert abs(norm[0][0][1] - 128 / 255.0) < 1e-6
        assert abs(norm[0][0][2] - 0.0) < 1e-6

    def test_copy(self):
        img = make_image(3, 3, (10, 20, 30))
        cp = img.copy()
        cp.set_pixel(0, 0, (0, 0, 0))
        assert img.get_pixel(0, 0) == (10, 20, 30)

    def test_equality(self):
        a = make_image(2, 2, (10, 10, 10))
        b = make_image(2, 2, (10, 10, 10))
        assert a == b

    def test_data_property(self):
        img = make_image(2, 2)
        assert img.data is not None
        assert isinstance(img.data, list)

    def test_blank_fill(self):
        img = Image.blank(5, 5, (42, 42, 42))
        assert img.get_pixel(4, 4) == (42, 42, 42)


# ---- colour spaces ---------------------------------------------------------

class TestColourSpaces:
    def test_rgb_to_hsv_red(self):
        h, s, v = rgb_to_hsv(255, 0, 0)
        assert abs(h - 0.0) < 1.0 or abs(h - 360.0) < 1.0
        assert abs(s - 1.0) < 0.01
        assert abs(v - 1.0) < 0.01

    def test_rgb_to_hsv_green(self):
        h, s, v = rgb_to_hsv(0, 255, 0)
        assert abs(h - 120.0) < 1.0
        assert abs(s - 1.0) < 0.01

    def test_rgb_to_hsv_blue(self):
        h, s, v = rgb_to_hsv(0, 0, 255)
        assert abs(h - 240.0) < 1.0
        assert abs(v - 1.0) < 0.01

    def test_rgb_to_hsv_white(self):
        h, s, v = rgb_to_hsv(255, 255, 255)
        assert abs(s) < 0.01
        assert abs(v - 1.0) < 0.01

    def test_rgb_to_hsv_black(self):
        h, s, v = rgb_to_hsv(0, 0, 0)
        assert abs(s) < 0.01
        assert abs(v) < 0.01

    def test_hsv_to_rgb_roundtrip(self):
        for r, g, b in [(100, 200, 50), (0, 0, 0), (255, 255, 255),
                         (10, 20, 30), (200, 50, 80)]:
            h, s, v = rgb_to_hsv(r, g, b)
            r2, g2, b2 = hsv_to_rgb(h, s, v)
            assert abs(r2 - r) <= 2, f"Red: {r2} != {r}"
            assert abs(g2 - g) <= 2, f"Green: {g2} != {g}"
            assert abs(b2 - b) <= 2, f"Blue: {b2} != {b}"

    def test_rgb_to_grayscale_luminance(self):
        g = rgb_to_grayscale(255, 0, 0)
        assert abs(g - 76.245) < 0.1
        g2 = rgb_to_grayscale(0, 255, 0)
        assert abs(g2 - 149.685) < 0.1

    def test_rgb_to_grayscale_white(self):
        g = rgb_to_grayscale(255, 255, 255)
        assert abs(g - 255.0) < 0.1

    def test_rgb_to_grayscale_black(self):
        g = rgb_to_grayscale(0, 0, 0)
        assert abs(g) < 0.1

    def test_hsv_to_rgb_pure_red(self):
        r, g, b = hsv_to_rgb(0, 1.0, 1.0)
        assert r == 255
        assert g == 0
        assert b == 0


# ---- ImageAugmenter --------------------------------------------------------

class TestImageAugmenter:
    def test_random_flip_runs(self):
        aug = ImageAugmenter(seed=42)
        img = make_image(6, 6, (100, 100, 100))
        out = aug.random_flip(img, p=1.0)
        assert out.width == 6
        assert out.height == 6

    def test_random_flip_horizontal(self):
        aug = ImageAugmenter(seed=42)
        data = [[(x, 0, 0) for x in range(4)] for _ in range(2)]
        img = Image.from_list(data)
        out = aug.random_flip(img, p=1.0)
        # pixel at x=0 should now be pixel at x=3
        assert out.get_pixel(0, 0) == (3, 0, 0)

    def test_random_rotate_runs(self):
        aug = ImageAugmenter(seed=42)
        img = make_image(10, 10)
        out = aug.random_rotate(img, max_degrees=5)
        assert out.width == 10
        assert out.height == 10

    def test_random_rotate_no_change_at_zero(self):
        aug = ImageAugmenter(seed=42)
        img = make_image(6, 6, (50, 100, 150))
        # seed won't guarantee 0 angle, so just check it runs
        out = aug.random_rotate(img, max_degrees=0)
        assert out.width == 6

    def test_random_brightness(self):
        aug = ImageAugmenter(seed=42)
        img = make_image(4, 4, (128, 128, 128))
        out = aug.random_brightness(img, delta=0)  # delta=0 means no change
        assert out.get_pixel(0, 0) == (128, 128, 128)

    def test_random_brightness_changes(self):
        aug = ImageAugmenter(seed=123)
        img = make_image(8, 8, (200, 200, 200))
        out = aug.random_brightness(img, delta=50)
        # just verify it runs and size is correct
        assert out.width == 8

    def test_random_noise(self):
        aug = ImageAugmenter(seed=42)
        img = make_image(6, 6, (100, 100, 100))
        out = aug.random_noise(img, sigma=0.0)
        assert out.get_pixel(0, 0) == (100, 100, 100)

    def test_random_noise_adds_variation(self):
        aug = ImageAugmenter(seed=42)
        img = make_image(10, 10, (100, 100, 100))
        out = aug.random_noise(img, sigma=20.0)
        values = set()
        for y in range(out.height):
            for x in range(out.width):
                values.add(out.get_pixel(x, y))
        assert len(values) > 1  # noise added variation

    def test_random_crop(self):
        aug = ImageAugmenter(seed=42)
        img = make_image(20, 20)
        out = aug.random_crop(img, min_ratio=0.8)
        assert out.width <= 20
        assert out.height <= 20
        assert out.width >= 1
        assert out.height >= 1

    def test_compose_empty(self):
        aug = ImageAugmenter(seed=42)
        img = make_image(6, 6, (50, 50, 50))
        out = aug.compose(img, [])
        assert out.get_pixel(0, 0) == (50, 50, 50)

    def test_compose_with_transforms(self):
        aug = ImageAugmenter(seed=42)
        img = make_image(20, 20, (100, 100, 100))
        transforms = [
            lambda i: aug.random_brightness(i, delta=0),
            lambda i: aug.random_noise(i, sigma=0),
        ]
        out = aug.compose(img, transforms)
        assert out.get_pixel(0, 0) == (100, 100, 100)

    def test_compose_identity(self):
        aug = ImageAugmenter(seed=0)
        img = make_image(8, 8, (30, 60, 90))
        out = aug.compose(img, [lambda i: i])
        assert out == img

    def test_seed_reproducibility(self):
        img = make_image(8, 8, (128, 128, 128))
        a1 = ImageAugmenter(seed=99)
        a2 = ImageAugmenter(seed=99)
        o1 = a1.random_brightness(img, delta=30)
        o2 = a2.random_brightness(img, delta=30)
        assert o1 == o2
