from __future__ import annotations

from collections import Counter
import colorsys
from statistics import median

import av

from . import utils


PALETTE = {
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "orange": (255, 128, 0),
    "yellow": (255, 255, 0),
    "green": (0, 255, 0),
    "cyan": (0, 255, 255),
    "blue": (0, 0, 255),
    "magenta": (255, 0, 255),
}


def _color_name(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    hue, saturation, value = colorsys.rgb_to_hsv(
        r / 255, g / 255, b / 255
    )

    # Bright yellow text should take priority over a darker red outline.
    if r >= 150 and g >= 120 and b <= min(r, g) * 0.75:
        return "yellow"

    if saturation < 0.18:
        if value >= 0.72:
            return "white"
        if value >= 0.35:
            return "gray"
        return "black"

    if hue < 0.045 or hue >= 0.96:
        return "red"
    if hue < 0.105:
        return "orange"
    if hue < 0.205:
        return "yellow"
    if hue < 0.47:
        return "green"
    if hue < 0.57:
        return "cyan"
    if hue < 0.76:
        return "blue"
    if hue < 0.96:
        return "magenta"
    return "red"


def estimate_ass_color(
    video_path: str,
    timestamp_ms: float,
    boxes: list[tuple[float, float, float, float]],
    sample_aspect_ratio: float,
    red_white_threshold: float | None = None,
    white_protection_strength: float | None = None,
) -> str | None:
    """
    Approximate the main fill color of text.
    Dark outlines and shadows are reduced by selecting the brightest pixels.
    """
    if not boxes:
        return None

    with av.open(video_path) as container:
        stream = container.streams.video[0]

        if stream.time_base is None:
            return None

        target_pts = max(
            0,
            int((timestamp_ms / 1000) / float(stream.time_base)),
        )

        container.seek(
            target_pts,
            stream=stream,
            backward=True,
        )

        chosen_frame = None

        for frame in container.decode(stream):
            chosen_frame = frame

            if frame.pts is not None:
                frame_ms = float(frame.pts * stream.time_base) * 1000

                if frame_ms >= timestamp_ms:
                    break

        if chosen_frame is None:
            return None

        rgb_frame = chosen_frame.reformat(format="rgb24")
        image = utils.video_frame_to_frame(rgb_frame)

    width, height = image.width, image.height
    pixels = memoryview(image.data)
    aspect = sample_aspect_ratio if sample_aspect_ratio > 0 else 1.0

    sampled_pixels: list[tuple[int, int, int]] = []

    def pixel(x: int, y: int) -> tuple[int, int, int]:
        offset = (y * width + x) * 3
        return (
            pixels[offset],
            pixels[offset + 1],
            pixels[offset + 2],
        )

    def brightness(rgb: tuple[int, int, int]) -> float:
        r, g, b = rgb
        return 0.299 * r + 0.587 * g + 0.114 * b

    for left, top, right, bottom in boxes:
        x1 = max(0, min(width - 1, int(left / aspect)))
        x2 = max(0, min(width - 1, int(right / aspect)))
        y1 = max(0, min(height - 1, int(top)))
        y2 = max(0, min(height - 1, int(bottom)))

        if x2 <= x1 or y2 <= y1:
            continue

        step = max(1, (x2 - x1) // 120)
        pad = max(2, (y2 - y1) // 6)

        # Estimate the background color near the text area.
        border: list[tuple[int, int, int]] = []

        for y in (
            max(0, y1 - pad),
            min(height - 1, y2 + pad),
        ):
            for x in range(x1, x2 + 1, step):
                border.append(pixel(x, y))

        if not border:
            continue

        background = tuple(
            int(median(channel))
            for channel in zip(*border)
        )

        # Collect pixels that differ from the background.
        for y in range(y1, y2 + 1, max(1, step // 2)):
            for x in range(x1, x2 + 1, step):
                rgb = pixel(x, y)

                difference = sum(
                    abs(rgb[i] - background[i])
                    for i in range(3)
                )

                if difference >= 90:
                    sampled_pixels.append(rgb)

    if not sampled_pixels:
        return None

    # Look for saturated pixels that may belong to the text.
    # Exclude low-saturation pixels when selecting a colored candidate.
    colored_pixels: list[tuple[int, int, int]] = []

    for rgb in sampled_pixels:
        r, g, b = rgb
        _, saturation, value = colorsys.rgb_to_hsv(
            r / 255,
            g / 255,
            b / 255,
        )

        luminance = 0.299 * r + 0.587 * g + 0.114 * b

        # Filter out low-saturation and dark pixels.
        if saturation >= 0.18 and value >= 0.45 and luminance >= 85:
            colored_pixels.append(rgb)
    raw_counts = Counter(_color_name(rgb) for rgb in sampled_pixels)


    if colored_pixels:
        # Select the brightest colored pixels to reduce the influence of dark outlines.
        brightness_values = sorted(
            brightness(rgb) for rgb in colored_pixels
        )
        cutoff = brightness_values[int((len(brightness_values) - 1) * 0.75)]
        pixels_for_color = [
            rgb for rgb in colored_pixels
            if brightness(rgb) >= cutoff
        ]
    else:
        pixels_for_color = sampled_pixels

    counts: Counter[str] = Counter(
        _color_name(rgb) for rgb in pixels_for_color
    )
    if not counts:
        return None

    name = counts.most_common(1)[0][0]

    # Determine which white-pixel threshold applies.
    # The personal red threshold overrides general protection for red only.
    white_threshold: float | None = None

    if name == "red" and red_white_threshold is not None:
        white_threshold = red_white_threshold
    elif white_protection_strength is not None:
        # Higher strength means a lower required share of white pixels.
        white_threshold = 1.0 - (white_protection_strength / 100.0)

    # Apply protection only to colors, not to neutral shades.
    if (
        white_threshold is not None
        and name not in ("white", "gray", "black")
        and raw_counts["white"] >= len(sampled_pixels) * white_threshold
    ):
        name = "white"

    r, g, b = PALETTE[name]
    return f"&H{b:02X}{g:02X}{r:02X}"