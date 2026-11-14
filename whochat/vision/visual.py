"""Pixel-level helpers shared by layout refinement and OCR text classification.

Everything in this module works on the *captured window image* only.  It never
touches the WeChat process, its memory, or its database: it just looks at the
screenshot that the capture pipeline already produced.

Two questions are answered here:

* ``VisualContext.tone`` — what does a text box look like?  WeChat renders
  nicknames, timestamps and system hints in a *mid grey* ink, while real message
  text is near-black (light theme) or near-white (dark theme).  Percentile
  luminance of a small crop separates the two reliably, which is much more
  robust than guessing from geometry alone.
* ``VisualContext.bright_row_fractions`` / ``bright_column_fractions`` — where
  are the bright regions in a band of pixels?  Used to locate the chat-list /
  conversation divider and the top edge of the input box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageStat

from whochat.core.runtime import Rect

# A pixel counts as "page bright" when its luminance reaches this value.  In the
# WeChat light theme the chat-list surface sits around 230-240 and the
# conversation surface around 246-250, so 244 separates them cleanly.
BRIGHT_LUMINANCE = 244

# Tone classification thresholds (all in luminance units).
LIGHT_THEME_BACKGROUND = 160.0
DARK_SURFACE_ABSOLUTE_GAP = 90.0
DARK_SURFACE_RELATIVE_RATIO = 0.75
NEUTRAL_CHANNEL_SPREAD = 26.0
SECONDARY_INK_GAP_MIN = 20.0
SECONDARY_INK_GAP_MAX = 130.0
MIN_TONE_SAMPLES = 24

# A box whose 5th..95th luminance range is narrower than this holds no visible
# ink at all (an empty region, or a label the OCR located but whose glyphs are
# not actually painted).  Tone cannot classify such a box, so it must fall back
# to geometry instead of being silently reported as message text.
FLAT_PATCH_RANGE = 10.0

# Kinds returned by ``VisualContext.box_kind``.
KIND_PRIMARY = "primary"
KIND_SECONDARY = "secondary"
KIND_DARK_OVERLAY = "dark_overlay"
KIND_UNKNOWN = "unknown"


def load_rgb_image(source: str | Path | Image.Image | None) -> Image.Image | None:
    """Decode one path or borrow an already loaded RGB frame without copying.

    Path-backed files are closed before the independent RGB image is returned.
    An RGB input stays owned by its caller: users must not mutate or close it.
    Non-RGB inputs are converted into a new image without changing the source.
    There is no retained image or cache across calls, jobs or customers.
    """
    if source is None or source == "":
        return None
    try:
        if isinstance(source, Image.Image):
            return source if source.mode == "RGB" else source.convert("RGB")
        with Image.open(source) as image:
            return image.convert("RGB")
    except (OSError, ValueError, IndexError):
        return None


@dataclass(frozen=True)
class ToneSample:
    """Ink vs. surface luminance of a single OCR text box."""

    ink: float
    surface: float
    spread: float
    samples: int
    low: float = 0.0
    high: float = 255.0

    @property
    def is_neutral(self) -> bool:
        return self.spread <= NEUTRAL_CHANNEL_SPREAD

    @property
    def is_flat(self) -> bool:
        """No visible ink inside the box: every pixel has essentially one tone."""
        return (self.high - self.low) <= FLAT_PATCH_RANGE


class VisualContext:
    """Lazily derived pixel statistics for one captured window image."""

    def __init__(self, rgb: Image.Image, gray: Image.Image) -> None:
        self._rgb = rgb
        self._gray = gray
        self._percentile_cache: dict[tuple[int, int, int, int, int], float] = {}
        self._background_cache: dict[tuple[int, int, int, int], float] = {}

    # ------------------------------------------------------------------ setup

    @classmethod
    def load(cls, image_path: str | Path | Image.Image | None) -> VisualContext | None:
        try:
            rgb = load_rgb_image(image_path)
            if rgb is None or rgb.width < 64 or rgb.height < 64:
                return None
            return cls(rgb, rgb.convert("L"))
        except (OSError, ValueError, IndexError):
            return None

    @property
    def width(self) -> int:
        return self._rgb.width

    @property
    def height(self) -> int:
        return self._rgb.height

    # -------------------------------------------------------------- geometry

    def clamp(self, rect: Rect) -> Rect | None:
        left = max(0, min(rect.left, self.width))
        top = max(0, min(rect.top, self.height))
        right = max(left, min(rect.right, self.width))
        bottom = max(top, min(rect.bottom, self.height))
        if right - left < 2 or bottom - top < 2:
            return None
        return Rect(left, top, right, bottom)

    # ----------------------------------------------------------------- tones

    def tone(self, rect: Rect) -> ToneSample | None:
        """Ink / surface / colour-spread statistics of the pixels inside ``rect``."""
        box = self.clamp(rect)
        if box is None:
            return None
        crop = self._gray.crop(box.as_tuple())
        histogram = crop.histogram()
        total = sum(histogram)
        if total < MIN_TONE_SAMPLES:
            return None
        ink = _percentile(histogram, total, 0.15)
        surface = _percentile(histogram, total, 0.60)
        mean = ImageStat.Stat(self._rgb.crop(box.as_tuple())).mean
        spread = max(mean) - min(mean)
        low = _percentile(histogram, total, 0.05)
        high = _percentile(histogram, total, 0.95)
        return ToneSample(ink=ink, surface=surface, spread=spread, samples=total, low=low, high=high)

    def page_background(self, rect: Rect) -> float:
        """Bright surface tone of a region (90th percentile luminance)."""
        box = self.clamp(rect)
        if box is None:
            return 0.0
        key = box.as_tuple()
        cached = self._background_cache.get(key)
        if cached is not None:
            return cached
        histogram = self._gray.crop(key).histogram()
        total = sum(histogram)
        value = _percentile(histogram, total, 0.90) if total else 0.0
        self._background_cache[key] = value
        return value

    def box_kind(self, rect: Rect, reference_background: float) -> str:
        """Classify a text box as primary text, grey metadata, or dark overlay.

        ``reference_background`` is the bright surface tone of the surrounding
        region (usually ``page_background(message_rect)``).  The dark-theme
        branch simply mirrors the light-theme rule, so both themes are covered
        without a separate "theme" switch.
        """
        sample = self.tone(rect)
        if sample is None or sample.is_flat:
            return KIND_UNKNOWN
        if _looks_like_dark_surface(sample.surface, reference_background):
            return KIND_DARK_OVERLAY
        if sample.is_neutral and _ink_is_secondary(sample.ink, reference_background):
            return KIND_SECONDARY
        return KIND_PRIMARY

    # ------------------------------------------------- pixel band utilities

    def bright_column_fractions(self, box: Rect, threshold: int = BRIGHT_LUMINANCE) -> list[float]:
        """Fraction of bright pixels per column inside ``box`` (left to right)."""
        return self._band_fractions(box, threshold, horizontal=True)

    def bright_row_fractions(self, box: Rect, threshold: int = BRIGHT_LUMINANCE) -> list[float]:
        """Fraction of bright pixels per row inside ``box`` (top to bottom)."""
        return self._band_fractions(box, threshold, horizontal=False)

    def column_mean_luminance(self, box: Rect) -> list[float]:
        """Mean luminance per column inside ``box`` (left to right)."""
        clipped = self.clamp(box)
        if clipped is None:
            return []
        region = self._gray.crop(clipped.as_tuple())
        reduced = region.resize((clipped.width, 1), Image.Resampling.BOX)
        return [float(value) for value in reduced.tobytes()]

    def row_mean_luminance(self, box: Rect) -> list[float]:
        """Mean luminance per row inside the requested box."""
        clipped = self.clamp(box)
        if clipped is None:
            return []
        region = self._gray.crop(clipped.as_tuple())
        reduced = region.resize((1, clipped.height), Image.Resampling.BOX)
        return [float(value) for value in reduced.tobytes()]

    def _band_fractions(self, box: Rect, threshold: int, *, horizontal: bool) -> list[float]:
        clipped = self.clamp(box)
        if clipped is None:
            return []
        region = self._gray.crop(clipped.as_tuple())
        mask = region.point(lambda value: 255 if value >= threshold else 0)
        size = (clipped.width, 1) if horizontal else (1, clipped.height)
        reduced = mask.resize(size, Image.Resampling.BOX)
        return [value / 255.0 for value in reduced.tobytes()]

    # ----------------------------------------------------------- bubble fill

    def horizontal_separator_support(self, box: Rect, y: int, *, run: int = 1, min_contrast: float = 5.0) -> float:
        """Fraction of columns with the same thin line against both surfaces.

        A row mean alone mistakes a text stroke in a bubble for a divider.
        Real input borders span almost the full conversation width.
        """
        clipped = self.clamp(box)
        if clipped is None or clipped.width < 100 or run not in (1, 2, 3):
            return 0.0
        if y - 4 < clipped.top or y + run + 3 >= clipped.bottom:
            return 0.0
        pixels = self._gray.load()
        supporting = 0
        total = 0
        for x in range(clipped.left, clipped.right):
            above = sum(pixels[x, row] for row in range(y - 4, y - 1)) / 3
            below = sum(pixels[x, row] for row in range(y + run + 1, y + run + 4)) / 3
            line = sum(pixels[x, row] for row in range(y, y + run)) / run
            total += 1
            if (abs(above - below) <= 14 and abs(line - above) >= min_contrast
                    and abs(line - below) >= min_contrast and (line - above) * (line - below) > 0):
                supporting += 1
        return supporting / max(1, total)

    def green_fill_fraction(self, upper: Rect, lower: Rect) -> float:
        """Share of *outgoing bubble green* in the band between two text boxes.

        Used to prove that two OCR fragments belong to one outgoing chat bubble:
        when a quoted line and the main line are separated by solid green fill,
        they are one message, not two.  Only green counts — the page background
        is a neutral grey that would otherwise match every pair of neighbours.
        """
        from whochat.vision.bubbles import _is_green_pixel

        top = upper.bottom
        bottom = lower.top
        if bottom - top < 2:
            return 0.0
        left = max(upper.left, lower.left) - 6
        right = min(upper.right, lower.right) + 6
        box = self.clamp(Rect(int(left), int(top), int(right), int(bottom)))
        if box is None or box.width < 6 or box.height < 2:
            return 0.0
        pixels = list(self._rgb.crop(box.as_tuple()).tobytes())
        total = box.width * box.height
        filled = 0
        for index in range(0, total * 3, 3):
            if _is_green_pixel((pixels[index], pixels[index + 1], pixels[index + 2])):
                filled += 1
        return filled / max(1, total)


def _percentile(histogram: list[int], total: int, fraction: float) -> float:
    if total <= 0:
        return 0.0
    target = total * fraction
    running = 0
    for value, count in enumerate(histogram):
        running += count
        if running >= target:
            return float(value)
    return float(len(histogram) - 1)


def _looks_like_dark_surface(surface: float, reference_background: float) -> bool:
    """True when a box sits on a dark badge *painted over a bright page*.

    This is how WeChat renders a voice / video duration label: a small dark pill
    on the light conversation surface.  The rule is only meaningful in the light
    theme, where the surrounding surface is bright.  In the dark theme the
    reference is itself dark (typically the bubble fill), and a box sitting on
    the plain dark page would look "darker than the reference" for an entirely
    different reason — that is page background, not a badge.  Requiring a bright
    reference keeps the duration verdict from firing on ordinary dark-theme page
    pixels.
    """
    if reference_background < LIGHT_THEME_BACKGROUND:
        return False
    if surface <= reference_background - DARK_SURFACE_ABSOLUTE_GAP:
        return True
    return surface <= reference_background * DARK_SURFACE_RELATIVE_RATIO - 2


def _ink_is_secondary(ink: float, reference_background: float) -> bool:
    if reference_background <= 0:
        return False
    if reference_background >= LIGHT_THEME_BACKGROUND:
        low = reference_background - SECONDARY_INK_GAP_MAX
        high = reference_background - SECONDARY_INK_GAP_MIN
    else:
        low = reference_background + SECONDARY_INK_GAP_MIN
        high = reference_background + SECONDARY_INK_GAP_MAX
    return low <= ink <= high
