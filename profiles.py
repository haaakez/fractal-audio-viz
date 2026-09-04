"""Named render presets shared by the CLI and the small desktop GUI.

The values are ordinary command-line defaults.  Any option written after
``--profile`` still wins, so a profile is a starting point rather than a
locked configuration.
"""

from __future__ import annotations


# Normal renders favour source detail and encoder quality. The explicitly fast
# 4K preset remains available for live-view-style experiments and quick checks.
DEFAULT_PROFILE = "4k-e150-lossless"


PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "preview": {
        "width": 960,
        "height": 540,
        "fps": 24,
        "quality": "draft",
        # Keep the quick preview at its requested pixel density.  A half-size
        # scalar source made even this 960x540 profile visibly blocky after
        # the final upscale, especially around KFP's one-pixel edge detail.
        "fractal_scale": 1.0,
        "keyframe_factor": 4.0,
        "max_zoom": "1e24",
        "separation": "none",
        "video_preset": "ultrafast",
        "crf": 18,
        "resample": "bilinear",
        "lossless": False,
    },
    "fullhd": {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "quality": "balanced",
        "fractal_scale": 1.0,
        "keyframe_factor": 4.0,
        "max_zoom": "1e80",
        "separation": "auto",
        "video_preset": "veryfast",
        "crf": 18,
        "resample": "bilinear",
        "lossless": False,
    },
    # Keep the older spelling for scripts and commands that already use it.
    "1080p": {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "quality": "balanced",
        "fractal_scale": 1.0,
        "keyframe_factor": 4.0,
        "max_zoom": "1e80",
        "separation": "auto",
        "video_preset": "veryfast",
        "crf": 18,
        "resample": "bilinear",
        "lossless": False,
    },
    "4k-e150": {
        "width": 3840,
        "height": 2160,
        "fps": 60,
        "quality": "balanced",
        # A 960x540 scalar source is enough for a smooth 4K atlas export and
        # keeps both native field work and per-frame colourisation inside the
        # practical e150 speed budget. Full HD and master profiles retain
        # their native-density sources.
        "fractal_scale": 0.25,
        "keyframe_factor": 8.0,
        "max_zoom": "1e150",
        "separation": "auto",
        "video_preset": "ultrafast",
        "crf": 18,
        "resample": "bilinear",
        "lossless": False,
    },
    "4k-e150-lossless": {
        "width": 3840,
        "height": 2160,
        "fps": 60,
        "quality": "quality",
        "fractal_scale": 1.0,
        "keyframe_factor": 4.0,
        "max_zoom": "1e150",
        "separation": "auto",
        "video_preset": "slow",
        "crf": 0,
        "resample": "lanczos",
        "lossless": True,
    },
    "master-e150": {
        "width": 3840,
        "height": 2160,
        "fps": 30,
        "quality": "quality",
        "fractal_scale": 1.0,
        "keyframe_factor": 2.0,
        "max_zoom": "1e150",
        "separation": "auto",
        "video_preset": "fast",
        "crf": 18,
        "resample": "bilinear",
        "lossless": False,
    },
    "beat": {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "quality": "balanced",
        "fractal_scale": 1.0,
        "keyframe_factor": 4.0,
        "max_zoom": "1e80",
        "separation": "auto",
        "beat_strength": 1.25,
        "video_preset": "veryfast",
        "crf": 18,
        "resample": "bilinear",
        "lossless": False,
    },
}

PROFILE_DESCRIPTIONS: dict[str, str] = {
    "preview": "Fast low-resolution check render.",
    "fullhd": "Balanced Full HD (1920x1080) render through 1e80.",
    "1080p": "Alias for the Full HD profile.",
    "4k-e150": "The practical 4K/e150 speed target: 960x540 fractal source.",
    "4k-e150-lossless": (
        "Native-density 4K/60 e150 render with lossless H.264 and 4:4:4 colour; "
        "expect a much larger file and a longer render."
    ),
    "master-e150": "Higher-density 4K/e150 master; expect a longer render.",
    "beat": "1080p render with onset-driven zoom punches enabled.",
}

PROFILE_CHOICES = tuple(PROFILE_DEFAULTS)
