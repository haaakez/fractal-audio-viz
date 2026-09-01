"""Named render presets shared by the CLI and the small desktop GUI.

The values are ordinary command-line defaults.  Any option written after
``--profile`` still wins, so a profile is a starting point rather than a
locked configuration.
"""

from __future__ import annotations


PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "preview": {
        "width": 960,
        "height": 540,
        "fps": 24,
        "quality": "draft",
        "fractal_scale": 0.5,
        "keyframe_factor": 4.0,
        "max_zoom": "1e24",
        "separation": "none",
        "video_preset": "ultrafast",
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
    },
    "4k-e150": {
        "width": 3840,
        "height": 2160,
        "fps": 30,
        "quality": "balanced",
        "fractal_scale": 0.5,
        "keyframe_factor": 8.0,
        "max_zoom": "1e150",
        "separation": "auto",
        "video_preset": "ultrafast",
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
    },
}

PROFILE_DESCRIPTIONS: dict[str, str] = {
    "preview": "Fast low-resolution check render.",
    "fullhd": "Balanced Full HD (1920x1080) render through 1e80.",
    "1080p": "Alias for the Full HD profile.",
    "4k-e150": "The practical 4K/e150 speed target: 2K fractal source.",
    "master-e150": "Higher-density 4K/e150 master; expect a longer render.",
    "beat": "1080p render with onset-driven zoom punches enabled.",
}

PROFILE_CHOICES = tuple(PROFILE_DEFAULTS)
