"""Named render presets shared by the CLI and the small desktop GUI.

The values are ordinary command-line defaults.  Any option written after
``--profile`` still wins, so a profile is a starting point rather than a
locked configuration.
"""

from __future__ import annotations


# The canonical presets are deliberately boring and predictable: the name
# describes the output tier, every tier is 60 fps, every tier targets e100,
# and CRF 10 is visually near-lossless for this material while still producing
# a useful compressed file. Outputs above Full HD use a native C++ field capped
# at 1920x1080; users who need one scalar sample per output pixel can add
# ``--quality quality --fractal-scale 1``. Exact bit-level output remains
# available with ``--lossless``.
DEFAULT_PROFILE = "4k60"
NEAR_LOSSLESS_CRF = 10
SOURCE_MODE_CHOICES = ("lossless-compressed", "native", "upscaled")
SOURCE_MODE_LABELS = {
    "lossless-compressed": "lossless compressed",
    "native": "native",
    "upscaled": "upscaled",
}
# The fast export path renders a quarter-size scalar field and lets FFmpeg
# enlarge it to the requested output dimensions. It remains an explicit source
# mode so it cannot be selected by accident for a quality export.
UPSCALED_SOURCE_SCALE = 0.25
CANONICAL_PROFILE_CHOICES = (
    "sd60",
    "hd60",
    "fhd60",
    "2k60",
    "4k60",
    "8k60",
)


def _native_quality_profile(
    width: int,
    height: int,
    *,
    quality: str = "quality",
    fractal_scale: float = 1.0,
    keyframe_factor: float = 4.0,
) -> dict[str, object]:
    return {
        "width": width,
        "height": height,
        "fps": 60,
        "quality": quality,
        "fractal_scale": fractal_scale,
        "keyframe_factor": keyframe_factor,
        "max_zoom": "1e100",
        "separation": "auto",
        # CRF controls the visual quality; a faster preset primarily trades
        # compression efficiency for much shorter encode time. This is the
        # sensible default for a near-lossless export, especially at 8K.
        "video_preset": "faster",
        "crf": NEAR_LOSSLESS_CRF,
        # Bilinear keeps the fused native atlas colourizer active for Aurora
        # accents and KFP profiles; it selects the renderer's crop primitive
        # without adding a Python image round-trip.
        "resample": "bilinear",
        "lossless": False,
        "source_mode": "lossless-compressed",
    }


PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "sd60": _native_quality_profile(
        720, 480, quality="balanced", fractal_scale=1.0, keyframe_factor=8.0
    ),
    "hd60": _native_quality_profile(
        1280, 720, quality="balanced", fractal_scale=1.0, keyframe_factor=8.0
    ),
    "fhd60": _native_quality_profile(
        1920, 1080, quality="balanced", fractal_scale=1.0, keyframe_factor=8.0
    ),
    # The 1080p field cap keeps every canonical native profile on the same
    # fast, fused C++ path.  At or below Full HD the field is output-density;
    # larger outputs receive one final high-quality upscale.
    "2k60": _native_quality_profile(
        2560,
        1440,
        quality="balanced",
        fractal_scale=0.75,
        keyframe_factor=8.0,
    ),
    "4k60": _native_quality_profile(
        3840,
        2160,
        quality="balanced",
        fractal_scale=0.5,
        keyframe_factor=8.0,
    ),
    "8k60": _native_quality_profile(
        7680,
        4320,
        quality="balanced",
        fractal_scale=0.25,
        keyframe_factor=8.0,
    ),
}

PROFILE_DESCRIPTIONS: dict[str, str] = {
    "sd60": "Lossless-compressed 720x480 60 fps, CRF 10, e100.",
    "hd60": "Lossless-compressed 1280x720 60 fps, CRF 10, e100.",
    "fhd60": "Lossless-compressed 1920x1080 60 fps, CRF 10, e100.",
    "2k60": "2K/60 lossless-compressed C++ pipeline from 1920x1080, CRF 10, e100.",
    "4k60": "4K/60 lossless-compressed C++ pipeline from 1920x1080, CRF 10, e100.",
    "8k60": "8K/60 lossless-compressed C++ pipeline from 1920x1080, CRF 10, e100.",
}

# Keep existing command lines usable while making the resolution presets the
# canonical choices shown by the GTK selector. These aliases intentionally
# inherit the new lossless-compressed/e100/near-lossless defaults instead of
# preserving the old undersampled or 30 fps behaviour under a misleading
# profile name. The old name that explicitly promised losslessness keeps that
# one stronger
# guarantee as well.
PROFILE_ALIASES: dict[str, str] = {
    "preview": "sd60",
    "fullhd": "fhd60",
    "1080p": "fhd60",
    "beat": "fhd60",
}
for _alias, _canonical in PROFILE_ALIASES.items():
    PROFILE_DEFAULTS[_alias] = dict(PROFILE_DEFAULTS[_canonical])
    PROFILE_DESCRIPTIONS[_alias] = f"Compatibility alias for --profile {_canonical}."

# CLI accepts the canonical profiles and aliases for existing scripts. The GTK
# selector chooses only from the canonical resolution tiers; aliases remain
# CLI-only.
PROFILE_CHOICES = CANONICAL_PROFILE_CHOICES + tuple(PROFILE_ALIASES)
