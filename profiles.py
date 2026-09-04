"""Named render presets shared by the CLI and the small desktop GUI.

The values are ordinary command-line defaults.  Any option written after
``--profile`` still wins, so a profile is a starting point rather than a
locked configuration.
"""

from __future__ import annotations


# The canonical presets are deliberately boring and predictable: the name
# describes the output tier, every tier is 60 fps, every tier targets e100,
# and CRF 10 is visually near-lossless for this material while still producing
# a useful compressed file. The 4K and 8K defaults use the proven half-density
# native field path; users who need one scalar sample per output pixel can add
# ``--quality quality --fractal-scale 1``. Exact bit-level output remains
# available with ``--lossless``.
DEFAULT_PROFILE = "4k60"
NEAR_LOSSLESS_CRF = 10
SOURCE_MODE_CHOICES = ("native", "upscaled")
# The fast export path renders a quarter-size scalar field and lets FFmpeg
# enlarge it to the requested output dimensions.  This is the source density
# used by the old practical 4K speed render, and is intentionally explicit so
# it cannot be selected by accident for a quality export.
UPSCALED_SOURCE_SCALE = 0.25
FAST_PROFILE_CHOICES = ("4k-e150",)
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
        # accents and KFP profiles. With a native source this does not blur
        # the output; it only selects the renderer's crop primitive.
        "resample": "bilinear",
        "lossless": False,
        "source_mode": "native",
    }


PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "sd60": _native_quality_profile(720, 480),
    "hd60": _native_quality_profile(1280, 720),
    "fhd60": _native_quality_profile(1920, 1080),
    "2k60": _native_quality_profile(2560, 1440),
    # This is the pre-live/native pipeline that produced the practical
    # ~250-second 4K renders: native C++ fields at 2K density, then one final
    # output upscale. It is deliberately distinct from the quarter-size
    # ``4k-e150`` compatibility speed profile below.
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
        fractal_scale=0.5,
        keyframe_factor=8.0,
    ),
}

# This is the exact practical 4K speed profile used before the live-view
# work: a 960x540 source, the fused native colour path, and a fast encoder.
# Keep it explicit rather than making a quality profile silently undersample.
PROFILE_DEFAULTS["4k-e150"] = {
    "width": 3840,
    "height": 2160,
    "fps": 60,
    "quality": "balanced",
    "fractal_scale": UPSCALED_SOURCE_SCALE,
    "keyframe_factor": 8.0,
    "max_zoom": "1e150",
    "separation": "auto",
    "video_preset": "ultrafast",
    "crf": 18,
    "resample": "bilinear",
    "lossless": False,
    "source_mode": "upscaled",
}

PROFILE_DESCRIPTIONS: dict[str, str] = {
    "sd60": "Native 720x480 60 fps, near-lossless CRF 10, e100.",
    "hd60": "Native 1280x720 60 fps, near-lossless CRF 10, e100.",
    "fhd60": "Native 1920x1080 60 fps, near-lossless CRF 10, e100.",
    "2k60": "Native 2560x1440 60 fps, near-lossless CRF 10, e100.",
    "4k60": "4K/60 native C++ pipeline from a 1920x1080 source, CRF 10, e100.",
    "8k60": "8K/60 native C++ pipeline from a 3840x2160 source, CRF 10, e100.",
    "4k-e150": "Fast 4K/60 e150 export from a 960x540 source, enlarged at the end.",
}

# Keep existing command lines usable while making the resolution presets the
# canonical choices shown by the GTK selector. These aliases intentionally
# inherit the new native/e100/near-lossless defaults instead of preserving the
# old undersampled or 30 fps behaviour under a misleading profile name. The
# old name that explicitly promised losslessness keeps that one stronger
# guarantee as well.
PROFILE_ALIASES: dict[str, str] = {
    "preview": "sd60",
    "fullhd": "fhd60",
    "1080p": "fhd60",
    "4k-e150-lossless": "4k60",
    "master-e150": "4k60",
    "beat": "fhd60",
}
for _alias, _canonical in PROFILE_ALIASES.items():
    PROFILE_DEFAULTS[_alias] = dict(PROFILE_DEFAULTS[_canonical])
    PROFILE_DESCRIPTIONS[_alias] = f"Compatibility alias for --profile {_canonical}."

# Preserve the observable meaning of the old quality-master name while moving
# its otherwise problematic e150 target to the safe e100 default. This is
# still a native 4K/60 profile, just exact-lossless instead of CRF 10.
PROFILE_DEFAULTS["4k-e150-lossless"].update(crf=0, lossless=True)
PROFILE_DESCRIPTIONS["4k-e150-lossless"] = (
    "Compatibility alias for --profile 4k60 with exact H.264 lossless output."
)

# CLI accepts the canonical profiles, the explicit fast compatibility profile,
# and aliases for existing scripts. The GTK selector chooses from the
# canonical tiers plus the fast profile; aliases remain CLI-only.
PROFILE_CHOICES = (
    CANONICAL_PROFILE_CHOICES
    + FAST_PROFILE_CHOICES
    + tuple(PROFILE_ALIASES)
)
