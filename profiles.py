"""Named render presets shared by the CLI and the small desktop GUI.

The values are ordinary command-line defaults.  Any option written after
``--profile`` still wins, so a profile is a starting point rather than a
locked configuration.
"""

from __future__ import annotations


# The canonical presets are deliberately boring and predictable: the name
# describes the output tier, every tier is 60 fps, and every tier gets a
# native-resolution scalar field instead of an undersampled fractal enlarged
# at the end. CRF 10 is visually near-lossless for this material while still
# producing a useful compressed file; users who need bit-exact output can add
# ``--lossless``.
DEFAULT_PROFILE = "4k60"
NEAR_LOSSLESS_CRF = 10
CANONICAL_PROFILE_CHOICES = (
    "sd60",
    "hd60",
    "fhd60",
    "2k60",
    "4k60",
    "8k60",
)


def _native_quality_profile(width: int, height: int) -> dict[str, object]:
    return {
        "width": width,
        "height": height,
        "fps": 60,
        "quality": "quality",
        "fractal_scale": 1.0,
        "keyframe_factor": 4.0,
        "max_zoom": "1e100",
        "separation": "auto",
        "video_preset": "slow",
        "crf": NEAR_LOSSLESS_CRF,
        "resample": "lanczos",
        "lossless": False,
    }


PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "sd60": _native_quality_profile(720, 480),
    "hd60": _native_quality_profile(1280, 720),
    "fhd60": _native_quality_profile(1920, 1080),
    "2k60": _native_quality_profile(2560, 1440),
    "4k60": _native_quality_profile(3840, 2160),
    "8k60": _native_quality_profile(7680, 4320),
}

PROFILE_DESCRIPTIONS: dict[str, str] = {
    "sd60": "Native 720x480 60 fps, near-lossless CRF 10, e100.",
    "hd60": "Native 1280x720 60 fps, near-lossless CRF 10, e100.",
    "fhd60": "Native 1920x1080 60 fps, near-lossless CRF 10, e100.",
    "2k60": "Native 2560x1440 60 fps, near-lossless CRF 10, e100.",
    "4k60": "Native 3840x2160 60 fps, near-lossless CRF 10, e100.",
    "8k60": "Native 7680x4320 60 fps, near-lossless CRF 10, e100.",
}

# Keep existing command lines usable while making the resolution presets the
# only canonical choices shown by the GTK selector. These aliases intentionally
# inherit the new native/e100/near-lossless defaults instead of preserving the
# old undersampled or 30 fps behaviour under a misleading profile name. The
# old name that explicitly promised losslessness keeps that one stronger
# guarantee as well.
PROFILE_ALIASES: dict[str, str] = {
    "preview": "sd60",
    "fullhd": "fhd60",
    "1080p": "fhd60",
    "4k-e150": "4k60",
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

# CLI accepts aliases for existing scripts; GUI and profile documentation use
# the canonical six-tier list.
PROFILE_CHOICES = CANONICAL_PROFILE_CHOICES + tuple(PROFILE_ALIASES)
