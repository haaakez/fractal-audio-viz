# Product hardening plan

This plan tracks the reliability work needed before calling the renderer
production-ready.

## P0 — safety and correctness

- [x] Reject output/audio/manifest path collisions before analysis or encoding.
- [x] Validate render dimensions, iteration budgets, zoom controls, and audio
  feature arrays at public API boundaries.
- [x] Reject malformed or non-finite cache fields instead of treating them as
  valid rendered data.
- [x] Include the active field renderer and native backend in cache identity.
- [x] Make native ABI error messages thread-local and parse native coordinates
  completely.

## P1 — graceful operation

- [x] Make native-library probing continue past malformed/incompatible shared
  libraries.
- [x] Add timeouts and fallback behavior to optional Demucs separation.
- [x] Make preview generation atomic and reject input/output collisions.
- [x] Capture and surface richer FFmpeg diagnostics for every encoder path.
- [x] Make GUI termination kill the complete subprocess group on every
  supported platform.

## P2 — confidence and polish

- [ ] Add dependency-backed CI coverage for cache corruption, path safety,
  native concurrency, and preview failure cleanup.
- [x] Add a small end-to-end fixture render that verifies audio duration,
  frame count, pixel format, and manifest status.
- [x] Probe hardware encoders with a real short encode before selecting them
  automatically; retain deterministic software fallback.
- [x] Document reproducibility limits and the exact native/Python fallback
  matrix in the README.

The remaining P1/P2 items are intentionally separate from the initial safety
patches: they improve diagnostics and coverage without changing the numerical
rendering model.
