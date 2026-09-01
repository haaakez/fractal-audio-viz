#!/usr/bin/env python3
"""Small Tkinter launcher for the command-line renderer.

Tkinter is part of Python's standard library and is available on Windows and
Linux. The renderer still runs in a child process, so the window stays
responsive and the CLI remains the source of truth for reproducible renders.
"""

from __future__ import annotations

import queue
import os
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover - depends on the Python distribution
    tk = None  # type: ignore[assignment]
    filedialog = messagebox = ttk = None  # type: ignore[assignment]

from deep_zoom_points import FORMULA_POINT_CATALOGUES, FORMULA_POINTS_BY_SLUG
from profiles import PROFILE_CHOICES, PROFILE_DEFAULTS


ROOT = Path(__file__).resolve().parent
VISUALIZER = ROOT / "visualizer.py"
OUTPUT_QUEUE_LIMIT = 2048
MAX_LOG_LINES = 20_000
MAX_LOG_LINE_CHARS = 16_384


def _read_bounded_line(stream: object, limit: int) -> str | None:
    """Read renderer output without buffering an unterminated line forever."""

    readline = getattr(stream, "readline")
    chunk = readline(limit + 1)
    if not chunk:
        return None
    truncated = len(chunk) > limit
    if truncated and not chunk.endswith("\n"):
        while True:
            remainder = readline(limit + 1)
            if not remainder or remainder.endswith("\n"):
                break
    if truncated:
        chunk = chunk[:limit] + "… [line truncated]\n"
    return str(chunk)


class RenderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Fractal Audio Viz")
        self.root.minsize(720, 560)
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str] = queue.Queue(maxsize=OUTPUT_QUEUE_LIMIT)

        self.audio = tk.StringVar(value=str(ROOT / "song.mp3"))
        self.output = tk.StringVar(value=str(ROOT / "fractal_viz.mp4"))
        self.profile = tk.StringVar(value="preview")
        self.formula = tk.StringVar(value="mandelbrot")
        self.point = tk.StringVar(value="")
        self.random_point = tk.BooleanVar(value=False)
        self.random_seed = tk.StringVar(value="")
        self.julia_c = tk.StringVar(value="-0.8,0.156")
        self.x_center = tk.StringVar(value="")
        self.y_center = tk.StringVar(value="")
        self.base_zoom = tk.StringVar(value="1.0")
        self.max_zoom = tk.StringVar(value="1e24")
        self.width = tk.StringVar(value="960")
        self.height = tk.StringVar(value="540")
        self.fps = tk.StringVar(value="24")
        self.sample_rate = tk.StringVar(value="44100")
        self.separation = tk.StringVar(value="auto")
        self.render_scale = tk.StringVar(value="1.0")
        self.fractal_scale = tk.StringVar(value="0.5")
        self.quality = tk.StringVar(value="draft")
        self.keyframe_factor = tk.StringVar(value="4.0")
        self.keyframe_mode = tk.StringVar(value="atlas")
        self.allow_underspecified_center = tk.BooleanVar(value=False)
        self.iteration_base = tk.StringVar(value="384")
        self.iterations_per_decade = tk.StringVar(value="500")
        self.iteration_cap = tk.StringVar(value="100000")
        self.zoom_punch = tk.StringVar(value="3.0")
        self.zoom_speed = tk.StringVar(value="-0.04")
        self.attack = tk.StringVar(value="0.025")
        self.release = tk.StringVar(value="0.12")
        self.series_order = tk.StringVar(value="3")
        self.series_block = tk.StringVar(value="256")
        self.renderer = tk.StringVar(value="auto")
        self.native_threads = tk.StringVar(value="0")
        self.native_backend = tk.StringVar(value="auto")
        self.video_preset = tk.StringVar(value="ultrafast")
        self.codec = tk.StringVar(value="auto")
        self.crf = tk.StringVar(value="18")
        self.resample = tk.StringVar(value="bilinear")
        self.encoder_threads = tk.StringVar(value="0")
        self.cache = tk.StringVar(value=str(ROOT / "cache"))
        self.cache_limit_mb = tk.StringVar(value="0")
        self.durable_cache = tk.BooleanVar(value=False)
        self.manifest = tk.StringVar(value="")
        self.no_manifest = tk.BooleanVar(value=False)
        self.palette = tk.StringVar(value="aurora")
        self.palette_file = tk.StringVar(value="")
        self.beat_strength = tk.DoubleVar(value=0.0)
        self.glow = tk.DoubleVar(value=0.0)
        self.motion_blur = tk.DoubleVar(value=0.0)
        self.beat_strength_value = tk.StringVar(value="0.00")
        self.glow_value = tk.StringVar(value="0.00")
        self.motion_blur_value = tk.StringVar(value="0.00")

        self._build_widgets()
        self.profile.trace_add("write", self._profile_changed)
        self.formula.trace_add("write", self._formula_changed)
        self.point.trace_add("write", self._point_changed)
        self._profile_changed()
        self._formula_changed()
        self.root.after(100, self._drain_output)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_widgets(self) -> None:
        self.root.minsize(780, 640)
        shell = ttk.Frame(self.root)
        shell.pack(fill="both", expand=True)
        shell.rowconfigure(0, weight=1)
        shell.columnconfigure(0, weight=1)

        canvas = tk.Canvas(shell, highlightthickness=0)
        scrollbar = ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        root_frame = ttk.Frame(canvas, padding=12)
        root_frame.columnconfigure(0, weight=1)
        window_id = canvas.create_window((0, 0), window=root_frame, anchor="nw")

        def update_scroll_region(_event: object = None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def resize_content(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=max(event.width, root_frame.winfo_reqwidth()))

        root_frame.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", resize_content)

        def scroll(event: tk.Event) -> None:
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(int(-event.delta / 120), "units")

        canvas.bind_all("<MouseWheel>", scroll)
        canvas.bind_all("<Button-4>", scroll)
        canvas.bind_all("<Button-5>", scroll)

        render_frame = ttk.LabelFrame(root_frame, text="Render")
        render_frame.grid(row=0, column=0, sticky="ew")
        render_frame.columnconfigure(1, weight=1)
        row = 0
        row = self._path_row(render_frame, row, "Audio", self.audio, [
            ("Audio files", "*.mp3 *.wav *.flac *.ogg"),
            ("All files", "*"),
        ])
        row = self._path_row(render_frame, row, "Output", self.output, [
            ("MP4", "*.mp4"),
            ("All files", "*"),
        ], save=True)
        row = self._combo_row(render_frame, row, "Profile", self.profile, PROFILE_CHOICES)
        row = self._combo_row(
            render_frame,
            row,
            "Formula",
            self.formula,
            tuple(FORMULA_POINT_CATALOGUES),
        )

        ttk.Label(render_frame, text="Point").grid(row=row, column=0, sticky="w", pady=4)
        point_frame = ttk.Frame(render_frame)
        point_frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        point_frame.columnconfigure(0, weight=1)
        self.point_combo = ttk.Combobox(
            point_frame,
            textvariable=self.point,
            values=("", "random"),
        )
        self.point_combo.grid(row=0, column=0, sticky="ew")
        self.point_combo.bind("<<ComboboxSelected>>", self._point_changed)
        ttk.Checkbutton(
            point_frame,
            text="Random catalogue point",
            variable=self.random_point,
        ).grid(row=0, column=1, padx=(8, 0))
        row += 1
        row = self._entry_row(render_frame, row, "Julia c", self.julia_c, "REAL,IMAG")
        row = self._entry_row(render_frame, row, "Max zoom", self.max_zoom, "for example 1e150")

        dimensions = ttk.Frame(render_frame)
        dimensions.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        for index in range(6):
            dimensions.columnconfigure(index, weight=1 if index in {1, 3, 5} else 0)
        ttk.Label(render_frame, text="Video").grid(row=row, column=0, sticky="w", pady=4)
        for index, (label, variable) in enumerate(
            (("W", self.width), ("H", self.height), ("FPS", self.fps))
        ):
            ttk.Label(dimensions, text=label).grid(row=0, column=index * 2, padx=(0, 3))
            ttk.Entry(dimensions, textvariable=variable, width=8).grid(
                row=0, column=index * 2 + 1, sticky="ew", padx=(0, 12)
            )
        row += 1
        row = self._combo_row(
            render_frame,
            row,
            "Palette",
            self.palette,
            ("aurora", "fire", "ocean", "neon", "sunset", "mono"),
        )
        row = self._path_row(
            render_frame,
            row,
            "Palette file",
            self.palette_file,
            [("Palette text", "*.txt"), ("All files", "*")],
        )

        effects = ttk.LabelFrame(root_frame, text="Audio and effects")
        effects.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        effects.columnconfigure(1, weight=1)
        row = 0
        row = self._slider_row(
            effects, row, "Beat strength", self.beat_strength, self.beat_strength_value,
            3.0, "onset contribution; 0 disables it",
        )
        row = self._slider_row(
            effects, row, "Glow", self.glow, self.glow_value,
            1.0, "bloom amount; adds compositor work",
        )
        self._slider_row(
            effects, row, "Motion blur", self.motion_blur, self.motion_blur_value,
            0.99, "blend with the previous frame",
        )

        technical = ttk.LabelFrame(root_frame, text="Technical options")
        technical.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        technical.columnconfigure(1, weight=1)
        row = 0
        row = self._technical_row(technical, row, "Sample rate", self.sample_rate, "Hz")
        row = self._technical_row(
            technical, row, "Separation", self.separation, "audio stem strategy",
            ("auto", "demucs", "spectral", "none"),
        )
        row = self._technical_row(technical, row, "Render scale", self.render_scale, "keyframe resolution multiplier")
        row = self._technical_row(technical, row, "Fractal scale", self.fractal_scale, "minimum source resolution multiplier")
        row = self._technical_row(
            technical, row, "Quality", self.quality, "draft / balanced / quality / extreme",
            ("draft", "balanced", "quality", "extreme"),
        )
        row = self._technical_row(technical, row, "Keyframe factor", self.keyframe_factor, "maximum zoom jump between keyframes")
        row = self._technical_row(
            technical, row, "Keyframe mode", self.keyframe_mode, "atlas reuses nested tiles",
            ("atlas", "legacy"),
        )
        row = self._technical_row(technical, row, "Random seed", self.random_seed, "reproducible catalogue selection")
        row = self._technical_row(technical, row, "X centre", self.x_center, "paired with Y centre; conflicts with Point")
        row = self._technical_row(technical, row, "Y centre", self.y_center, "paired with X centre; conflicts with Point")
        row = self._technical_row(technical, row, "Base zoom", self.base_zoom, "starting zoom")
        row = self._check_row(
            technical, row, "Allow underspecified centre", self.allow_underspecified_center,
            "exploratory deep render only",
        )
        row = self._technical_row(technical, row, "Iteration base", self.iteration_base, "minimum shallow iteration budget")
        row = self._technical_row(technical, row, "Iterations / decade", self.iterations_per_decade, "added per decimal zoom")
        row = self._technical_row(technical, row, "Iteration cap", self.iteration_cap, "maximum iteration budget")
        row = self._technical_row(technical, row, "Zoom punch", self.zoom_punch, "loudness contrast")
        row = self._technical_row(technical, row, "Zoom speed", self.zoom_speed, "quiet-time log zoom velocity")
        row = self._technical_row(technical, row, "Attack", self.attack, "audio envelope seconds")
        row = self._technical_row(technical, row, "Release", self.release, "audio envelope seconds")
        row = self._technical_row(technical, row, "Series order", self.series_order, "native BLA polynomial degree")
        row = self._technical_row(technical, row, "Series block", self.series_block, "native BLA block length")
        row = self._technical_row(
            technical, row, "Renderer", self.renderer, "auto / native / python",
            ("auto", "native", "python"),
        )
        row = self._technical_row(technical, row, "Native threads", self.native_threads, "0 uses the runtime default")
        row = self._technical_row(
            technical, row, "Native backend", self.native_backend, "hardware field backend",
            ("auto", "scalar", "avx2", "opencl"),
        )
        row = self._technical_row(
            technical, row, "Video preset", self.video_preset, "FFmpeg speed / size trade-off",
            ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"),
        )
        row = self._technical_row(technical, row, "Codec", self.codec, "FFmpeg encoder name or auto")
        row = self._technical_row(technical, row, "CRF", self.crf, "encoder quality, 0 to 51")
        row = self._technical_row(
            technical, row, "Resample", self.resample, "crop filter",
            ("bilinear", "lanczos"),
        )
        row = self._technical_row(technical, row, "Encoder threads", self.encoder_threads, "0 lets FFmpeg choose")
        row = self._technical_row(technical, row, "Cache limit MB", self.cache_limit_mb, "0 means unlimited")
        row = self._check_row(
            technical, row, "Durable cache", self.durable_cache,
            "fsync each tile; safer but slower",
        )
        row = self._path_row(
            technical, row, "Manifest", self.manifest,
            [("JSON", "*.json"), ("All files", "*")], save=True,
        )
        self._check_row(
            technical, row, "Disable manifest", self.no_manifest,
            "do not write the automatic JSON sidecar",
        )

        buttons = ttk.Frame(root_frame)
        buttons.grid(row=3, column=0, sticky="ew", pady=(10, 6))
        self.start_button = ttk.Button(buttons, text="Render", command=self._start)
        self.start_button.pack(side="left")
        self.estimate_button = ttk.Button(buttons, text="Estimate", command=self._estimate)
        self.estimate_button.pack(side="left", padx=6)
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left")

        ttk.Label(root_frame, text="Renderer output").grid(row=4, column=0, sticky="w", pady=(4, 0))
        self.log = tk.Text(
            root_frame,
            height=16,
            wrap="word",
            state="disabled",
            background="#101218",
            foreground="#e8eaf0",
        )
        self.log.grid(row=5, column=0, sticky="nsew", pady=(4, 0))

    def _combo_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            parent, textvariable=variable, values=values, state="readonly"
        ).grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        return row + 1

    def _slider_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.DoubleVar,
        display: tk.StringVar,
        maximum: float,
        hint: str,
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=5)
        ttk.Scale(
            parent,
            from_=0.0,
            to=maximum,
            variable=variable,
            command=lambda value: display.set(f"{float(value):.2f}"),
        ).grid(row=row, column=1, sticky="ew", padx=(4, 8), pady=5)
        ttk.Label(parent, textvariable=display, width=6, anchor="e").grid(
            row=row, column=2, sticky="e", padx=(0, 8), pady=5
        )
        ttk.Label(parent, text=hint).grid(row=row, column=3, sticky="w", padx=(0, 8), pady=5)
        return row + 1

    def _technical_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        hint: str,
        values: tuple[str, ...] | None = None,
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=3)
        if values is None:
            widget: tk.Widget = ttk.Entry(parent, textvariable=variable)
        else:
            widget = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        widget.grid(row=row, column=1, sticky="ew", padx=4, pady=3)
        ttk.Label(parent, text=hint).grid(row=row, column=2, sticky="w", padx=(8, 8), pady=3)
        return row + 1

    def _check_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.BooleanVar,
        hint: str,
    ) -> int:
        ttk.Checkbutton(parent, text=label, variable=variable).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=3
        )
        ttk.Label(parent, text=hint).grid(row=row, column=2, sticky="w", padx=(8, 8), pady=3)
        return row + 1

    def _entry_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, hint: str) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        entry.insert(0, "") if not variable.get() else None
        entry.configure(width=max(20, len(hint)))
        return row + 1

    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        filetypes: list[tuple[str, str]],
        save: bool = False,
        directory: bool = False,
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)
        def browse() -> None:
            if directory:
                chosen = filedialog.askdirectory()
            elif save:
                chosen = filedialog.asksaveasfilename(filetypes=filetypes)
            else:
                chosen = filedialog.askopenfilename(filetypes=filetypes)
            if chosen:
                variable.set(chosen)
        ttk.Button(parent, text="Browse", command=browse).grid(row=row, column=2, padx=(6, 0), pady=4)
        return row + 1

    def _formula_changed(self, *_: object) -> None:
        formula = self.formula.get()
        points = FORMULA_POINT_CATALOGUES.get(formula, ())
        values = ("", "random", *(point.slug for point in points))
        self.point_combo.configure(values=values)

        current = self.point.get().strip()
        known_values = {value.casefold() for value in values if value}
        if current and "," not in current and current.casefold() not in known_values:
            # A slug belongs to one formula.  Exact coordinate pairs remain
            # editable when the user changes formula.
            self.point.set("")
        self._point_changed()

    def _point_changed(self, *_: object) -> None:
        preset = FORMULA_POINTS_BY_SLUG.get(self.formula.get(), {}).get(
            self.point.get().strip().casefold()
        )
        if preset is not None and preset.julia_c is not None:
            self.julia_c.set(",".join(preset.julia_c))

    def _profile_changed(self, *_: object) -> None:
        values = PROFILE_DEFAULTS.get(self.profile.get(), {})
        for key, variable in (
            ("width", self.width),
            ("height", self.height),
            ("fps", self.fps),
            ("max_zoom", self.max_zoom),
            ("separation", self.separation),
            ("fractal_scale", self.fractal_scale),
            ("quality", self.quality),
            ("keyframe_factor", self.keyframe_factor),
            ("video_preset", self.video_preset),
        ):
            if key in values:
                variable.set(str(values[key]))
        self.beat_strength.set(float(values.get("beat_strength", 0.0)))
        self.beat_strength_value.set(f"{self.beat_strength.get():.2f}")
        self.glow_value.set(f"{self.glow.get():.2f}")
        self.motion_blur_value.set(f"{self.motion_blur.get():.2f}")

    def _command(self, estimate: bool = False) -> list[str]:
        point_spec = self.point.get().strip()
        has_point = self.random_point.get() or bool(point_spec)
        has_x = bool(self.x_center.get().strip())
        has_y = bool(self.y_center.get().strip())
        if has_point and (has_x or has_y):
            raise ValueError("Point/random point cannot be combined with X/Y centre")
        if has_x != has_y:
            raise ValueError("X and Y centre must be supplied together")
        audio_path = Path(self.audio.get()).expanduser().resolve()
        # Preserve the final component so the CLI can reject an output
        # symlink safely instead of resolving it into the target file.
        output_path = Path(self.output.get()).expanduser().absolute()
        command = [sys.executable, "-u", str(VISUALIZER), str(audio_path)]
        command.extend(["--output", str(output_path), "--profile", self.profile.get()])
        formula = self.formula.get()
        julia_preset = FORMULA_POINTS_BY_SLUG.get(formula, {}).get(point_spec.casefold())
        command.extend([
            "--formula", formula,
            "--max-zoom", self.max_zoom.get(),
            "--width", self.width.get(), "--height", self.height.get(), "--fps", self.fps.get(),
            "--sample-rate", self.sample_rate.get(),
            "--separation", self.separation.get(),
            "--render-scale", self.render_scale.get(),
            "--fractal-scale", self.fractal_scale.get(),
            "--quality", self.quality.get(),
            "--keyframe-factor", self.keyframe_factor.get(),
            "--keyframe-mode", self.keyframe_mode.get(),
            "--base-zoom", self.base_zoom.get(),
            "--iteration-base", self.iteration_base.get(),
            "--iterations-per-decade", self.iterations_per_decade.get(),
            "--iteration-cap", self.iteration_cap.get(),
            "--zoom-punch", self.zoom_punch.get(),
            "--zoom-speed", self.zoom_speed.get(),
            "--palette", self.palette.get(),
            "--beat-strength", str(self.beat_strength.get()),
            "--attack", self.attack.get(),
            "--release", self.release.get(),
            "--series-order", self.series_order.get(),
            "--series-block", self.series_block.get(),
            "--renderer", self.renderer.get(),
            "--native-threads", self.native_threads.get(),
            "--native-backend", self.native_backend.get(),
            "--video-preset", self.video_preset.get(),
            "--codec", self.codec.get(),
            "--crf", self.crf.get(),
            "--resample", self.resample.get(),
            "--glow", str(self.glow.get()), "--motion-blur", str(self.motion_blur.get()),
            "--encoder-threads", self.encoder_threads.get(),
            "--cache-limit-mb", self.cache_limit_mb.get(),
        ])
        if formula == "julia":
            # Resolve the preset at command-build time too.  This covers a
            # slug typed into the editable combobox, even when Tk has not
            # emitted a ComboboxSelected event.  Random selection must not
            # inherit a stale preset constant from the editable combobox.
            if not self.random_point.get() and point_spec.casefold() != "random" and (
                julia_preset is not None and julia_preset.julia_c is not None
            ):
                command.append("--julia-c=" + ",".join(julia_preset.julia_c))
            elif not self.random_point.get() and point_spec.casefold() != "random":
                # A value such as -0.8,0.156 is interpreted as an option by
                # argparse when it is passed as a separate argv item.
                command.append("--julia-c=" + self.julia_c.get().strip())
        if self.random_point.get():
            command.append("--random-point")
        elif point_spec:
            # The same applies to negative custom Mandelbrot coordinates.
            command.append("--point=" + point_spec)
        if self.random_seed.get().strip():
            command.extend(["--random-seed", self.random_seed.get().strip()])
        if self.x_center.get().strip():
            command.append("--x-center=" + self.x_center.get().strip())
        if self.y_center.get().strip():
            command.append("--y-center=" + self.y_center.get().strip())
        if self.allow_underspecified_center.get():
            command.append("--allow-underspecified-center")
        if self.cache.get().strip():
            command.extend([
                "--cache-dir",
                str(Path(self.cache.get().strip()).expanduser().resolve()),
            ])
        if self.durable_cache.get():
            command.append("--durable-cache")
        if self.manifest.get().strip():
            command.extend([
                "--manifest",
                str(Path(self.manifest.get().strip()).expanduser().absolute()),
            ])
        if self.no_manifest.get():
            command.append("--no-manifest")
        if self.palette_file.get().strip():
            command.extend([
                "--palette-file",
                str(Path(self.palette_file.get().strip()).expanduser().resolve()),
            ])
        if estimate:
            command.append("--estimate")
        return command

    @staticmethod
    def _process_options() -> dict[str, object]:
        if os.name == "nt":
            return {
                "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            }
        return {"start_new_session": True}

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        """Terminate the renderer and its FFmpeg descendants as one unit."""

        if process.poll() is not None:
            return
        if os.name == "nt":
            try:
                process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
            except (OSError, ValueError):
                try:
                    process.terminate()
                except OSError:
                    return
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                try:
                    process.terminate()
                except OSError:
                    return
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                try:
                    process.kill()
                except OSError:
                    return
            else:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    try:
                        process.kill()
                    except OSError:
                        return
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                return

    def _start(self) -> None:
        if self.process is not None:
            return
        if not Path(self.audio.get()).expanduser().resolve().is_file():
            messagebox.showerror("Audio file", "Choose an existing audio file first.")
            return
        try:
            command = self._command()
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **self._process_options(),
            )
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not start renderer", str(error))
            return
        self._set_running(True)
        self._append("Command: " + shlex.join(command) + "\n")
        threading.Thread(target=self._read_process, daemon=True).start()

    def _estimate(self) -> None:
        if self.process is not None:
            return
        if not Path(self.audio.get()).expanduser().resolve().is_file():
            messagebox.showerror("Audio file", "Choose an existing audio file first.")
            return
        try:
            command = self._command(estimate=True)
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **self._process_options(),
            )
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not start renderer", str(error))
            return
        self._set_running(True)
        self._append("Estimate command: " + shlex.join(command) + "\n")
        threading.Thread(target=self._read_process, daemon=True).start()

    def _read_process(self) -> None:
        process = self.process
        assert process is not None and process.stdout is not None
        while True:
            line = _read_bounded_line(process.stdout, MAX_LOG_LINE_CHARS)
            if line is None:
                break
            self._queue_output(line)
        code = process.wait()
        self._queue_output(f"\nProcess exited with status {code}.\n")
        self._queue_output("__FRACTAL_PROCESS_DONE__")

    def _queue_output(self, text: str) -> None:
        """Keep a stalled GUI from retaining unbounded renderer output."""

        if len(text) > MAX_LOG_LINE_CHARS:
            text = text[:MAX_LOG_LINE_CHARS] + "… [line truncated]\n"
        try:
            self.output_queue.put_nowait(text)
            return
        except queue.Full:
            pass
        # Dropping the oldest log line is preferable to blocking the reader:
        # a blocked GUI pipe can also stall FFmpeg and make Stop ineffective.
        try:
            self.output_queue.get_nowait()
        except queue.Empty:
            return
        try:
            self.output_queue.put_nowait(text)
        except queue.Full:
            pass

    def _drain_output(self) -> None:
        try:
            while True:
                line = self.output_queue.get_nowait()
                if line == "__FRACTAL_PROCESS_DONE__":
                    self.process = None
                    self._set_running(False)
                else:
                    self._append(line)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_output)

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        line_number = int(self.log.index("end-1c").split(".", 1)[0])
        if line_number > MAX_LOG_LINES:
            first_kept_line = line_number - MAX_LOG_LINES + 1
            self.log.delete("1.0", f"{first_kept_line}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.start_button.configure(state=state)
        self.estimate_button.configure(state=state)
        self.stop_button.configure(state="normal" if running else "disabled")

    def _stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self._terminate_process(self.process)
            self._append("Stopping renderer...\n")

    def _close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Stop render?", "A render is still running. Stop it and close?"):
                return
            self._terminate_process(self.process)
        self.root.destroy()


def main() -> None:
    if tk is None:
        raise SystemExit(
            "Tkinter is not available in this Python. Install python3-tk on Linux "
            "or use the Python installer that includes Tcl/Tk on Windows."
        )
    root = tk.Tk()
    RenderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
