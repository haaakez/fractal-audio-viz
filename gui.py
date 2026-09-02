#!/usr/bin/env python3
"""GTK3 launcher for the reproducible command-line renderer.

The GUI is deliberately a thin launcher: it owns the controls, process
lifecycle, and progress log, while ``visualizer.py`` remains the single source
of rendering behaviour. It uses GTK's system theme and native scrolled
container/expander so the compact default view stays stable when technical
controls are opened.
"""

from __future__ import annotations

import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path

import visualizer

try:
    import gi

    gi.require_version("Gtk", "3.0")
    from gi.repository import GLib, Gtk
except (ImportError, ValueError) as error:  # pragma: no cover - environment dependent
    GLib = Gtk = None  # type: ignore[assignment]
    GTK_IMPORT_ERROR: Exception | None = error
else:
    GTK_IMPORT_ERROR = None

from deep_zoom_points import FORMULA_POINT_CATALOGUES, FORMULA_POINTS_BY_SLUG
from profiles import PROFILE_CHOICES, PROFILE_DEFAULTS


ROOT = Path(__file__).resolve().parent
VISUALIZER = ROOT / "visualizer.py"
OUTPUT_QUEUE_LIMIT = 2048
MAX_LOG_LINES = 20_000
MAX_LOG_LINE_CHARS = 16_384

_PROGRESS_PERCENT_RE = re.compile(r"(?<![\w.])(?P<percent>\d+(?:\.\d+)?)\s*%")
_PROGRESS_FRAME_RE = re.compile(
    r"\b(?:frame|frames)\s+(?P<current>\d+)\s*(?:/|of)\s*(?P<total>\d+)\b",
    re.IGNORECASE,
)


def _progress_from_output_line(line: str) -> tuple[float | None, str | None]:
    """Extract a determinate renderer progress value from one log line."""

    text = line.strip()
    percent_match = _PROGRESS_PERCENT_RE.search(text)
    if percent_match is not None:
        percent = float(percent_match.group("percent"))
        if percent <= 100.0:
            return max(0.0, percent / 100.0), f"{percent:.1f}%"
    frame_match = _PROGRESS_FRAME_RE.search(text)
    if frame_match is not None:
        current = int(frame_match.group("current"))
        total = int(frame_match.group("total"))
        if total > 0:
            fraction = min(1.0, max(0.0, current / total))
            return fraction, f"Frame {current:,} / {total:,} ({fraction * 100.0:.1f}%)"
    return None, None


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


if Gtk is not None:

    class RenderApp:
        """Own the GTK window and launch the shared CLI in a child process."""

        def __init__(self, application: Gtk.Application) -> None:
            self.application = application
            self.window = Gtk.ApplicationWindow(application=application)
            self.window.set_title("Fractal Audio Viz")
            self.window.set_default_size(980, 760)
            self.window.set_size_request(760, 560)
            self.window.connect("delete-event", self._on_delete_event)
            self.process: subprocess.Popen[str] | None = None
            self.output_queue: queue.Queue[str] = queue.Queue(maxsize=OUTPUT_QUEUE_LIMIT)
            self._log_follow_tail = True
            self._log_scroll_pending = False
            self._log_programmatic_scroll = False
            self._progress_determinate = False
            self._process_exit_code: int | None = None

            self._build_widgets()
            self.profile.connect("changed", self._profile_changed)
            self.formula.connect("changed", self._formula_changed)
            self.point_combo.connect("changed", self._point_changed)
            self.palette.connect("changed", self._palette_changed)
            self.palette_file.connect("changed", self._palette_changed)
            self._profile_changed()
            self._formula_changed()
            self._palette_changed()
            GLib.timeout_add(100, self._drain_output)
            self.window.show_all()
            self.technical_expander.set_expanded(False)
            self._set_technical_child_visible(False)

        @staticmethod
        def _entry(text: str = "") -> Gtk.Entry:
            entry = Gtk.Entry()
            entry.set_text(text)
            entry.set_hexpand(True)
            return entry

        @staticmethod
        def _combo(values: tuple[str, ...], selected: str) -> Gtk.ComboBoxText:
            combo = Gtk.ComboBoxText()
            for value in values:
                combo.append(value, value)
            combo.set_active_id(selected)
            combo.set_hexpand(True)
            return combo

        @staticmethod
        def _grid() -> Gtk.Grid:
            grid = Gtk.Grid()
            grid.set_column_spacing(12)
            grid.set_row_spacing(8)
            grid.set_border_width(12)
            grid.set_hexpand(True)
            return grid

        @staticmethod
        def _label(text: str) -> Gtk.Label:
            label = Gtk.Label(label=text)
            label.set_halign(Gtk.Align.START)
            return label

        def _add_widget_row(
            self,
            grid: Gtk.Grid,
            row: int,
            label: str,
            widget: Gtk.Widget,
            hint: str = "",
        ) -> int:
            grid.attach(self._label(label), 0, row, 1, 1)
            widget.set_hexpand(True)
            grid.attach(widget, 1, row, 1, 1)
            if hint:
                hint_label = self._label(hint)
                hint_label.get_style_context().add_class("dim-label")
                grid.attach(hint_label, 2, row, 1, 1)
            return row + 1

        def _add_path_row(
            self,
            grid: Gtk.Grid,
            row: int,
            label: str,
            entry: Gtk.Entry,
            *,
            action: Gtk.FileChooserAction,
            patterns: tuple[str, ...],
            save: bool = False,
        ) -> int:
            grid.attach(self._label(label), 0, row, 1, 1)
            grid.attach(entry, 1, row, 1, 1)
            button = Gtk.Button(label="Browse")
            button.connect(
                "clicked",
                self._choose_path,
                entry,
                action,
                patterns,
                save,
            )
            grid.attach(button, 2, row, 1, 1)
            return row + 1

        @staticmethod
        def _frame(title: str, child: Gtk.Widget) -> Gtk.Frame:
            frame = Gtk.Frame(label=title)
            frame.add(child)
            frame.set_hexpand(True)
            return frame

        def _build_widgets(self) -> None:
            self.audio = self._entry(str(ROOT / "song.mp3"))
            self.output = self._entry(str(ROOT / "fractal_viz.mp4"))
            self.profile = self._combo(PROFILE_CHOICES, "preview")
            self.formula = self._combo(tuple(FORMULA_POINT_CATALOGUES), "mandelbrot")
            self.point_combo = Gtk.ComboBoxText.new_with_entry()
            self.point_entry = self.point_combo.get_child()
            assert isinstance(self.point_entry, Gtk.Entry)
            self.point_entry.set_placeholder_text("catalogue slug or REAL,IMAG")
            self.point_combo.set_hexpand(True)
            self.random_point = Gtk.CheckButton(label="Random catalogue point")
            self.julia_c = self._entry("-0.8,0.156")
            self.max_zoom = self._entry("1e24")
            self.width = self._entry("960")
            self.height = self._entry("540")
            self.fps = self._entry("24")
            self.palette = self._combo(
                visualizer.PALETTE_CHOICES,
                "aurora",
            )
            self.palette_file = self._entry()
            self.palette_preview = Gtk.DrawingArea()
            self.palette_preview.set_hexpand(True)
            self.palette_preview.set_size_request(-1, 54)
            self.palette_preview.set_tooltip_text(
                "live preview of the selected theme; the lower strip is the interior colour"
            )
            self.palette_preview.connect("draw", self._draw_palette_preview)

            render_grid = self._grid()
            row = 0
            row = self._add_path_row(
                render_grid,
                row,
                "Audio",
                self.audio,
                action=Gtk.FileChooserAction.OPEN,
                patterns=("*.mp3", "*.wav", "*.flac", "*.ogg"),
            )
            row = self._add_path_row(
                render_grid,
                row,
                "Output",
                self.output,
                action=Gtk.FileChooserAction.SAVE,
                patterns=("*.mp4", "*.mkv", "*.webm"),
                save=True,
            )
            row = self._add_widget_row(render_grid, row, "Profile", self.profile)
            row = self._add_widget_row(render_grid, row, "Formula", self.formula)
            point_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            point_box.set_hexpand(True)
            point_box.pack_start(self.point_combo, True, True, 0)
            point_box.pack_start(self.random_point, False, False, 0)
            row = self._add_widget_row(render_grid, row, "Point", point_box)
            row = self._add_widget_row(render_grid, row, "Julia c", self.julia_c, "REAL,IMAG")
            row = self._add_widget_row(render_grid, row, "Max zoom", self.max_zoom, "for example 1e150")

            dimensions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            for label, entry in (
                ("W", self.width),
                ("H", self.height),
                ("FPS", self.fps),
            ):
                dimensions.pack_start(self._label(label), False, False, 0)
                entry.set_width_chars(7)
                dimensions.pack_start(entry, True, True, 0)
            row = self._add_widget_row(render_grid, row, "Video", dimensions)
            row = self._add_widget_row(render_grid, row, "Palette", self.palette)
            row = self._add_widget_row(
                render_grid,
                row,
                "Theme preview",
                self.palette_preview,
            )
            self._add_path_row(
                render_grid,
                row,
                "Palette file",
                self.palette_file,
                action=Gtk.FileChooserAction.OPEN,
                patterns=("*.txt", "*.kfp", "*.kfr"),
            )

            self.beat_strength, beat_value = self._scale_row(0.0, 3.0)
            self.glow, glow_value = self._scale_row(0.0, 1.0)
            self.motion_blur, blur_value = self._scale_row(0.0, 0.99)
            effects_grid = self._grid()
            effects_grid.set_row_spacing(6)
            effects_grid.attach(self._label("Beat strength"), 0, 0, 1, 1)
            effects_grid.attach(self.beat_strength, 1, 0, 1, 1)
            effects_grid.attach(beat_value, 2, 0, 1, 1)
            effects_grid.attach(self._label("onset contribution; 0 disables it"), 3, 0, 1, 1)
            effects_grid.attach(self._label("Glow"), 0, 1, 1, 1)
            effects_grid.attach(self.glow, 1, 1, 1, 1)
            effects_grid.attach(glow_value, 2, 1, 1, 1)
            effects_grid.attach(self._label("bloom amount; adds compositor work"), 3, 1, 1, 1)
            effects_grid.attach(self._label("Motion blur"), 0, 2, 1, 1)
            effects_grid.attach(self.motion_blur, 1, 2, 1, 1)
            effects_grid.attach(blur_value, 2, 2, 1, 1)
            effects_grid.attach(self._label("blend with the previous frame"), 3, 2, 1, 1)

            self.sample_rate = self._entry("44100")
            self.separation = self._combo(("auto", "demucs", "spectral", "none"), "auto")
            self.render_scale = self._entry("1.0")
            self.fractal_scale = self._entry("0.5")
            self.quality = self._combo(("draft", "balanced", "quality", "extreme"), "draft")
            self.keyframe_factor = self._entry("4.0")
            self.keyframe_mode = self._combo(("atlas", "legacy"), "atlas")
            self.random_seed = self._entry()
            self.x_center = self._entry()
            self.y_center = self._entry()
            self.base_zoom = self._entry("1.0")
            self.allow_underspecified_center = Gtk.CheckButton(label="Allow underspecified centre")
            self.iteration_base = self._entry("384")
            self.iterations_per_decade = self._entry("500")
            self.iteration_cap = self._entry("100000")
            self.zoom_punch = self._entry("3.0")
            self.zoom_speed = self._entry("-0.04")
            self.attack = self._entry("0.025")
            self.release = self._entry("0.12")
            self.series_order = self._entry("3")
            self.series_block = self._entry("256")
            self.renderer = self._combo(("auto", "native", "python"), "auto")
            self.native_threads = self._entry("0")
            self.native_backend = self._combo(("auto", "scalar", "avx2", "opencl"), "auto")
            self.video_preset = self._combo(
                ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"),
                "ultrafast",
            )
            self.codec = self._entry("auto")
            self.crf = self._entry("18")
            self.resample = self._combo(("bilinear", "lanczos"), "bilinear")
            self.encoder_threads = self._entry("0")
            self.cache = self._entry(str(ROOT / "cache"))
            self.cache_limit_mb = self._entry("0")
            self.durable_cache = Gtk.CheckButton(label="Durable cache")
            self.manifest = self._entry()
            self.no_manifest = Gtk.CheckButton(label="Disable manifest")

            technical_grid = self._grid()
            technical_grid.set_row_spacing(6)
            technical_row = 0
            for label, widget, hint in (
                ("Sample rate", self.sample_rate, "Hz"),
                ("Separation", self.separation, "audio stem strategy"),
                ("Render scale", self.render_scale, "keyframe resolution multiplier"),
                ("Fractal scale", self.fractal_scale, "minimum source resolution multiplier"),
                ("Quality", self.quality, "draft / balanced / quality / extreme"),
                ("Keyframe factor", self.keyframe_factor, "maximum zoom jump between keyframes"),
                ("Keyframe mode", self.keyframe_mode, "atlas reuses nested tiles"),
                ("Random seed", self.random_seed, "reproducible catalogue selection"),
                ("X centre", self.x_center, "paired with Y centre; conflicts with Point"),
                ("Y centre", self.y_center, "paired with X centre; conflicts with Point"),
                ("Base zoom", self.base_zoom, "starting zoom"),
            ):
                technical_row = self._add_widget_row(technical_grid, technical_row, label, widget, hint)
            technical_grid.attach(self.allow_underspecified_center, 0, technical_row, 2, 1)
            technical_grid.attach(self._label("exploratory deep render only"), 2, technical_row, 1, 1)
            technical_row += 1
            for label, widget, hint in (
                ("Iteration base", self.iteration_base, "minimum shallow iteration budget"),
                ("Iterations / decade", self.iterations_per_decade, "added per decimal zoom"),
                ("Iteration cap", self.iteration_cap, "maximum iteration budget"),
                ("Zoom punch", self.zoom_punch, "loudness contrast"),
                ("Zoom speed", self.zoom_speed, "quiet-time log zoom velocity"),
                ("Attack", self.attack, "audio envelope seconds"),
                ("Release", self.release, "audio envelope seconds"),
                ("Series order", self.series_order, "native BLA polynomial degree"),
                ("Series block", self.series_block, "native BLA block length"),
                ("Renderer", self.renderer, "auto / native / python"),
                ("Native threads", self.native_threads, "0 uses the runtime default"),
                ("Native backend", self.native_backend, "hardware field backend"),
                ("Video preset", self.video_preset, "FFmpeg speed / size trade-off"),
                ("Codec", self.codec, "FFmpeg encoder name or auto"),
                ("CRF", self.crf, "encoder quality, 0 to 51"),
                ("Resample", self.resample, "crop filter"),
                ("Encoder threads", self.encoder_threads, "0 lets FFmpeg choose"),
                ("Cache limit MB", self.cache_limit_mb, "0 means unlimited"),
            ):
                technical_row = self._add_widget_row(technical_grid, technical_row, label, widget, hint)
            technical_grid.attach(self.durable_cache, 0, technical_row, 2, 1)
            technical_grid.attach(self._label("fsync each tile; safer but slower"), 2, technical_row, 1, 1)
            technical_row += 1
            technical_row = self._add_path_row(
                technical_grid,
                technical_row,
                "Manifest",
                self.manifest,
                action=Gtk.FileChooserAction.SAVE,
                patterns=("*.json",),
                save=True,
            )
            technical_grid.attach(self.no_manifest, 0, technical_row, 2, 1)
            technical_grid.attach(self._label("do not write the automatic JSON sidecar"), 2, technical_row, 1, 1)

            self.technical_expander = Gtk.Expander(label="Technical options")
            self.technical_expander.set_hexpand(True)
            self.technical_frame = self._frame("Technical options", technical_grid)
            self.technical_expander.connect("notify::expanded", self._technical_expanded_changed)

            self.start_button = Gtk.Button(label="Render")
            self.start_button.connect("clicked", self._start)
            self.estimate_button = Gtk.Button(label="Estimate")
            self.estimate_button.connect("clicked", self._estimate)
            self.stop_button = Gtk.Button(label="Stop")
            self.stop_button.connect("clicked", self._stop)
            self.stop_button.set_sensitive(False)
            buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            buttons.pack_start(self.start_button, False, False, 0)
            buttons.pack_start(self.estimate_button, False, False, 0)
            buttons.pack_start(self.stop_button, False, False, 0)

            self.log_buffer = Gtk.TextBuffer()
            self.log = Gtk.TextView(buffer=self.log_buffer)
            self.log.set_editable(False)
            self.log.set_cursor_visible(False)
            self.log.set_monospace(True)
            self.log.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
            self.log.set_vexpand(True)
            self.log.set_size_request(-1, 240)
            self.log_scroll = Gtk.ScrolledWindow()
            self.log_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
            self.log_scroll.set_min_content_height(240)
            self.log_scroll.set_vexpand(True)
            self.log_scroll.add(self.log)
            self.log_adjustment = self.log_scroll.get_vadjustment()
            self.log_adjustment.connect("value-changed", self._log_adjustment_changed)

            self.progress = Gtk.ProgressBar()
            self.progress.set_hexpand(True)
            self.progress.set_show_text(True)
            self.progress.set_fraction(0.0)
            self.progress.set_text("Ready")

            content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
            content.set_border_width(12)
            content.set_hexpand(True)
            content.pack_start(self._frame("Render", render_grid), False, False, 0)
            content.pack_start(self._frame("Audio and effects", effects_grid), False, False, 0)
            content.pack_start(self.technical_expander, False, False, 0)
            content.pack_start(buttons, False, False, 0)
            content.pack_start(self._label("Progress"), False, False, 0)
            content.pack_start(self.progress, False, False, 0)
            content.pack_start(self._label("Renderer output"), False, False, 0)
            content.pack_start(self.log_scroll, True, True, 0)

            scrolled = Gtk.ScrolledWindow()
            scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
            scrolled.set_overlay_scrolling(False)
            scrolled.set_hexpand(True)
            scrolled.set_vexpand(True)
            scrolled.add(content)
            self.scroll_adjustment = scrolled.get_vadjustment()
            self.window.add(scrolled)

        @staticmethod
        def _scale_row(value: float, maximum: float) -> tuple[Gtk.Scale, Gtk.Label]:
            scale = Gtk.Scale.new_with_range(
                Gtk.Orientation.HORIZONTAL,
                0.0,
                maximum,
                maximum / 100.0,
            )
            scale.set_value(value)
            scale.set_digits(2)
            scale.set_hexpand(True)
            display = Gtk.Label(label=f"{value:.2f}")
            display.set_width_chars(6)
            display.set_xalign(1.0)
            scale.connect(
                "value-changed",
                lambda control: display.set_text(f"{control.get_value():.2f}"),
            )
            return scale, display

        def _technical_expanded_changed(self, expander: Gtk.Expander, _param: object) -> None:
            self._set_technical_child_visible(expander.get_expanded())
            if not expander.get_expanded():
                # GTK recalculates the adjustment after the expander's child
                # is removed from the allocation. Reset on idle so a collapse
                # from the bottom can never expose an empty top.
                GLib.idle_add(self._reset_scroll_position)

        def _set_technical_child_visible(self, visible: bool) -> None:
            if visible:
                if self.technical_frame.get_parent() is None:
                    self.technical_expander.add(self.technical_frame)
                self.technical_frame.show_all()
            else:
                if self.technical_frame.get_parent() is self.technical_expander:
                    self.technical_expander.remove(self.technical_frame)
                self.technical_frame.hide()

        def _reset_scroll_position(self) -> bool:
            self.scroll_adjustment.set_value(self.scroll_adjustment.get_lower())
            return False

        def _log_at_bottom(self) -> bool:
            upper = self.log_adjustment.get_upper()
            page_size = self.log_adjustment.get_page_size()
            value = self.log_adjustment.get_value()
            remaining = upper - page_size - value
            return remaining <= max(2.0, self.log_adjustment.get_step_increment())

        def _log_adjustment_changed(self, _adjustment: Gtk.Adjustment) -> None:
            if not self._log_programmatic_scroll:
                self._log_follow_tail = self._log_at_bottom()

        def _schedule_log_tail(self) -> None:
            if self._log_scroll_pending:
                return
            self._log_scroll_pending = True
            GLib.idle_add(self._scroll_log_to_tail)

        def _scroll_log_to_tail(self) -> bool:
            self._log_scroll_pending = False
            if not self._log_follow_tail:
                return False
            self._log_programmatic_scroll = True
            try:
                end = self.log_buffer.get_end_iter()
                self.log.scroll_to_iter(end, 0.0, False, 0.0, 1.0)
                lower = self.log_adjustment.get_lower()
                upper = self.log_adjustment.get_upper()
                page_size = self.log_adjustment.get_page_size()
                target = max(lower, upper - page_size)
                self.log_adjustment.set_value(target)
            finally:
                self._log_programmatic_scroll = False
            return False

        def _set_progress(self, fraction: float | None, text: str) -> None:
            if fraction is None:
                self._progress_determinate = False
                self.progress.set_fraction(0.0)
                self.progress.set_text(text)
                self.progress.pulse()
                return
            self._progress_determinate = True
            fraction = min(1.0, max(0.0, float(fraction)))
            self.progress.set_fraction(fraction)
            self.progress.set_text(text)

        def _update_progress_from_output(self, text: str) -> bool:
            found = False
            for line in text.splitlines():
                fraction, label = _progress_from_output_line(line)
                if fraction is not None and label is not None:
                    self._set_progress(fraction, label)
                    found = True
            return found

        def _choose_path(
            self,
            _button: Gtk.Button,
            entry: Gtk.Entry,
            action: Gtk.FileChooserAction,
            patterns: tuple[str, ...],
            save: bool,
        ) -> None:
            title = "Choose output file" if save else "Choose file"
            dialog = Gtk.FileChooserDialog(
                title=title,
                transient_for=self.window,
                action=action,
            )
            dialog.add_buttons(
                "Cancel",
                Gtk.ResponseType.CANCEL,
                "Select",
                Gtk.ResponseType.ACCEPT,
            )
            if save:
                dialog.set_do_overwrite_confirmation(True)
            file_filter = Gtk.FileFilter()
            file_filter.set_name("Supported files")
            for pattern in patterns:
                file_filter.add_pattern(pattern)
            dialog.add_filter(file_filter)
            existing = entry.get_text().strip()
            if existing:
                try:
                    dialog.set_filename(existing)
                except Exception:
                    pass
            response = dialog.run()
            if response == Gtk.ResponseType.ACCEPT:
                filename = dialog.get_filename()
                if filename:
                    entry.set_text(filename)
            dialog.destroy()

        @staticmethod
        def _combo_text(combo: Gtk.ComboBoxText) -> str:
            return combo.get_active_id() or combo.get_active_text() or ""

        def _point_text(self) -> str:
            return self.point_entry.get_text().strip()

        @staticmethod
        def _set_combo(combo: Gtk.ComboBoxText, value: object) -> None:
            combo.set_active_id(str(value))

        def _formula_changed(self, *_args: object) -> None:
            formula = self._combo_text(self.formula)
            current = self._point_text()
            self.point_combo.remove_all()
            self.point_combo.append("", "")
            self.point_combo.append("random", "random")
            points = FORMULA_POINT_CATALOGUES.get(formula, ())
            for point in points:
                self.point_combo.append(point.slug, point.slug)
            known_values = {
                value.casefold()
                for value in ("random", *(point.slug for point in points))
            }
            if current and "," not in current and current.casefold() not in known_values:
                current = ""
            self.point_entry.set_text(current)
            self._point_changed()

        def _point_changed(self, *_args: object) -> None:
            point = self._point_text().casefold()
            preset = FORMULA_POINTS_BY_SLUG.get(self._combo_text(self.formula), {}).get(point)
            if preset is not None and preset.julia_c is not None:
                self.julia_c.set_text(",".join(preset.julia_c))

        def _palette_changed(self, *_args: object) -> None:
            self.palette_preview.queue_draw()

        def _palette_preview_samples(self) -> tuple[tuple[int, int, int], ...]:
            name = self._combo_text(self.palette)
            file_text = self.palette_file.get_text().strip()
            palette_file = Path(file_text).expanduser() if file_text else None
            try:
                profile = visualizer._kfp_profile_for_selection(name, palette_file)
                if profile is not None:
                    lut = visualizer._kfp_palette_lut(profile, 128)
                    return tuple(tuple(int(channel) for channel in row) for row in lut)
                if palette_file is not None:
                    palette = visualizer._palette_from_file(palette_file, 128)
                    return tuple(
                        tuple(int(channel) for channel in row) for row in palette
                    )
                if name == "aurora":
                    # Aurora is a field-driven liquid gradient rather than a
                    # fixed stop list; these representative anchors keep the
                    # preview honest without running a render during repaint.
                    stops = (
                        (1, 4, 18),
                        (24, 54, 130),
                        (92, 188, 255),
                        (255, 255, 255),
                    )
                else:
                    stops = visualizer.BUILTIN_AURORA_ACCENTS.get(
                        name,
                        visualizer.BUILTIN_PALETTE_STOPS[name],
                    )
                samples = []
                for index in range(128):
                    position = index * (len(stops) - 1) / 127.0
                    left = min(len(stops) - 2, int(position))
                    fraction = position - left
                    samples.append(tuple(
                        int(round(
                            stops[left][channel] * (1.0 - fraction)
                            + stops[left + 1][channel] * fraction
                        ))
                        for channel in range(3)
                    ))
                return tuple(samples)
            except (OSError, KeyError, ValueError, TypeError):
                return ((32, 32, 32), (96, 96, 96), (192, 192, 192))

        def _draw_palette_preview(
            self,
            widget: Gtk.DrawingArea,
            context: object,
        ) -> bool:
            width = max(1, widget.get_allocated_width())
            height = max(1, widget.get_allocated_height())
            samples = self._palette_preview_samples()
            cairo = context
            gradient_height = max(1, height - 12)
            count = max(1, len(samples))
            for index, colour in enumerate(samples):
                x0 = width * index / count
                x1 = width * (index + 1) / count
                cairo.set_source_rgb(
                    colour[0] / 255.0,
                    colour[1] / 255.0,
                    colour[2] / 255.0,
                )
                cairo.rectangle(x0, 0, max(1.0, x1 - x0), gradient_height)
                cairo.fill()
            name = self._combo_text(self.palette)
            file_text = self.palette_file.get_text().strip()
            palette_file = Path(file_text).expanduser() if file_text else None
            profile = None
            try:
                profile = visualizer._kfp_profile_for_selection(name, palette_file)
            except (OSError, ValueError):
                pass
            interior = (
                profile.interior_color
                if profile is not None
                else visualizer._ordinary_interior_color(name, palette_file)
            )
            cairo.set_source_rgb(
                interior[0] / 255.0,
                interior[1] / 255.0,
                interior[2] / 255.0,
            )
            cairo.rectangle(0, gradient_height, width, height - gradient_height)
            cairo.fill()
            cairo.set_source_rgba(0.0, 0.0, 0.0, 0.6)
            cairo.rectangle(0.5, 0.5, width - 1.0, height - 1.0)
            cairo.set_line_width(1.0)
            cairo.stroke()
            return False

        def _profile_changed(self, *_args: object) -> None:
            values = PROFILE_DEFAULTS.get(self._combo_text(self.profile), {})
            for key, widget in (
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
                if key not in values:
                    continue
                value = values[key]
                if isinstance(widget, Gtk.ComboBoxText):
                    self._set_combo(widget, value)
                else:
                    widget.set_text(str(value))

        def _command(self, estimate: bool = False) -> list[str]:
            point_spec = self._point_text()
            has_point = self.random_point.get_active() or bool(point_spec)
            has_x = bool(self.x_center.get_text().strip())
            has_y = bool(self.y_center.get_text().strip())
            if has_point and (has_x or has_y):
                raise ValueError("Point/random point cannot be combined with X/Y centre")
            if has_x != has_y:
                raise ValueError("X and Y centre must be supplied together")
            audio_path = Path(self.audio.get_text()).expanduser().resolve()
            # Preserve the final component so the CLI can reject an output
            # symlink safely instead of resolving it into the target file.
            output_path = Path(self.output.get_text()).expanduser().absolute()
            formula = self._combo_text(self.formula)
            julia_preset = FORMULA_POINTS_BY_SLUG.get(formula, {}).get(point_spec.casefold())
            command = [sys.executable, "-u", str(VISUALIZER), str(audio_path)]
            command.extend(["--output", str(output_path), "--profile", self._combo_text(self.profile)])
            command.extend([
                "--formula", formula,
                "--max-zoom", self.max_zoom.get_text(),
                "--width", self.width.get_text(), "--height", self.height.get_text(),
                "--fps", self.fps.get_text(),
                "--sample-rate", self.sample_rate.get_text(),
                "--separation", self._combo_text(self.separation),
                "--render-scale", self.render_scale.get_text(),
                "--fractal-scale", self.fractal_scale.get_text(),
                "--quality", self._combo_text(self.quality),
                "--keyframe-factor", self.keyframe_factor.get_text(),
                "--keyframe-mode", self._combo_text(self.keyframe_mode),
                "--base-zoom", self.base_zoom.get_text(),
                "--iteration-base", self.iteration_base.get_text(),
                "--iterations-per-decade", self.iterations_per_decade.get_text(),
                "--iteration-cap", self.iteration_cap.get_text(),
                "--zoom-punch", self.zoom_punch.get_text(),
                "--zoom-speed", self.zoom_speed.get_text(),
                "--palette", self._combo_text(self.palette),
                "--beat-strength", f"{self.beat_strength.get_value():g}",
                "--attack", self.attack.get_text(),
                "--release", self.release.get_text(),
                "--series-order", self.series_order.get_text(),
                "--series-block", self.series_block.get_text(),
                "--renderer", self._combo_text(self.renderer),
                "--native-threads", self.native_threads.get_text(),
                "--native-backend", self._combo_text(self.native_backend),
                "--video-preset", self._combo_text(self.video_preset),
                "--codec", self.codec.get_text(),
                "--crf", self.crf.get_text(),
                "--resample", self._combo_text(self.resample),
                "--glow", f"{self.glow.get_value():g}",
                "--motion-blur", f"{self.motion_blur.get_value():g}",
                "--encoder-threads", self.encoder_threads.get_text(),
                "--cache-limit-mb", self.cache_limit_mb.get_text(),
            ])
            if formula == "julia":
                # Values beginning with '-' must be attached to the option;
                # argparse otherwise treats a negative custom c as a flag.
                if (
                    not self.random_point.get_active()
                    and point_spec.casefold() != "random"
                    and julia_preset is not None
                    and julia_preset.julia_c is not None
                ):
                    command.append("--julia-c=" + ",".join(julia_preset.julia_c))
                elif not self.random_point.get_active() and point_spec.casefold() != "random":
                    command.append("--julia-c=" + self.julia_c.get_text().strip())
            if self.random_point.get_active():
                command.append("--random-point")
            elif point_spec:
                command.append("--point=" + point_spec)
            if self.random_seed.get_text().strip():
                command.extend(["--random-seed", self.random_seed.get_text().strip()])
            if self.x_center.get_text().strip():
                command.append("--x-center=" + self.x_center.get_text().strip())
            if self.y_center.get_text().strip():
                command.append("--y-center=" + self.y_center.get_text().strip())
            if self.allow_underspecified_center.get_active():
                command.append("--allow-underspecified-center")
            if self.cache.get_text().strip():
                command.extend(["--cache-dir", str(Path(self.cache.get_text().strip()).expanduser().resolve())])
            if self.durable_cache.get_active():
                command.append("--durable-cache")
            if self.manifest.get_text().strip():
                command.extend(["--manifest", str(Path(self.manifest.get_text().strip()).expanduser().absolute())])
            if self.no_manifest.get_active():
                command.append("--no-manifest")
            if self.palette_file.get_text().strip():
                command.extend(["--palette-file", str(Path(self.palette_file.get_text().strip()).expanduser().resolve())])
            if estimate:
                command.append("--estimate")
            return command

        @staticmethod
        def _process_options() -> dict[str, object]:
            if os.name == "nt":
                return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
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

        def _show_error(self, title: str, message: str) -> None:
            dialog = Gtk.MessageDialog(
                transient_for=self.window,
                modal=True,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text=title,
            )
            dialog.format_secondary_text(message)
            dialog.run()
            dialog.destroy()

        def _start_or_estimate(self, estimate: bool) -> None:
            if self.process is not None:
                return
            if not Path(self.audio.get_text()).expanduser().resolve().is_file():
                self._show_error("Audio file", "Choose an existing audio file first.")
                return
            try:
                command = self._command(estimate=estimate)
                self._process_exit_code = None
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
                self._show_error("Could not start renderer", str(error))
                return
            self._set_progress(None, "Estimating…" if estimate else "Starting…")
            self._set_running(True)
            prefix = "Estimate command: " if estimate else "Command: "
            self._append(prefix + shlex.join(command) + "\n")
            threading.Thread(target=self._read_process, daemon=True).start()

        def _start(self, _button: Gtk.Button | None = None) -> None:
            self._start_or_estimate(False)

        def _estimate(self, _button: Gtk.Button | None = None) -> None:
            self._start_or_estimate(True)

        def _read_process(self) -> None:
            process = self.process
            assert process is not None and process.stdout is not None
            while True:
                line = _read_bounded_line(process.stdout, MAX_LOG_LINE_CHARS)
                if line is None:
                    break
                self._queue_output(line)
            code = process.wait()
            self._process_exit_code = code
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
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                return
            try:
                self.output_queue.put_nowait(text)
            except queue.Full:
                pass

        def _drain_output(self) -> bool:
            chunks: list[str] = []
            completed = False
            try:
                while True:
                    line = self.output_queue.get_nowait()
                    if line == "__FRACTAL_PROCESS_DONE__":
                        completed = True
                    else:
                        chunks.append(line)
            except queue.Empty:
                pass
            if chunks:
                output = "".join(chunks)
                self._update_progress_from_output(output)
                self._append(output)
            if self.process is not None and not self._progress_determinate:
                self.progress.pulse()
            if completed:
                code = self._process_exit_code
                if code == 0:
                    self._set_progress(1.0, "Complete")
                elif code is None:
                    self._set_progress(None, "Finished without an exit status")
                else:
                    self._set_progress(0.0, f"Failed (status {code})")
                self.process = None
                self._set_running(False)
            return True

        def _append(self, text: str) -> None:
            follow_tail = self._log_follow_tail or self._log_at_bottom()
            self.log_buffer.insert(self.log_buffer.get_end_iter(), text)
            line_count = self.log_buffer.get_line_count()
            if line_count > MAX_LOG_LINES:
                start = self.log_buffer.get_start_iter()
                trim_end = self.log_buffer.get_iter_at_line(line_count - MAX_LOG_LINES)
                self.log_buffer.delete(start, trim_end)
            if follow_tail:
                self._log_follow_tail = True
                self._schedule_log_tail()

        def _set_running(self, running: bool) -> None:
            self.start_button.set_sensitive(not running)
            self.estimate_button.set_sensitive(not running)
            self.stop_button.set_sensitive(running)

        def _stop(self, _button: Gtk.Button | None = None) -> None:
            if self.process is not None and self.process.poll() is None:
                self._set_progress(None, "Stopping…")
                self._terminate_process(self.process)
                self._append("Stopping renderer…\n")

        def _on_delete_event(self, _window: Gtk.Window, _event: object) -> bool:
            if self.process is not None and self.process.poll() is None:
                dialog = Gtk.MessageDialog(
                    transient_for=self.window,
                    modal=True,
                    message_type=Gtk.MessageType.QUESTION,
                    buttons=Gtk.ButtonsType.NONE,
                    text="Stop render and close?",
                )
                dialog.add_buttons(
                    "Cancel",
                    Gtk.ResponseType.CANCEL,
                    "Stop and close",
                    Gtk.ResponseType.ACCEPT,
                )
                response = dialog.run()
                dialog.destroy()
                if response != Gtk.ResponseType.ACCEPT:
                    return True
                self._terminate_process(self.process)
            return False


def main() -> None:
    if Gtk is None:
        detail = str(GTK_IMPORT_ERROR) if GTK_IMPORT_ERROR is not None else "GTK3 is unavailable"
        raise SystemExit(
            "GTK3/PyGObject is required for the GUI. Install GTK 3 and PyGObject "
            f"(the Nix shell provides both). Import error: {detail}"
        )
    application = Gtk.Application(
        application_id="com.haakez.FractalAudioViz",
        flags=0,
    )

    # Keep the Python wrapper alive and make repeated application activation
    # (for example, launching the desktop entry twice) focus the existing
    # window instead of creating a second launcher.
    window_holder: list[RenderApp] = []

    def activate(app: Gtk.Application) -> None:
        if not window_holder:
            window_holder.append(RenderApp(app))
        window_holder[0].window.present()

    application.connect("activate", activate)
    raise SystemExit(application.run(sys.argv))


if __name__ == "__main__":
    main()
