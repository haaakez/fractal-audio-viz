#!/usr/bin/env python3
"""GTK3 launcher for the reproducible command-line renderer.

The GUI is deliberately a thin launcher: it owns the controls, process
lifecycle, and progress log, while ``visualizer.py`` remains the single source
of rendering behaviour. GTK's native scrolled container and expander keep the
compact default view stable even when the technical controls are opened.
"""

from __future__ import annotations

import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path

try:
    import gi

    gi.require_version("Gdk", "3.0")
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gdk, GLib, Gtk
except (ImportError, ValueError) as error:  # pragma: no cover - environment dependent
    Gdk = GLib = Gtk = None  # type: ignore[assignment]
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

            self._configure_dark_theme()
            self._build_widgets()
            self.profile.connect("changed", self._profile_changed)
            self.formula.connect("changed", self._formula_changed)
            self.point_combo.connect("changed", self._point_changed)
            self._profile_changed()
            self._formula_changed()
            GLib.timeout_add(100, self._drain_output)
            self.window.show_all()
            self.technical_expander.set_expanded(False)
            self._set_technical_child_visible(False)

        @staticmethod
        def _configure_dark_theme() -> None:
            """Use a coherent dark palette for GTK and native popovers."""

            settings = Gtk.Settings.get_default()
            if settings is not None:
                settings.set_property("gtk-application-prefer-dark-theme", True)
                settings.set_property("gtk-enable-animations", True)

            css = b"""
            * {
                color: #e8eaf0;
            }
            window, .fractal-root, scrolledwindow, viewport {
                background-color: #111318;
            }
            headerbar {
                background-image: none;
                background-color: #171b22;
                border-bottom: 1px solid #343b49;
                box-shadow: none;
            }
            frame {
                border: 1px solid #343b49;
                border-radius: 8px;
                background-color: #191d25;
            }
            frame > label {
                color: #f4f6fb;
                background-color: #191d25;
                padding: 0 6px;
            }
            label {
                color: #e8eaf0;
            }
            entry, combobox box, spinbutton {
                color: #f1f3f8;
                background-image: none;
                background-color: #242a35;
                border: 1px solid #3b4555;
                border-radius: 5px;
                padding: 5px 7px;
                caret-color: #ffffff;
            }
            entry:focus, combobox:focus, spinbutton:focus {
                border-color: #5b9cff;
                box-shadow: 0 0 0 1px #315b9a;
            }
            entry:disabled, combobox:disabled, spinbutton:disabled {
                color: #697383;
                background-color: #171b22;
            }
            button {
                color: #f1f3f8;
                background-image: none;
                background-color: #252c38;
                border: 1px solid #3b4555;
                border-radius: 5px;
                padding: 6px 12px;
            }
            button:hover {
                background-color: #3d72bd;
                border-color: #5b9cff;
            }
            button:active, button:checked {
                background-color: #315b9a;
            }
            button:disabled {
                color: #626b78;
                background-color: #191d25;
                border-color: #2a303b;
            }
            checkbutton, radiobutton {
                color: #e8eaf0;
            }
            checkbutton check, radiobutton radio {
                background-color: #242a35;
                border: 1px solid #526073;
            }
            checkbutton:checked check, radiobutton:checked radio {
                background-color: #315b9a;
                border-color: #5b9cff;
            }
            scale trough {
                background-color: #2a313e;
                border: 1px solid #3b4555;
                border-radius: 4px;
                min-height: 6px;
            }
            scale highlight {
                background-color: #3d72bd;
                border-radius: 4px;
            }
            scale slider {
                background-image: none;
                background-color: #d9e6ff;
                border: 1px solid #5b9cff;
                min-width: 14px;
                min-height: 14px;
            }
            expander title {
                color: #f1f3f8;
                padding: 3px 0;
            }
            expander arrow {
                color: #8db9ff;
            }
            scrollbar trough {
                background-color: #0b0e13;
            }
            scrollbar slider {
                background-color: #3b4555;
                border: 1px solid #526073;
                border-radius: 6px;
                min-width: 10px;
                min-height: 10px;
            }
            scrollbar slider:hover {
                background-color: #5b6d89;
            }
            textview, textview text {
                color: #e8eaf0;
                background-color: #0b0e13;
            }
            tooltip {
                color: #f1f3f8;
                background-color: #242a35;
                border: 1px solid #526073;
            }
            """
            provider = Gtk.CssProvider()
            provider.load_from_data(css)
            screen = Gdk.Screen.get_default()
            if screen is not None:
                Gtk.StyleContext.add_provider_for_screen(
                    screen,
                    provider,
                    Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
                )

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
            header = Gtk.HeaderBar()
            header.set_show_close_button(True)
            header.set_title("Fractal Audio Viz")
            header.set_subtitle("Audio-reactive deep-zoom renderer")
            self.window.set_titlebar(header)

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
                ("aurora", "fire", "ocean", "neon", "sunset", "mono"),
                "aurora",
            )
            self.palette_file = self._entry()

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
            self._add_path_row(
                render_grid,
                row,
                "Palette file",
                self.palette_file,
                action=Gtk.FileChooserAction.OPEN,
                patterns=("*.txt",),
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
            log_scroll = Gtk.ScrolledWindow()
            log_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
            log_scroll.set_min_content_height(240)
            log_scroll.set_vexpand(True)
            log_scroll.add(self.log)

            content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
            content.get_style_context().add_class("fractal-root")
            content.set_border_width(16)
            content.set_hexpand(True)
            content.pack_start(self._frame("Render", render_grid), False, False, 0)
            content.pack_start(self._frame("Audio and effects", effects_grid), False, False, 0)
            content.pack_start(self.technical_expander, False, False, 0)
            content.pack_start(buttons, False, False, 0)
            content.pack_start(self._label("Renderer output"), False, False, 0)
            content.pack_start(log_scroll, True, True, 0)

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
                self._append("".join(chunks))
            if completed:
                self.process = None
                self._set_running(False)
            return True

        def _append(self, text: str) -> None:
            self.log_buffer.insert(self.log_buffer.get_end_iter(), text)
            line_count = self.log_buffer.get_line_count()
            if line_count > MAX_LOG_LINES:
                start = self.log_buffer.get_start_iter()
                trim_end = self.log_buffer.get_iter_at_line(line_count - MAX_LOG_LINES)
                self.log_buffer.delete(start, trim_end)
            end = self.log_buffer.get_end_iter()
            self.log.scroll_to_iter(end, 0.0, False, 0.0, 1.0)

        def _set_running(self, running: bool) -> None:
            self.start_button.set_sensitive(not running)
            self.estimate_button.set_sensitive(not running)
            self.stop_button.set_sensitive(running)

        def _stop(self, _button: Gtk.Button | None = None) -> None:
            if self.process is not None and self.process.poll() is None:
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
