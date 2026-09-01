#!/usr/bin/env python3
"""Small Tkinter launcher for the command-line renderer.

Tkinter is part of Python's standard library and is available on Windows and
Linux. The renderer still runs in a child process, so the window stays
responsive and the CLI remains the source of truth for reproducible renders.
"""

from __future__ import annotations

import queue
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

from deep_zoom_points import DEEP_ZOOM_POINTS
from profiles import PROFILE_CHOICES, PROFILE_DEFAULTS


ROOT = Path(__file__).resolve().parent
VISUALIZER = ROOT / "visualizer.py"


class RenderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Fractal Audio Viz")
        self.root.minsize(720, 560)
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()

        self.audio = tk.StringVar(value=str(ROOT / "song.mp3"))
        self.output = tk.StringVar(value=str(ROOT / "fractal_viz.mp4"))
        self.profile = tk.StringVar(value="preview")
        self.formula = tk.StringVar(value="mandelbrot")
        self.point = tk.StringVar(value="")
        self.julia_c = tk.StringVar(value="-0.8,0.156")
        self.max_zoom = tk.StringVar(value="1e24")
        self.width = tk.StringVar(value="960")
        self.height = tk.StringVar(value="540")
        self.fps = tk.StringVar(value="24")
        self.cache = tk.StringVar(value=str(ROOT / "cache"))
        self.palette = tk.StringVar(value="aurora")
        self.palette_file = tk.StringVar(value="")
        self.beat_strength = tk.StringVar(value="0")
        self.glow = tk.StringVar(value="0")
        self.motion_blur = tk.StringVar(value="0")

        self._build_widgets()
        self.profile.trace_add("write", self._profile_changed)
        self.root.after(100, self._drain_output)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_widgets(self) -> None:
        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(1, weight=1)

        row = 0
        row = self._path_row(root_frame, row, "Audio", self.audio, [
            ("Audio files", "*.mp3 *.wav *.flac *.ogg"),
            ("All files", "*"),
        ])
        row = self._path_row(root_frame, row, "Output", self.output, [
            ("MP4", "*.mp4"),
            ("All files", "*"),
        ], save=True)

        ttk.Label(root_frame, text="Profile").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            root_frame, textvariable=self.profile, values=PROFILE_CHOICES, state="readonly"
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(root_frame, text="Formula").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            root_frame,
            textvariable=self.formula,
            values=("mandelbrot", "julia", "burning-ship", "tricorn", "multibrot3"),
            state="readonly",
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1
        row = self._entry_row(root_frame, row, "Point", self.point, "slug, random, or REAL,IMAG")
        row = self._entry_row(root_frame, row, "Julia c", self.julia_c, "used by the Julia formula")
        row = self._entry_row(root_frame, row, "Max zoom", self.max_zoom, "for example 1e150")

        dimensions = ttk.Frame(root_frame)
        dimensions.grid(row=row, column=1, sticky="ew", pady=4)
        for index in range(6):
            dimensions.columnconfigure(index, weight=1 if index in {1, 3, 5} else 0)
        ttk.Label(root_frame, text="Video").grid(row=row, column=0, sticky="w", pady=4)
        for index, (label, variable) in enumerate(
            (("W", self.width), ("H", self.height), ("FPS", self.fps))
        ):
            ttk.Label(dimensions, text=label).grid(row=0, column=index * 2, padx=(0, 3))
            ttk.Entry(dimensions, textvariable=variable, width=8).grid(
                row=0, column=index * 2 + 1, sticky="ew", padx=(0, 12)
            )
        row += 1
        row = self._path_row(
            root_frame,
            row,
            "Cache",
            self.cache,
            [("Directories", "*")],
            directory=True,
        )
        row = self._entry_row(root_frame, row, "Palette", self.palette, "aurora, fire, ocean, neon...")
        row = self._path_row(
            root_frame,
            row,
            "Palette file",
            self.palette_file,
            [("Palette text", "*.txt"), ("All files", "*")],
        )
        row = self._entry_row(root_frame, row, "Beat strength", self.beat_strength, "0 disables onset sync")
        row = self._entry_row(root_frame, row, "Glow", self.glow, "0 to 1")
        row = self._entry_row(root_frame, row, "Motion blur", self.motion_blur, "0 to <1")

        buttons = ttk.Frame(root_frame)
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        self.start_button = ttk.Button(buttons, text="Render", command=self._start)
        self.start_button.pack(side="left")
        self.estimate_button = ttk.Button(buttons, text="Estimate", command=self._estimate)
        self.estimate_button.pack(side="left", padx=6)
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left")
        row += 1

        self.log = tk.Text(root_frame, height=16, wrap="word", state="disabled", background="#101218", foreground="#e8eaf0")
        self.log.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(4, 0))
        root_frame.rowconfigure(row, weight=1)

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

    def _profile_changed(self, *_: object) -> None:
        values = PROFILE_DEFAULTS.get(self.profile.get(), {})
        for key, variable in (
            ("width", self.width),
            ("height", self.height),
            ("fps", self.fps),
            ("max_zoom", self.max_zoom),
            ("beat_strength", self.beat_strength),
        ):
            if key in values:
                variable.set(str(values[key]))

    def _command(self, estimate: bool = False) -> list[str]:
        command = [sys.executable, "-u", str(VISUALIZER), self.audio.get()]
        command.extend(["--output", self.output.get(), "--profile", self.profile.get()])
        command.extend([
            "--formula", self.formula.get(),
            "--julia-c", self.julia_c.get(),
            "--max-zoom", self.max_zoom.get(),
            "--width", self.width.get(), "--height", self.height.get(), "--fps", self.fps.get(),
            "--palette", self.palette.get(),
            "--beat-strength", self.beat_strength.get(),
            "--glow", self.glow.get(), "--motion-blur", self.motion_blur.get(),
        ])
        if self.point.get().strip():
            command.extend(["--point", self.point.get().strip()])
        if self.cache.get().strip():
            command.extend(["--cache-dir", self.cache.get().strip()])
        if self.palette_file.get().strip():
            command.extend(["--palette-file", self.palette_file.get().strip()])
        if estimate:
            command.append("--estimate")
        return command

    def _start(self) -> None:
        if self.process is not None:
            return
        if not Path(self.audio.get()).expanduser().is_file():
            messagebox.showerror("Audio file", "Choose an existing audio file first.")
            return
        try:
            self.process = subprocess.Popen(
                self._command(),
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            messagebox.showerror("Could not start renderer", str(error))
            return
        self._set_running(True)
        self._append("Command: " + " ".join(self._command()) + "\n")
        threading.Thread(target=self._read_process, daemon=True).start()

    def _estimate(self) -> None:
        if self.process is not None:
            return
        if not Path(self.audio.get()).expanduser().is_file():
            messagebox.showerror("Audio file", "Choose an existing audio file first.")
            return
        try:
            self.process = subprocess.Popen(
                self._command(estimate=True),
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            messagebox.showerror("Could not start renderer", str(error))
            return
        self._set_running(True)
        self._append("Estimate command: " + " ".join(self._command(True)) + "\n")
        threading.Thread(target=self._read_process, daemon=True).start()

    def _read_process(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.output_queue.put(line)
        code = self.process.wait()
        self.output_queue.put(f"\nProcess exited with status {code}.\n")
        self.output_queue.put("__FRACTAL_PROCESS_DONE__")

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
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.start_button.configure(state=state)
        self.estimate_button.configure(state=state)
        self.stop_button.configure(state="normal" if running else "disabled")

    def _stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self._append("Stopping renderer...\n")

    def _close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Stop render?", "A render is still running. Stop it and close?"):
                return
            self.process.terminate()
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
