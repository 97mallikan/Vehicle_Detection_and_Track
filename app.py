import os
import sys
import shlex
import platform
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

MAIN_PY = os.path.join(os.path.dirname(__file__), "main.py")


class TrackerGUI(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("YOLO Tracker Launcher")
        self.geometry("900x650")

        if platform.system() == "Windows":
            self.state("zoomed")
        else:
            self.attributes("-zoomed", True)

        self.proc = None

        # -----------------------
        # Variables
        # -----------------------

        self.video_path = tk.StringVar()
        self.weights_path = tk.StringVar()

        self.device = tk.StringVar(value="cuda")

        self.imgsz = tk.IntVar(value=1280)
        self.conf = tk.DoubleVar(value=0.25)
        self.stride = tk.IntVar(value=5)

        self.pol_points = tk.IntVar(value=4)

        self._build_form()
        self._build_log()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------------------------------------------

    def _build_form(self):

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="x")

        def row(label, widget, r):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=6)
            widget.grid(row=r, column=1, sticky="ew", pady=6)
            frm.grid_columnconfigure(1, weight=1)

        # Video
        video_entry = ttk.Entry(frm, textvariable=self.video_path)
        row("Video path", video_entry, 0)
        ttk.Button(frm, text="Browse", command=self.browse_video).grid(row=0, column=2)

        # Weights
        weights_entry = ttk.Entry(frm, textvariable=self.weights_path)
        row("YOLO weights", weights_entry, 1)
        ttk.Button(frm, text="Browse", command=self.browse_weights).grid(row=1, column=2)

        # Device
        row("Device", ttk.Entry(frm, textvariable=self.device), 2)

        # YOLO params
        row("YOLO imgsz", ttk.Spinbox(frm, from_=320, to=4096, increment=32, textvariable=self.imgsz), 3)
        row("YOLO conf", ttk.Spinbox(frm, from_=0.01, to=0.99, increment=0.01, textvariable=self.conf), 4)
        row("YOLO stride", ttk.Spinbox(frm, from_=1, to=60, increment=1, textvariable=self.stride), 5)

        row("Polygon points", ttk.Spinbox(frm, from_=3, to=12, increment=1, textvariable=self.pol_points), 6)

        # Buttons
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=9, column=0, columnspan=3, pady=10)

        self.run_btn = ttk.Button(btn_frame, text="Run Tracker", command=self.run_tracker)
        self.run_btn.pack(side="left")

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_tracker, state="disabled")
        self.stop_btn.pack(side="left", padx=10)

        ttk.Button(btn_frame, text="Clear Log", command=self.clear_log).pack(side="left")

    # -------------------------------------------------------

    def _build_log(self):

        log_frame = ttk.Frame(self, padding=(12, 8))
        log_frame.pack(fill="both", expand=True)

        ttk.Label(log_frame, text="Console Output").pack(anchor="w")

        self.log = tk.Text(log_frame, height=20)
        self.log.pack(fill="both", expand=True)

    # -------------------------------------------------------

    def browse_video(self):
        path = filedialog.askopenfilename(
            parent=self,
            initialdir=os.path.expanduser("~"),
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.video_path.set(path)

    def browse_weights(self):
        path = filedialog.askopenfilename(
            parent=self,
            initialdir=os.path.expanduser("~"),
            title="Select YOLO weights",
            filetypes=[
                ("PyTorch weights", "*.pt"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.weights_path.set(path)

    # -------------------------------------------------------

    def build_command(self):

        vp = self.video_path.get().strip()
        wp = self.weights_path.get().strip()

        if not vp:
            raise ValueError("Select video")

        if not wp:
            raise ValueError("Select weights")

        if not os.path.exists(vp):
            raise ValueError("Video not found")

        if not os.path.exists(wp):
            raise ValueError("Weights not found")

        cmd = [
            sys.executable,
            MAIN_PY,

            "--video", vp,
            "--weights", wp,

            "--device", self.device.get(),

            "--imgsz", str(self.imgsz.get()),
            "--conf", str(self.conf.get()),
            "--stride", str(self.stride.get()),

            "--poly", str(self.pol_points.get())
        ]

        return cmd

    # -------------------------------------------------------

    def run_tracker(self):

        if self.proc is not None:
            messagebox.showinfo("Running", "Tracker already running")
            return

        try:
            cmd = self.build_command()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.append_log("Running:\n" + " ".join(shlex.quote(x) for x in cmd) + "\n\n")

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        threading.Thread(target=self._read_output, daemon=True).start()

    # -------------------------------------------------------

    def _read_output(self):

        for line in self.proc.stdout:
            self.log.after(0, self.append_log, line)

        self.proc = None
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    # -------------------------------------------------------

    def append_log(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def clear_log(self):
        self.log.delete("1.0", "end")

    # -------------------------------------------------------

    def stop_tracker(self):
        if self.proc:
            self.proc.terminate()

    def on_close(self):
        if self.proc:
            self.proc.terminate()
        self.destroy()


if __name__ == "__main__":
    app = TrackerGUI()
    app.mainloop()