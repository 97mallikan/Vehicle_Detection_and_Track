"""
===============================================================
CPU--GPU COMPUTATIONAL ABLATION STUDY
===============================================================

Configurations:

C1 : YOLOv8 CPU       + OpenCV KNN CPU
C2 : YOLOv8 GPU       + OpenCV KNN CPU
C3 : YOLOv8 GPU       + Custom CUDA KNN/Motion Backend
C4 : YOLOv8 GPU       + Custom CUDA KNN/Motion Backend
     YOLO executed periodically every YOLO_STRIDE frames

Metrics:

- FPS
- Average frame latency (ms)
- Average GPU utilization (%)

===============================================================
"""

import os
import time
import csv
import threading

import cv2
import numpy as np
import torch

from ultralytics import YOLO

# Your custom CUDA backend
from cuda_backend import CudaMotionBackend


# ===============================================================
# CONFIGURATION
# ===============================================================

VIDEO_PATH = "/home/anurag/python-environments/yolov8-object-tracking/RealLifeVideo.mp4"

MODEL_PATH = "/home/anurag/yolo26/runs/detect/train12/weights/best.pt"

CUDA_BACKEND_LIB = ("/home/anurag/python-environments/yolov8-object-tracking/Vehicle_Detection_and_Track/libmotion_cuda.so")

OUTPUT_CSV = "hardware_ablation_results.csv"


# ---------------------------------------------------------------
# YOLO parameters
# ---------------------------------------------------------------

IMGSZ = 1280

CONF_THRESHOLD = 0.25

YOLO_STRIDE = 5


# ---------------------------------------------------------------
# KNN parameters
# ---------------------------------------------------------------

KNN_HISTORY = 500

KNN_DIST2_THRESHOLD = 400.0

KNN_LEARNING_RATE = 0.0015

KNN_MIN_AREA = 800


# ---------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------

WARMUP_FRAMES = 20

MAX_BENCHMARK_FRAMES = 500


# ===============================================================
# GPU UTILIZATION MONITOR
# ===============================================================

class GPUMonitor:

    def __init__(self, gpu_index=0):

        self.gpu_index = gpu_index

        self.samples = []

        self.running = False

        self.thread = None

        try:

            import pynvml

            self.pynvml = pynvml

            pynvml.nvmlInit()

            self.handle = pynvml.nvmlDeviceGetHandleByIndex(
                gpu_index
            )

            self.available = True

        except Exception as e:

            print(
                "Warning: GPU monitoring unavailable:",
                e
            )

            self.available = False


    # -----------------------------------------------------------

    def _monitor(self):

        while self.running:

            try:

                util = self.pynvml.nvmlDeviceGetUtilizationRates(
                    self.handle
                )

                self.samples.append(
                    float(util.gpu)
                )

            except Exception:

                pass

            time.sleep(0.1)


    # -----------------------------------------------------------

    def start(self):

        self.samples = []

        if not self.available:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )

        self.thread.start()


    # -----------------------------------------------------------

    def stop(self):

        if not self.available:
            return 0.0

        self.running = False

        if self.thread is not None:

            self.thread.join(
                timeout=1.0
            )

        if not self.samples:

            return 0.0

        return float(
            np.mean(self.samples)
        )


# ===============================================================
# CPU KNN PIPELINE
# ===============================================================

class CPUKNNPipeline:

    def __init__(self):

        self.knn = cv2.createBackgroundSubtractorKNN(

            history=KNN_HISTORY,

            dist2Threshold=KNN_DIST2_THRESHOLD,

            detectShadows=True

        )

        self.kernel = cv2.getStructuringElement(

            cv2.MORPH_RECT,

            (3, 3)

        )


    # -----------------------------------------------------------

    def process(self, frame):

        fgmask = self.knn.apply(

            frame,

            learningRate=KNN_LEARNING_RATE

        )


        # Threshold
        _, fgmask = cv2.threshold(

            fgmask,

            200,

            255,

            cv2.THRESH_BINARY

        )


        # Morphological opening
        fgmask = cv2.morphologyEx(

            fgmask,

            cv2.MORPH_OPEN,

            self.kernel

        )


        # Morphological closing
        fgmask = cv2.morphologyEx(

            fgmask,

            cv2.MORPH_CLOSE,

            self.kernel

        )


        # Median filtering
        fgmask = cv2.medianBlur(

            fgmask,

            5

        )


        # Connected components
        num_labels, labels, stats, centroids = (

            cv2.connectedComponentsWithStats(

                fgmask,

                connectivity=8

            )

        )


        blobs = []


        for i in range(
            1,
            num_labels
        ):

            area = int(
                stats[
                    i,
                    cv2.CC_STAT_AREA
                ]
            )


            if area < KNN_MIN_AREA:

                continue


            x = int(
                stats[
                    i,
                    cv2.CC_STAT_LEFT
                ]
            )


            y = int(
                stats[
                    i,
                    cv2.CC_STAT_TOP
                ]
            )


            w = int(
                stats[
                    i,
                    cv2.CC_STAT_WIDTH
                ]
            )


            h = int(
                stats[
                    i,
                    cv2.CC_STAT_HEIGHT
                ]
            )


            cx, cy = centroids[i]


            blobs.append({

                "area": area,

                "x": x,

                "y": y,

                "w": w,

                "h": h,

                "cx": float(cx),

                "cy": float(cy)

            })


        return fgmask, blobs


# ===============================================================
# YOLO INFERENCE
# ===============================================================

def run_yolo(
    model,
    frame,
    device
):

    results = model.predict(

        frame,

        imgsz=IMGSZ,

        conf=CONF_THRESHOLD,

        device=device,

        verbose=False

    )

    return results


# ===============================================================
# CUDA SYNCHRONIZATION
# ===============================================================

def synchronize_cuda():

    if torch.cuda.is_available():

        torch.cuda.synchronize()


# ===============================================================
# LOAD VIDEO INFORMATION
# ===============================================================

def get_video_information():

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )


    if not cap.isOpened():

        raise RuntimeError(

            f"Unable to open video: "
            f"{VIDEO_PATH}"

        )


    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )


    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    cap.release()


    return (
        width,
        height,
        total_frames
    )


# ===============================================================
# GPU WARMUP
# ===============================================================

def warmup_yolo(
    model,
    frame
):

    print(
        "Warming up YOLO GPU..."
    )


    for _ in range(5):

        run_yolo(

            model,

            frame,

            0

        )


    synchronize_cuda()


# ===============================================================
# BENCHMARK CONFIGURATION
# ===============================================================

def benchmark_configuration(
    config_name,
    yolo_device,
    use_cuda_knn,
    periodic_yolo
):

    print()

    print(
        "=" * 75
    )

    print(
        f"Running {config_name}"
    )

    print(
        "=" * 75
    )


    # -----------------------------------------------------------
    # Load YOLO separately for each configuration
    # -----------------------------------------------------------

    print(
        "Loading YOLO model..."
    )

    model = YOLO(
        MODEL_PATH
    )


    # -----------------------------------------------------------
    # Open video
    # -----------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )


    if not cap.isOpened():

        raise RuntimeError(
            "Unable to open video"
        )


    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )


    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    # -----------------------------------------------------------
    # Initialize background processing
    # -----------------------------------------------------------

    if use_cuda_knn:

        print(
            "Using custom CUDA KNN backend"
        )


        motion_backend = CudaMotionBackend(

            CUDA_BACKEND_LIB,

            width,

            height

        )


        cpu_knn = None


    else:

        print(
            "Using OpenCV CPU KNN"
        )


        cpu_knn = (
            CPUKNNPipeline()
        )


        motion_backend = None


    # -----------------------------------------------------------
    # Read one frame for GPU warmup
    # -----------------------------------------------------------

    ret, first_frame = cap.read()


    if not ret:

        raise RuntimeError(
            "Unable to read video"
        )


    # -----------------------------------------------------------
    # Warmup GPU
    # -----------------------------------------------------------

    if yolo_device != "cpu":

        warmup_yolo(

            model,

            first_frame

        )


    # Reset video
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )


    # -----------------------------------------------------------
    # GPU monitor
    # -----------------------------------------------------------

    gpu_monitor = (
        GPUMonitor(0)
    )


    # -----------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------

    frame_count = 0

    benchmark_count = 0


    total_processing_time = 0.0


    latency_values = []


    # -----------------------------------------------------------
    # Start GPU monitoring
    # -----------------------------------------------------------

    gpu_monitor.start()


    # ===========================================================
    # FRAME LOOP
    # ===========================================================

    while True:

        ret, frame = cap.read()


        if not ret:

            break


        # -------------------------------------------------------
        # Limit benchmark length
        # -------------------------------------------------------

        if (

            MAX_BENCHMARK_FRAMES > 0

            and

            benchmark_count >=
            MAX_BENCHMARK_FRAMES

        ):

            break


        # -------------------------------------------------------
        # Ignore first frames from final statistics
        # -------------------------------------------------------

        warmup_phase = (

            frame_count <
            WARMUP_FRAMES

        )


        # =======================================================
        # Synchronize before timing
        # =======================================================

        if yolo_device != "cpu":

            synchronize_cuda()


        frame_start = (
            time.perf_counter()
        )


        # =======================================================
        # KNN / MOTION PROCESSING
        # =======================================================

        if use_cuda_knn:

            fgmask, blobs = (

                motion_backend.process(

                    frame,

                    learning_rate=
                        KNN_LEARNING_RATE,

                    min_area=
                        KNN_MIN_AREA

                )

            )


        else:

            fgmask, blobs = (

                cpu_knn.process(
                    frame
                )

            )


        # =======================================================
        # YOLO
        # =======================================================

        if periodic_yolo:

            run_detector = (

                frame_count %
                YOLO_STRIDE == 0

            )

        else:

            run_detector = True


        if run_detector:

            run_yolo(

                model,

                frame,

                yolo_device

            )


        # =======================================================
        # Synchronize after GPU operations
        # =======================================================

        if yolo_device != "cpu":

            synchronize_cuda()


        frame_end = (
            time.perf_counter()
        )


        elapsed = (

            frame_end -
            frame_start

        )


        # -------------------------------------------------------
        # Do not include warmup frames
        # -------------------------------------------------------

        if not warmup_phase:

            total_processing_time += (
                elapsed
            )


            latency_values.append(

                elapsed * 1000.0

            )


            benchmark_count += 1


        frame_count += 1


        print(

            f"\r{config_name}: "
            f"{benchmark_count}/"
            f"{MAX_BENCHMARK_FRAMES}",

            end=""

        )


    print()


    # -----------------------------------------------------------
    # Finish GPU monitoring
    # -----------------------------------------------------------

    gpu_utilization = (
        gpu_monitor.stop()
    )


    cap.release()


    if motion_backend is not None:

        motion_backend.close()


    # -----------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------

    if total_processing_time > 0:

        fps = (

            benchmark_count /
            total_processing_time

        )

    else:

        fps = 0.0


    if latency_values:

        mean_latency = float(

            np.mean(
                latency_values
            )

        )

    else:

        mean_latency = 0.0


    # -----------------------------------------------------------
    # CPU-only GPU utilization
    # -----------------------------------------------------------

    if (

        yolo_device == "cpu"

        and

        not use_cuda_knn

    ):

        gpu_utilization = 0.0


    result = {

        "config":
            config_name,

        "fps":
            float(fps),

        "latency_ms":
            mean_latency,

        "gpu_utilization":
            float(gpu_utilization)

    }


    print(
        f"FPS              : "
        f"{fps:.2f}"
    )


    print(
        f"Latency          : "
        f"{mean_latency:.2f} ms"
    )


    print(
        f"GPU Utilization  : "
        f"{gpu_utilization:.2f}%"
    )


    return result


# ===============================================================
# SAVE CSV
# ===============================================================

def save_results(
    results
):

    with open(

        OUTPUT_CSV,

        "w",

        newline=""

    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow([

            "Configuration",

            "YOLOv8",

            "KNN/Post",

            "FPS",

            "Latency (ms)",

            "GPU Utilization (%)"

        ])


        descriptions = {

            "C1":
                (
                    "CPU",
                    "CPU"
                ),

            "C2":
                (
                    "GPU",
                    "CPU"
                ),

            "C3":
                (
                    "GPU",
                    "CUDA"
                ),

            "C4":
                (
                    "GPU (Periodic)",
                    "CUDA"
                )

        }


        for result in results:

            yolo_desc, knn_desc = (

                descriptions[
                    result["config"]
                ]

            )


            writer.writerow([

                result[
                    "config"
                ],

                yolo_desc,

                knn_desc,

                f"{result['fps']:.2f}",

                f"{result['latency_ms']:.2f}",

                f"{result['gpu_utilization']:.2f}"

            ])


    print(
        f"\nResults saved to "
        f"{OUTPUT_CSV}"
    )


# ===============================================================
# PRINT LATEX TABLE
# ===============================================================

def print_latex_table(
    results
):

    lookup = {

        result["config"]:
            result

        for result in results

    }


    print()

    print(
        "=" * 75
    )

    print(
        "LATEX TABLE"
    )

    print(
        "=" * 75
    )


    print(
r"""
\begin{table}[htbp]
\centering
\caption{CPU--GPU Computational Ablation Study}
\label{tab:hardware_ablation}
\renewcommand{\arraystretch}{1.2}
\resizebox{\columnwidth}{!}{
\begin{tabular}{c c c c c c}
\hline
\textbf{Config.} &
\textbf{YOLOv8} &
\textbf{KNN/Post.} &
\textbf{FPS} &
\textbf{Latency (ms)} &
\textbf{GPU Util. (\%)} \\
\hline
"""
    )


    rows = [

        (
            "C1",
            "CPU",
            "CPU"
        ),

        (
            "C2",
            "GPU",
            "CPU"
        ),

        (
            "C3",
            "GPU",
            "CUDA"
        ),

        (
            "C4",
            "GPU (Periodic)",
            "CUDA"
        )

    ]


    for config, yolo, knn in rows:

        result = lookup[
            config
        ]


        print(

            f"{config} & "

            f"{yolo} & "

            f"{knn} & "

            f"{result['fps']:.2f} & "

            f"{result['latency_ms']:.2f} & "

            f"{result['gpu_utilization']:.2f} \\\\"

        )


    print(
r"""\hline
\end{tabular}
}
\end{table}
"""
    )


# ===============================================================
# MAIN
# ===============================================================

def main():

    if not os.path.isfile(
        VIDEO_PATH
    ):

        raise FileNotFoundError(

            f"Video not found: "
            f"{VIDEO_PATH}"

        )


    if not os.path.isfile(
        MODEL_PATH
    ):

        raise FileNotFoundError(

            f"YOLO model not found: "
            f"{MODEL_PATH}"

        )


    if not torch.cuda.is_available():

        raise RuntimeError(

            "CUDA is not available. "
            "C2-C4 cannot be evaluated."

        )


    if not os.path.isfile(
        CUDA_BACKEND_LIB
    ):

        raise FileNotFoundError(

            f"CUDA backend library not found: "
            f"{CUDA_BACKEND_LIB}"

        )


    print(
        "Video:",
        VIDEO_PATH
    )

    print(
        "YOLO model:",
        MODEL_PATH
    )

    print(
        "YOLO stride:",
        YOLO_STRIDE
    )


    results = []


    # ===========================================================
    # C1
    # YOLO CPU + KNN CPU
    # ===========================================================

    results.append(

        benchmark_configuration(

            config_name="C1",

            yolo_device="cpu",

            use_cuda_knn=False,

            periodic_yolo=False

        )

    )


    # ===========================================================
    # C2
    # YOLO GPU + KNN CPU
    # ===========================================================

    results.append(

        benchmark_configuration(

            config_name="C2",

            yolo_device=0,

            use_cuda_knn=False,

            periodic_yolo=False

        )

    )


    # ===========================================================
    # C3
    # YOLO GPU + CUDA KNN
    # ===========================================================

    results.append(

        benchmark_configuration(

            config_name="C3",

            yolo_device=0,

            use_cuda_knn=True,

            periodic_yolo=False

        )

    )


    # ===========================================================
    # C4
    # Periodic YOLO GPU + CUDA KNN
    # ===========================================================

    results.append(

        benchmark_configuration(

            config_name="C4",

            yolo_device=0,

            use_cuda_knn=True,

            periodic_yolo=True

        )

    )


    save_results(
        results
    )


    print_latex_table(
        results
    )


    print()

    print(
        "=" * 75
    )

    print(
        "HARDWARE ABLATION COMPLETED"
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":

    main()
