import ctypes
import numpy as np


class BlobRegion(ctypes.Structure):
    _fields_ = [
        ("label", ctypes.c_int),
        ("area", ctypes.c_int),
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("w", ctypes.c_int),
        ("h", ctypes.c_int),
        ("cx", ctypes.c_float),
        ("cy", ctypes.c_float),
    ]


class TrackCenter(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
    ]


class ObjectFlowResult(ctypes.Structure):
    _fields_ = [
        ("majority_vx", ctypes.c_float),
        ("majority_vy", ctypes.c_float),
        ("majority_mag", ctypes.c_float),
        ("valid_count", ctypes.c_int),
        ("anomaly_count", ctypes.c_int),
        ("anomaly_ratio", ctypes.c_float),
        ("kernel_x1", ctypes.c_int),
        ("kernel_y1", ctypes.c_int),
        ("kernel_x2", ctypes.c_int),
        ("kernel_y2", ctypes.c_int),
    ]


class FlowPoint(ctypes.Structure):
    _fields_ = [
        ("x0", ctypes.c_int),
        ("y0", ctypes.c_int),
        ("vx", ctypes.c_float),
        ("vy", ctypes.c_float),
        ("mag", ctypes.c_float),
        ("anomalous", ctypes.c_int),
    ]


class CudaMotionBackend:
    def __init__(self, lib_path, width, height, max_regions=2048, point_stride=128):
        self.lib = ctypes.CDLL(lib_path)

        self.lib.motion_create.restype = ctypes.c_void_p
        self.lib.motion_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

        self.lib.motion_destroy.restype = None
        self.lib.motion_destroy.argtypes = [ctypes.c_void_p]

        self.lib.motion_process.restype = ctypes.c_int
        self.lib.motion_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
        ]

        self.lib.motion_download_mask.restype = None
        self.lib.motion_download_mask.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
        ]

        self.lib.motion_get_blob_count.restype = ctypes.c_int
        self.lib.motion_get_blob_count.argtypes = [ctypes.c_void_p]

        self.lib.motion_get_blobs.restype = ctypes.c_int
        self.lib.motion_get_blobs.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(BlobRegion),
            ctypes.c_int,
        ]

        self.lib.motion_analyze_objects.restype = ctypes.c_int
        self.lib.motion_analyze_objects.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(TrackCenter),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ObjectFlowResult),
            ctypes.POINTER(FlowPoint),
            ctypes.c_int,
        ]

        self.width = int(width)
        self.height = int(height)
        self.max_regions = int(max_regions)
        self.point_stride = int(point_stride)
        self.handle = self.lib.motion_create(self.width, self.height, self.max_regions)
        if not self.handle:
            raise RuntimeError("Failed to create CUDA motion backend")

    def process(self, frame_bgr, learning_rate=0.0015, min_area=800):
        learning_rate = float(np.clip(learning_rate, 0.0, 0.05))
        if frame_bgr.dtype != np.uint8:
            raise ValueError("frame_bgr must be uint8")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be HxWx3 BGR")
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            raise ValueError("Frame size mismatch")

        frame_c = np.ascontiguousarray(frame_bgr)
        ptr = frame_c.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        ok = self.lib.motion_process(
            self.handle,
            ptr,
            frame_c.strides[0],
            ctypes.c_float(learning_rate),
            ctypes.c_int(min_area),
        )
        if ok == 0:
            raise RuntimeError("motion_process failed")

        mask = np.empty((self.height, self.width), dtype=np.uint8)
        self.lib.motion_download_mask(
            self.handle,
            mask.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        )

        n = self.lib.motion_get_blob_count(self.handle)
        n = max(0, min(n, self.max_regions))

        blob_arr = (BlobRegion * n)()
        got = self.lib.motion_get_blobs(self.handle, blob_arr, n)

        blobs = []
        for i in range(got):
            b = blob_arr[i]
            blobs.append({
                "label": int(b.label),
                "area": int(b.area),
                "x": int(b.x),
                "y": int(b.y),
                "w": int(b.w),
                "h": int(b.h),
                "cx": float(b.cx),
                "cy": float(b.cy),
            })

        return mask, blobs

    def analyze_objects(self, centers, min_mag=0.35, angle_thr_deg=35.0, mag_ratio_low=0.35):
        centers = list(centers)
        n = len(centers)
        if n == 0:
            return []

        centers_arr = (TrackCenter * n)(*[TrackCenter(int(x), int(y)) for x, y in centers])
        results_arr = (ObjectFlowResult * n)()
        points_arr = (FlowPoint * (n * self.point_stride))()

        got = self.lib.motion_analyze_objects(
            self.handle,
            centers_arr,
            ctypes.c_int(n),
            ctypes.c_float(min_mag),
            ctypes.c_float(angle_thr_deg),
            ctypes.c_float(mag_ratio_low),
            results_arr,
            points_arr,
            ctypes.c_int(self.point_stride),
        )

        if got <= 0:
            return []

        output = []
        for i in range(n):
            r = results_arr[i]
            pts = []
            base = i * self.point_stride
            for j in range(self.point_stride):
                p = points_arr[base + j]
                if p.mag <= 0:
                    continue
                pts.append({
                    "x0": int(p.x0),
                    "y0": int(p.y0),
                    "vx": float(p.vx),
                    "vy": float(p.vy),
                    "mag": float(p.mag),
                    "anomalous": bool(p.anomalous),
                })
            output.append({
                "majority_vx": float(r.majority_vx),
                "majority_vy": float(r.majority_vy),
                "majority_mag": float(r.majority_mag),
                "valid_count": int(r.valid_count),
                "anomaly_count": int(r.anomaly_count),
                "anomaly_ratio": float(r.anomaly_ratio),
                "kernel_box": (int(r.kernel_x1), int(r.kernel_y1), int(r.kernel_x2), int(r.kernel_y2)),
                "points": pts,
            })
        return output

    def close(self):
        if self.handle:
            self.lib.motion_destroy(self.handle)
            self.handle = None
