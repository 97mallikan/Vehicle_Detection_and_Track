# cuda_backend.py
import ctypes
import numpy as np
import cv2


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


class CudaMotionBackend:
    def __init__(self, lib_path, width, height, max_regions=2048):
        self.lib = ctypes.CDLL(lib_path)

        self.lib.motion_create.restype = ctypes.c_void_p
        self.lib.motion_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.motion_destroy.restype = None
        self.lib.motion_destroy.argtypes = [ctypes.c_void_p]

        self.lib.motion_process.restype = ctypes.c_int
        self.lib.motion_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),  # frame BGR
            ctypes.c_int,                    # step
            ctypes.c_float,                  # learning rate
            ctypes.c_int                     # min area
        ]

        self.lib.motion_download_mask.restype = None
        self.lib.motion_download_mask.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte)
        ]

        self.lib.motion_get_blob_count.restype = ctypes.c_int
        self.lib.motion_get_blob_count.argtypes = [ctypes.c_void_p]

        self.lib.motion_get_blobs.restype = ctypes.c_int
        self.lib.motion_get_blobs.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(BlobRegion),
            ctypes.c_int
        ]

        self.width = int(width)
        self.height = int(height)
        self.max_regions = int(max_regions)
        self.handle = self.lib.motion_create(self.width, self.height, self.max_regions)

        if not self.handle:
            raise RuntimeError("Failed to create CUDA motion backend")

    def process(self, frame_bgr, learning_rate=0.0015, min_area=800):
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
            ctypes.c_int(min_area)
        )
        if ok == 0:
            raise RuntimeError("motion_process failed")

        mask = np.empty((self.height, self.width), dtype=np.uint8)
        self.lib.motion_download_mask(
            self.handle,
            mask.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
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

    def close(self):
        if self.handle:
            self.lib.motion_destroy(self.handle)
            self.handle = None