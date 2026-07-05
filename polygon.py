# polygon.py
import cv2
import numpy as np

class PolygonZone:
    def __init__(self):
        self.points = []           # list[(x,y)]
        self.violated = set()      # track ids

    def mouse_cb(self, event, x, y, flags, pol_points):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Point selected: {x}, {y}")
            if len(self.points) == pol_points:
                print("Polygon ready.")
            elif len(self.points) > pol_points:
                print("Reset polygon points.")
                self.points = []

    def ready(self, pol_points):
        return len(self.points) == pol_points

    def poly_np(self, pol_points):
        if not self.ready(pol_points):
            return None
        return np.array(self.points, dtype=np.int32)

    @staticmethod
    def inside(x, y, poly_pts_int):
        # >=0 means inside/on edge
        return cv2.pointPolygonTest(poly_pts_int, (float(x), float(y)), False) >= 0

    def future_any_inside(self, fut_pts, poly_pts_int):
        for fx, fy in fut_pts:
            if self.inside(fx, fy, poly_pts_int):
                return True
        return False

    def reset(self):
        self.points.clear()
        self.violated.clear()