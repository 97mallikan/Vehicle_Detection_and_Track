# road_direction.py
import numpy as np

class RoadDirectionEstimator:
    def __init__(self, ema=0.92, min_speed=0.8):
        self.u = None
        self.ema = float(ema)
        self.min_speed = float(min_speed)

    def update(self, tracks, min_hits=5):
        vs = []
        for t in tracks:
            if t.hits < min_hits:
                continue
            vx = float(t.kf.statePost[2, 0])
            vy = float(t.kf.statePost[3, 0])
            sp = float(np.hypot(vx, vy))
            if sp < self.min_speed:
                continue
            vs.append([vx / (sp + 1e-6), vy / (sp + 1e-6)])

        if not vs:
            return self.u

        vmean = np.mean(np.array(vs, dtype=np.float32), axis=0)
        n = float(np.linalg.norm(vmean))
        if n < 1e-6:
            return self.u

        u_new = (vmean / n).astype(np.float32)

        if self.u is not None and float(np.dot(self.u, u_new)) < 0:
            u_new = -u_new

        if self.u is None:
            self.u = u_new
        else:
            u = self.ema * self.u + (1.0 - self.ema) * u_new
            self.u = (u / (np.linalg.norm(u) + 1e-6)).astype(np.float32)

        return self.u