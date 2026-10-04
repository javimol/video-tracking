"""Python port of the Lucas-Kanade baseline in ``lucas-kanade/lk.m``.

Translation-only Lucas-Kanade on a user-selected region: for every pair of
consecutive frames, iterate Gauss-Newton steps until the update is below
``eps`` pixels or ``max_iterations`` is reached, then move the region by the
rounded displacement.
"""
import cv2
import numpy as np
from scipy.signal import convolve2d


def gauss(sig):
    x = np.arange(np.floor(-3 * sig), np.ceil(3 * sig) + 1)
    g = np.exp(-0.5 * x ** 2 / sig ** 2)
    return g / g.sum()


def dgauss(sig):
    x = np.arange(np.floor(-3 * sig), np.ceil(3 * sig) + 1)
    return -x * gauss(sig) / sig ** 2


def estimate_shift(F, G, roi, sig=1.5, max_iterations=15, eps=0.01):
    """Displacement (u, v) in (rows, cols) such that F shifted by it ~ G."""
    g, dg = gauss(sig), dgauss(sig)
    rows, cols = F.shape
    u = v = 0.0
    for _ in range(max_iterations):
        Fx = convolve2d(F, np.outer(dg, g), mode='same')  # d/d row
        Fy = convolve2d(F, np.outer(g, dg), mode='same')  # d/d col
        fx, fy, ft = Fx[roi], Fy[roi], (G - F)[roi]
        C = np.array([[fx @ fx, fx @ fy], [fx @ fy, fy @ fy]])
        if np.linalg.matrix_rank(C) < 2:
            break
        du, dv = np.linalg.solve(C, [ft @ fx, ft @ fy])
        u, v = u + du, v + dv
        # warp F by the update (MATLAB: interp2 with meshgrid(c+dv, r+du))
        X, Y = np.meshgrid(np.arange(cols) + dv, np.arange(rows) + du)
        F = cv2.remap(F.astype(np.float32), X.astype(np.float32),
                      Y.astype(np.float32), cv2.INTER_CUBIC,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if abs(du) <= eps and abs(dv) <= eps:
            break
    return u, v


def track(frames_gray, roi):
    """Track a boolean region through grey-level frames. Returns one mask per frame."""
    from colour_tracker import shift_roi
    rois = [roi]
    for F, G in zip(frames_gray[:-1], frames_gray[1:]):
        u, v = estimate_shift(F.astype(np.float64), G.astype(np.float64), roi)
        roi = shift_roi(roi, -int(round(u)), -int(round(v)))
        rois.append(roi)
        if not roi.any():
            break
    return rois
