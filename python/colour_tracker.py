"""Python port of the colour-signature head tracker in ``src/``.

Each function mirrors one MATLAB file so the two versions can be read side
by side. Images are uint8 RGB arrays (H x W x 3); coordinates are
(row, col) as in MATLAB, but 0-based.
"""
import numpy as np
from scipy.optimize import minimize


def rgb2ycbcr(rgb):
    """MATLAB's rgb2ycbcr for uint8 images (ITU-R BT.601, studio range)."""
    x = rgb.astype(np.float64) / 255.0
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112.0],
                  [112.0, -93.786, -18.214]])
    ycc = x @ m.T + np.array([16.0, 128.0, 128.0])
    return np.clip(np.round(ycc), 0, 255).astype(np.uint8)


def circular_roi(shape, centre, radius):
    rows, cols = np.mgrid[:shape[0], :shape[1]]
    return np.hypot(rows - centre[0], cols - centre[1]) <= radius


def cbcr_histogram(ycbcr, roi):
    """Luminance-weighted CbCr histogram, normalised so its peak is 255.

    Pixels with 60 < Y < 190 count 1, the rest 0.25 (cbcrHistogram.m).
    """
    y, cb, cr = (ycbcr[..., k][roi].astype(int) for k in range(3))
    weights = np.where((y > 60) & (y < 190), 1.0, 0.25)
    hist = np.zeros((256, 256))
    np.add.at(hist, (cb, cr), weights)
    peak = hist.max()
    return 255.0 * hist / peak if peak > 0 else hist


def shift_roi(roi, dx, dy):
    """Shift a boolean mask by (dx rows, dy cols), clipping at the border."""
    out = np.zeros_like(roi)
    i, j = np.nonzero(roi)
    i, j = i + dx, j + dy
    keep = (i >= 0) & (i < roi.shape[0]) & (j >= 0) & (j < roi.shape[1])
    out[i[keep], j[keep]] = True
    return out


def back_projection(ycbcr, signature, search_zone):
    """Replace each pixel of the search zone by the signature of its (Cb, Cr)."""
    likelihood = signature[ycbcr[..., 1], ycbcr[..., 2]]
    return np.where(search_zone, likelihood, 0.0)


def concentration(roi, likelihood, dx, dy):
    """Mean likelihood inside the displaced window, in [0, 1]."""
    shifted = shift_roi(roi, int(round(dx)), int(round(dy)))
    return min(1.0, likelihood[shifted].sum() / (255.0 * roi.sum()))


def search_zone_for(roi):
    """Circle centred on the window, three window diagonals across (trackROI.m)."""
    i, j = np.nonzero(roi)
    diameter = 3 * np.floor(np.hypot(i.max() - i.min(), j.max() - j.min()))
    return circular_roi(roi.shape, (i.mean(), j.mean()), diameter / 2)


def locate_new_roi(roi, signature, ycbcr):
    """Displacement (dx, dy) maximising the concentration (locateNewROI.m).

    Nelder-Mead starting at (0, 0) with a one-pixel initial simplex, which is
    what the MATLAB version obtains from fminsearch through variable scaling.
    """
    likelihood = back_projection(ycbcr, signature, search_zone_for(roi))
    res = minimize(lambda d: -concentration(roi, likelihood, d[0], d[1]),
                   x0=[0.0, 0.0], method='Nelder-Mead',
                   options={'initial_simplex': [[0, 0], [1, 0], [0, 1]],
                            'xatol': 0.4, 'fatol': 1e-4})
    return int(round(res.x[0])), int(round(res.x[1])), likelihood


def track(frames_rgb, centre, radius, update_weight=0.5):
    """Track a circular window through a list of RGB frames (headtrack.m).

    Returns the list of window masks, one per frame, and per-frame details
    (back-projection likelihood) for plotting.
    """
    ycbcr = rgb2ycbcr(frames_rgb[0])
    roi = circular_roi(ycbcr.shape[:2], centre, radius)
    signature = cbcr_histogram(ycbcr, roi)

    rois, likelihoods = [roi], [None]
    for rgb in frames_rgb[1:]:
        ycbcr = rgb2ycbcr(rgb)
        dx, dy, likelihood = locate_new_roi(roi, signature, ycbcr)
        roi = shift_roi(roi, dx, dy)
        signature = ((1 - update_weight) * signature
                     + update_weight * cbcr_histogram(ycbcr, roi))
        rois.append(roi)
        likelihoods.append(likelihood)
    return rois, likelihoods
