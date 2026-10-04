"""Recover the window positions of the original 2006 runs from the result videos.

The MATLAB tracker drew its window as a 1-pixel circle on the 1024x768
frames. The result videos are screen captures of a figure in which those
frames were shrunk to ~433x324 and JPEG-compressed, so only a faint dotted
ring survives. The ring has a fixed radius and moves with the window, so it
can be found with a ring-shaped matched filter on high-pass luminance after
removing the static background, plus a smoothness constraint across frames
(Viterbi over the best candidates of each frame).
"""
import cv2
import numpy as np


def _ring_kernel(radius):
    s = int(radius + 3)
    yy, xx = np.mgrid[-s:s + 1, -s:s + 1]
    k = (np.abs(np.hypot(yy, xx) - radius) <= 1.0).astype(np.float32)
    return k / k.sum()


def _ring_response(frames, radius):
    lum = np.stack(frames).astype(np.float32).mean(-1)
    high = np.stack([l - cv2.medianBlur(l.astype(np.uint8), 5) for l in lum])
    high = np.clip(high - np.median(high, axis=0), 0, None)  # drop static texture
    k = _ring_kernel(radius)
    return np.stack([cv2.filter2D(h, -1, k) for h in high])


def _candidates(resp, k=40, nms=6):
    r = resp.copy()
    out = []
    for _ in range(k):
        y, x = np.unravel_index(r.argmax(), r.shape)
        out.append((y, x, r[y, x]))
        r[max(0, y - nms):y + nms + 1, max(0, x - nms):x + nms + 1] = -np.inf
    return out


def recover(frames, start, radius, motion_cost=0.15, max_jump=45):
    """Return the (row, col) centre of the original window in every frame.

    ``start`` is the centre of the window in the first frame and ``radius``
    its radius, both read off the first frame of the video.
    """
    cands = []
    for resp in _ring_response(frames, radius):
        c = _candidates(resp)
        s = np.array([v for _, _, v in c])
        s = (s - s.mean()) / (s.std() + 1e-6)
        cands.append([(y, x, si) for (y, x, _), si in zip(c, s)])
    cands[0] = [(start[0], start[1], 0.0)]

    score, back = np.zeros(1), [None]
    for prev, cur in zip(cands[:-1], cands[1:]):
        p = np.array([c[:2] for c in prev], float)
        q = np.array([c[:2] for c in cur], float)
        d = np.linalg.norm(q[:, None, :] - p[None, :, :], axis=2)
        total = score[None, :] - motion_cost * d - 1e6 * (d > max_jump)
        back.append(total.argmax(1))
        score = total.max(1) + np.array([c[2] for c in cur])

    i = int(score.argmax())
    path = []
    for k in range(len(cands) - 1, -1, -1):
        path.append(cands[k][i][:2])
        if k:
            i = back[k][i]
    return path[::-1]
