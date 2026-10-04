"""Regenerate the GIFs and figures in media/ from the original result videos.

    cd python && python make_media.py

Inputs: media/videos/headtracking-v1.0.mov and headtracking-v1.1.mov.
Outputs: media/gifs/*.gif and media/figures/*.png.
"""
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont

import colour_tracker as ct
import lucas_kanade as lk
from recover_track import recover

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
MEDIA = os.path.join(ROOT, 'media')

# Image area of the MATLAB figure inside the captured videos.
CROP = (slice(41, 365), slice(74, 507))

# Window of the original runs on the first frame: (row, col) centre, radius.
SEQUENCES = {
    'headtracking-v1.0': dict(start=(229, 388), radius=14,
                              title='Walking away from the camera'),
    'headtracking-v1.1': dict(start=(145, 202), radius=8,
                              title='Walking towards the camera'),
}

# Reference palette (dataviz skill): categorical slots 1-2, text tokens.
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK_2, MUTED = '#0b0b0b', '#52514e', '#c3c2b7'
PANEL_W = 380
FONT = os.path.join(matplotlib.get_data_path(), 'fonts/ttf/DejaVuSans.ttf')


def read_frames(name):
    cap = cv2.VideoCapture(os.path.join(MEDIA, 'videos', name + '.mov'))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            return frames
        frames.append(cv2.cvtColor(f[CROP], cv2.COLOR_BGR2RGB))


def drop_glitches(frames):
    """Remove frames where something else was captured (e.g. a dialog box)."""
    stack = np.stack(frames).astype(np.float32)
    d = np.abs(stack - np.median(stack, 0)).mean((1, 2, 3))
    return [f for f, di in zip(frames, d) if di <= 3 * np.median(d) + 10]


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def draw_mask_outline(img, mask, colour, scale):
    """Outline drawn just outside the window, with a white halo so it reads
    on any background without hiding the head."""
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    big = cv2.resize(grown, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 3 * scale, cv2.LINE_AA)
    cv2.drawContours(img, contours, -1, hex_rgb(colour), scale + 1, cv2.LINE_AA)


def panel(frame, mask, colour, title, lost, scale=2):
    # Upscale before drawing so the outline stays thin.
    img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if mask is not None and mask.any():
        draw_mask_outline(img, mask, colour, scale)
    img = cv2.resize(img, (PANEL_W, round(PANEL_W * img.shape[0] / img.shape[1])),
                     interpolation=cv2.INTER_AREA)
    img = Image.fromarray(img)
    head = Image.new('RGB', (img.width, 30), 'white')
    d = ImageDraw.Draw(head)
    d.rectangle([8, 9, 20, 21], fill=colour)
    d.text((28, 6), title, fill=INK, font=ImageFont.truetype(FONT, 15))
    if lost:
        f2 = ImageFont.truetype(FONT, 13)
        note = 'window lost'
        d.text((img.width - f2.getsize(note)[0] - 8, 8), note, fill=INK_2, font=f2)
    out = Image.new('RGB', (img.width, img.height + 30), 'white')
    out.paste(head, (0, 0))
    out.paste(img, (0, 30))
    return out


def stabilise(frames, threshold=20):
    """Snap pixels that match the static background to it, so that unchanged
    areas are identical from frame to frame (much smaller GIFs)."""
    smooth = [cv2.medianBlur(f, 3) for f in frames]
    background = np.median(np.stack(smooth), 0).astype(np.uint8)
    out = []
    for f in smooth:
        same = np.abs(f.astype(int) - background).max(-1) < threshold
        out.append(np.where(same[..., None], background, f))
    return out


def save_gif(images, path, ms):
    """GIF with one shared palette. With a stabilised background, Pillow only
    re-encodes the region that changed between consecutive frames."""
    pal = images[len(images) // 2].quantize(colors=61, method=Image.MEDIANCUT)
    extra = [c for h in (BLUE, ORANGE) for c in hex_rgb(h)] + [255, 255, 255]
    pal.putpalette(pal.getpalette()[:61 * 3] + extra + [0] * (192 * 3))
    q = [im.quantize(palette=pal, dither=0) for im in images]
    q[0].save(path, save_all=True, append_images=q[1:], duration=ms, loop=0)


def comparison_gif(name, cfg):
    frames = drop_glitches(read_frames(name))
    n, r = len(frames), cfg['radius']
    shape = frames[0].shape[:2]

    centres = recover(frames, cfg['start'], r)
    colour_masks = [ct.circular_roi(shape, c, r) for c in centres]

    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    lk_masks = lk.track(grey, ct.circular_roi(shape, cfg['start'], r))
    lk_masks += [None] * (n - len(lk_masks))

    shown = stabilise(frames)
    colour_panels, lk_panels = [], []
    for k, f in enumerate(shown):
        colour_panels.append(panel(f, colour_masks[k], BLUE, 'Colour signature', False))
        lost = lk_masks[k] is None or not lk_masks[k].any()
        lk_panels.append(panel(f, lk_masks[k], ORANGE, 'Lucas–Kanade', lost))
    # One GIF per tracker: a GIF only stores the region that changed since the
    # previous frame, which stays small when each file has a single panel.
    save_gif(colour_panels, os.path.join(MEDIA, 'gifs', name + '-colour.gif'), 400)
    save_gif(lk_panels, os.path.join(MEDIA, 'gifs', name + '-lk.gif'), 400)
    return frames, centres


def zoom(ax, centre, half):
    ax.set_xlim(centre[1] - half, centre[1] + half)
    ax.set_ylim(centre[0] + half, centre[0] - half)
    ax.set_xticks([])
    ax.set_yticks([])


def style(ax, title):
    ax.set_title(title, loc='left', fontsize=11, color=INK, pad=8)
    for s in ax.spines.values():
        s.set_color(MUTED)


def pipeline_figure(frames, centres, radius, k):
    """Four panels explaining one tracking step, from frame k to frame k+1."""
    f0, f1 = frames[k], frames[k + 1]
    y0, y1 = ct.rgb2ycbcr(f0), ct.rgb2ycbcr(f1)
    roi = ct.circular_roi(f0.shape[:2], centres[k], radius)
    signature = ct.cbcr_histogram(y0, roi)
    zone = ct.search_zone_for(roi)
    likelihood = ct.back_projection(y1, signature, zone)
    dx, dy, _ = ct.locate_new_roi(roi, signature, y1)
    new_centre = (centres[k][0] + dx, centres[k][1] + dy)
    zone_r = np.sqrt(zone.sum() / np.pi)
    half = zone_r * 1.25

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    plt.subplots_adjust(wspace=0.18, left=0.02, right=0.98, top=0.86, bottom=0.12)

    # 1. Selected window.
    ax = axes[0]
    ax.imshow(f0, interpolation='bicubic')
    ax.add_patch(Circle(centres[k][::-1], radius, fill=False, ec='white', lw=4))
    ax.add_patch(Circle(centres[k][::-1], radius, fill=False, ec=BLUE, lw=2))
    zoom(ax, centres[k], half)
    style(ax, '1  Window on the head')

    # 2. Colour signature in the CbCr plane, against the colours of the frame.
    ax = axes[1]
    ycc = y0.reshape(-1, 3)
    bg = np.unique(ycc[::7, 1:], axis=0)
    ax.scatter(bg[:, 0], bg[:, 1], s=4, c=MUTED, lw=0, label='whole frame')
    cb, cr = np.nonzero(signature)
    w = signature[cb, cr]
    rgb = cv2.cvtColor(np.stack([np.full_like(cb, 150), cr, cb], 1)
                       .astype(np.uint8)[None], cv2.COLOR_YCrCb2RGB)[0] / 255
    ax.scatter(cb, cr, s=10 + 90 * w / 255, c=rgb, ec=INK, lw=0.4,
               label='head (size = weight)')
    ax.set_xlim(95, 145)
    ax.set_ylim(115, 175)
    ax.set_xlabel('Cb (blue difference)', color=INK_2)
    ax.set_ylabel('Cr (red difference)', color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    style(ax, '2  Colour signature (CbCr histogram)')

    # 3. Back-projection on the next frame.
    ax = axes[2]
    ax.imshow(cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY), cmap='gray', vmin=0, vmax=400,
              interpolation='bicubic')
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'seq', ['#cde2fb', '#3987e5', '#0d366b'])
    ax.imshow(np.ma.masked_where(likelihood <= 0, likelihood), cmap=cmap,
              vmin=0, vmax=likelihood.max(), alpha=0.9)
    ax.add_patch(Circle(centres[k][::-1], zone_r, fill=False, ec='white', lw=1.2, ls='--'))
    ax.text(0.02, 0.03, 'dashed circle: search zone\ndarker blue: better match',
            transform=ax.transAxes, fontsize=8, color=INK,
            bbox=dict(fc='white', ec='none', alpha=0.85))
    zoom(ax, centres[k], half)
    style(ax, '3  Pixels matching the signature')

    # 4. Window moved to the maximum concentration.
    ax = axes[3]
    ax.imshow(f1, interpolation='bicubic')
    ax.add_patch(Circle(centres[k][::-1], radius, fill=False, ec='white', lw=1.5, ls='--'))
    ax.add_patch(Circle(new_centre[::-1], radius, fill=False, ec='white', lw=4))
    ax.add_patch(Circle(new_centre[::-1], radius, fill=False, ec=BLUE, lw=2))
    if dx or dy:
        ax.add_patch(FancyArrowPatch(centres[k][::-1], new_centre[::-1],
                                     arrowstyle='-|>', mutation_scale=12,
                                     color='white', lw=1.5))
    zoom(ax, centres[k], half)
    style(ax, '4  Move to the best match')
    ax.text(0.02, 0.03, 'dashed: previous position', transform=ax.transAxes,
            fontsize=8, color=INK, bbox=dict(fc='white', ec='none', alpha=0.85))

    fig.savefig(os.path.join(MEDIA, 'figures', 'pipeline.png'), dpi=110,
                facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(os.path.join(MEDIA, 'gifs'), exist_ok=True)
    os.makedirs(os.path.join(MEDIA, 'figures'), exist_ok=True)
    results = {name: comparison_gif(name, cfg) for name, cfg in SEQUENCES.items()}
    frames, centres = results['headtracking-v1.1']
    pipeline_figure(frames, centres, SEQUENCES['headtracking-v1.1']['radius'], 30)
