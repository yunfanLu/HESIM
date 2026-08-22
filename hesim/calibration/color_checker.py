import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

COLOR_NAMES = [
    ["1E CardWhite", "1F PrimaryCyan", "1G Orange", "1H Aqua"],
    ["2E 20%Gray", "2F Magenta", "2G Blueprint", "2H Lavender"],
    ["3E 40%Gray", "3F Yellow", "3G Pink", "3H Evergreen"],
    ["4E 60%Gray", "4F Red", "4G Violet", "4H SteelBlue"],
    ["5E 80%Gray", "5F Green", "5G AppleGreen", "5H LightSkin"],
    ["6E Black", "6F Blue", "6G Sunflower", "6H DarkSkin"],
]

SRGB_REF = np.array(
    [
        [[247, 242, 237], [39, 126, 157], [198, 117, 44], [130, 186, 164]],
        [[199, 196, 193], [167, 76, 141], [70, 89, 156], [125, 124, 171]],
        [[158, 156, 153], [234, 204, 37], [170, 80, 94], [90, 106, 57]],
        [[120, 118, 115], [159, 32, 53], [78, 61, 104], [98, 119, 152]],
        [[81, 81, 79], [94, 145, 71], [165, 186, 69], [183, 144, 125]],
        [[46, 46, 47], [41, 58, 134], [218, 157, 46], [103, 77, 63]],
    ],
    dtype=np.uint8,
)

GRAY_INDICES = [0, 4, 8, 12, 16, 20]
GREEN16 = np.array([2, 3, 6, 7, 8, 9, 12, 13])


def get_checker_centers_homography(vertex_pts, grid_shape=(6, 4)):
    """
    Compute 24-patch centres for a ColorChecker placed under arbitrary perspective.

    Parameters
    ----------
    vertex_pts : (4,2) array-like
        Four corner points from JSON in the *given* order:
        [6H (brown), 1H (black), 1E (white), 6E (cyan)]
        i.e.  bottom-right, top-right, top-left, bottom-left.
        Values are (row, col) = (y, x) in RAW image space.
    grid_shape : tuple (rows, cols)
        Default (6, 4) for X-Rite 24-patch chart.

    Returns
    -------
    centres : (24,2) ndarray float32
        Patch centres in RAW coordinate, row-major order
        [(row0,col0), (row0,col1), … (row5,col3)].
    """
    rows, cols = grid_shape

    v = np.asarray(vertex_pts, np.float32)[:, ::-1]

    tl, tr, br, bl = v[[2, 1, 0, 3]]

    src = np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]], dtype=np.float32)
    dst = np.vstack([tl, tr, br, bl])

    H = cv2.getPerspectiveTransform(src, dst)

    centres = []
    for r in range(rows):
        for c in range(cols):
            src_pt = np.array([c + 0.5, r + 0.5, 1.0], np.float32)
            x, y, w = H @ src_pt
            centres.append([y / w, x / w])
    return np.asarray(centres, np.float32)


def get_checker_blocks_homography(vertex_pts_rc, grid_shape=(6, 4), roi_scale=0.6):

    def warp_xy(xy):
        """vectorised warp; xy shape (...,2)->(...,2)"""
        xy_h = np.concatenate([xy, np.ones_like(xy[..., :1])], -1)
        xyz = xy_h @ H.T
        return xyz[..., :2] / xyz[..., 2:]

    rows, cols = grid_shape
    v_xy = np.asarray(vertex_pts_rc, np.float32)[:, ::-1]
    tl_xy, tr_xy, br_xy, bl_xy = v_xy[[2, 1, 0, 3]]
    src = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    dst = np.vstack([tl_xy, tr_xy, br_xy, bl_xy])
    H = cv2.getPerspectiveTransform(src, dst)
    offset = (1 - roi_scale) / 2
    roi_grid = np.array(
        [[offset, offset], [1 - offset, offset], [1 - offset, 1 - offset], [offset, 1 - offset]], np.float32
    )
    blocks_rc, centres_rc = [], []
    for r in range(rows):
        for c in range(cols):
            base = np.array([c, r], np.float32)
            roi_xy = warp_xy(base + roi_grid)
            centre = warp_xy(base + 0.5)
            blocks_rc.append(roi_xy[:, ::-1])
            centres_rc.append(centre[::-1])
    return (np.stack(blocks_rc).astype(np.float32), np.vstack(centres_rc).astype(np.float32))


def extract_checker_colors_raw(raw_img, centers, block_height, block_width, roi_scale=0.4):
    h, w = raw_img.shape
    roi_h = int(block_height * roi_scale / 2)
    roi_w = int(block_width * roi_scale / 2)
    vals = []
    for x, y in centers:
        x, y = int(round(x)), int(round(y))
        x1, x2 = max(0, x - roi_w), min(w, x + roi_w + 1)
        y1, y2 = max(0, y - roi_h), min(h, y + roi_h + 1)
        roi = raw_img[y1:y2, x1:x2]
        vals.append(np.mean(roi))
    return np.array(vals)


def load_checker_json(json_file):
    """
    Read 4 checkerboard corner points from LabelMe-style JSON and convert
    to RAW image coordinate (row, col) = (y, x).

    Returns
    -------
    coords : ndarray (4,2)  order: (row, col) for
             ['brown', 'cyan', 'white', 'black']
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    corner_labels = ["brown", "cyan", "white", "black"]
    coords = []
    for lbl in corner_labels:
        for s in data["shapes"]:
            if s["label"] == lbl and s["points"]:
                x_json, y_json = s["points"][0]
                coords.append([x_json, y_json])
                break
        else:
            raise ValueError(f"{lbl} not found in {json_file}")
    return np.asarray(coords, dtype=np.float32)


def _qb16_map(h, w):
    rr = np.arange(h)[:, None] % 4
    cc = np.arange(w)[None, :] % 4
    idx = rr * 4 + cc
    return idx.astype(np.uint8)


def extract_qb16_means(raw, quad_rc, idx_map=None):
    h, w = raw.shape
    if idx_map is None:
        idx_map = _qb16_map(h, w)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, quad_rc[:, ::-1].astype(np.int32), 1)

    vals = np.full(16, np.nan, np.float32)
    for i in range(16):
        pix = raw[(mask == 1) & (idx_map == i)]
        if pix.size:
            vals[i] = pix.mean(dtype=np.float32)
    return vals


def extract_qb16(raw, quad_rc, idx_map=None):
    h, w = raw.shape
    if idx_map is None:
        idx_map = _qb16_map(h, w)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, quad_rc[:, ::-1].astype(np.int32), 1)

    pix = [np.array([], dtype=np.float32) for _ in range(16)]
    for i in range(16):
        pix_i = raw[(mask == 1) & (idx_map == i)]
        if pix_i.size:
            pix[i] = pix_i.astype(np.float32)
    return pix


def _quad_bayer_map(h, w):
    """
    GGRR  1100
    GGRR  1100
    BBGG  3322
    BBGG  3322
    """
    rr = np.arange(h)[:, None] % 4
    cc = np.arange(w)[None, :] % 4
    m = np.empty((h, w), np.uint8)

    m[(rr < 2) & (cc < 2)] = 1
    m[(rr < 2) & (cc >= 2)] = 0
    m[(rr >= 2) & (cc >= 2)] = 2
    m[(rr >= 2) & (cc < 2)] = 3
    return m


def extract_rgb_from_quad(raw, quad_rc):
    h, w = raw.shape
    chan_map = _quad_bayer_map(h, w)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, quad_rc[:, ::-1].astype(np.int32), 1)

    sel_R = (mask == 1) & (chan_map == 0)
    sel_G1 = (mask == 1) & (chan_map == 1)
    sel_G2 = (mask == 1) & (chan_map == 2)
    sel_B = (mask == 1) & (chan_map == 3)

    R = raw[sel_R].astype(np.float32)
    G1 = raw[sel_G1].astype(np.float32)
    G2 = raw[sel_G2].astype(np.float32)
    B = raw[sel_B].astype(np.float32)

    return R, G1, B, G2


def extract_rgb_from_rgb(img_rgb, quad_rc):
    """
    Average the R-, G-, B- channel values inside an arbitrary 4-point ROI.

    Parameters
    ----------
    img_rgb : (H, W, 3) float32 RGB image.
    quad_rc : (4, 2)   ROI vertices in (row, col) = (y, x) order.

    Returns
    -------
    R_vals, G_vals, B_vals : 1-D float arrays of all pixels inside ROI.
    """
    h, w, _ = img_rgb.shape
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, quad_rc[:, ::-1].astype(np.int32), 1)

    R_vals = img_rgb[:, :, 0][mask == 1]
    G_vals = img_rgb[:, :, 1][mask == 1]
    B_vals = img_rgb[:, :, 2][mask == 1]
    return R_vals.astype(np.float32), G_vals.astype(np.float32), B_vals.astype(np.float32)
