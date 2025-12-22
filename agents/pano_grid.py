# pano_grid.py
# Helpers to tile a spherical pano's "napkin ring" (top/bottom trims)
# and build Street View Static API URLs for fixed N = X*Y tiles.

import math
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


def _rad(x): return x * math.pi / 180.0
def _deg(x): return x * 180.0 / math.pi


def pano_band_limits(Tup_deg: float, Tdown_deg: float) -> tuple[float, float]:
    """Return (lambda_min, lambda_max) in radians after trimming polar caps."""
    lam_min = -math.pi / 2 + _rad(Tdown_deg)
    lam_max =  math.pi / 2 - _rad(Tup_deg)
    if not (lam_min < lam_max):
        raise ValueError("Invalid trims: T_up + T_down >= 180° or negative.")
    return lam_min, lam_max


def pano_centers(
    X: int, Y: int,
    Tup_deg: float, Tdown_deg: float,
    *, equal_area: bool = False,
    yaw0_deg: float = 0.0
) -> tuple[list[float], list[float]]:
    """
    Returns (headings_deg, pitches_deg) for centers.
    - headings_deg length = X, pitches_deg length = Y
    - yaw0_deg rotates the apple slices (0° = North).
    """
    lam_min, lam_max = pano_band_limits(Tup_deg, Tdown_deg)

    # headings (yaw): X equal slices around 360°
    d_yaw = 360.0 / X
    yaw0 = yaw0_deg % 360.0
    headings = [(yaw0 + (i + 0.5) * d_yaw) % 360.0 for i in range(X)]

    # pitches (latitude): equal-angle or equal-area
    if not equal_area:
        dlam = (lam_max - lam_min) / Y
        lats = [lam_min + (j + 0.5) * dlam for j in range(Y)]
    else:
        smin, smax = math.sin(lam_min), math.sin(lam_max)
        ds = (smax - smin) / Y
        lats = [math.asin(smin + (j + 0.5) * ds) for j in range(Y)]

    pitches = [_deg(lam) for lam in lats]   # Street View 'pitch' in degrees
    return headings, pitches


def tile_hfov_deg(
    X: int, Y: int,
    Tup_deg: float, Tdown_deg: float,
    *, size: tuple[int, int] = (512, 512),
    equal_area: bool = False,
    overlap_frac: float = 0.02
) -> float:
    """
    Choose one horizontal FOV (deg) so each tile covers its Δlon and Δlat
    given the output aspect (rectilinear pinhole model).
    """
    W, H = size
    aspect = W / H

    # add a small horizontal overlap so columns cover 360° without gaps
    dlon = (360.0 / X) * (1.0 + max(0.0, overlap_frac))  # deg

    lam_min, lam_max = pano_band_limits(Tup_deg, Tdown_deg)
    # compute vertical per-row angular requirement in degrees
    if not equal_area:
        dlam_deg = _deg((lam_max - lam_min) / Y)  # uniform in angle-space
    else:
        # equal-area rows are uniform in sin(lambda); angle spans vary by row
        smin, smax = math.sin(lam_min), math.sin(lam_max)
        ds = (smax - smin) / Y
        # worst-case (max) angular height across all rows
        dlam_max = 0.0
        for j in range(Y):
            s_lo = smin + j * ds
            s_hi = smin + (j + 1) * ds
            # clamp numerically into [-1,1]
            s_lo = max(-1.0, min(1.0, s_lo))
            s_hi = max(-1.0, min(1.0, s_hi))
            lam_lo = math.asin(s_lo)
            lam_hi = math.asin(s_hi)
            dlam_max = max(dlam_max, lam_hi - lam_lo)
        dlam_deg = _deg(dlam_max)

    # vertical_FOV(h) = 2 * atan( tan(h/2) / aspect )
    # require vertical_FOV >= dlam_deg → h_for_v = 2*atan(aspect * tan(dlam/2))
    h_for_v = _deg(2.0 * math.atan(aspect * math.tan(_rad(dlam_deg) / 2.0)))

    hfov = max(dlon, h_for_v)
    return min(120.0, max(10.0, hfov))  # clamp typical SV bounds


def set_params(base_url: str, **updates) -> str:
    """Return base_url with updated query params."""
    pu = urlparse(base_url)
    q = {k: v[0] for k, v in parse_qs(pu.query).items()}
    for k, v in updates.items():
        q[k] = str(v)
    return urlunparse(pu._replace(query=urlencode(q)))


def make_base_url_from_coords(lat: float, lng: float, api_key: str, size=(512, 512)) -> str:
    """
    Build a Street View Static API URL rooted at coordinates.
    Note: using `location=lat,lng` lets Google pick the nearest pano.
    """
    W, H = size
    return (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size={W}x{H}&location={lat:.8f},{lng:.8f}&key={api_key}"
    )


def build_fixed_grid_urls(
    base_url: str,
    X: int, Y: int,
    Tup_deg: float = 15.0, Tdown_deg: float = 15.0,
    *, size: tuple[int, int] = (512, 512),
    equal_area: bool = False,
    yaw0_deg: float = 0.0
) -> list[str]:
    """
    Return N = X*Y URLs, each centered on a tile center with a single FOV
    so tiles cover at least their intended angular extents.
    Works with base_url that already has either `pano=` or `location=lat,lng`.
    """
    W, H = size
    headings, pitches = pano_centers(X, Y, Tup_deg, Tdown_deg,
                                     equal_area=equal_area, yaw0_deg=yaw0_deg)
    hfov = tile_hfov_deg(X, Y, Tup_deg, Tdown_deg, size=size, equal_area=equal_area)

    urls = []
    for j, pitch in enumerate(pitches):
        pitch = max(-90.0, min(90.0, pitch))
        for i, heading in enumerate(headings):
            urls.append(
                set_params(
                    base_url,
                    heading=round(heading, 6),
                    pitch=round(pitch,   6),
                    fov=round(hfov,      6),
                    size=f"{W}x{H}",
                )
            )
    return urls
