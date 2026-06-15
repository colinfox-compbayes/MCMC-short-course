"""
cells_render.py

One-file utility for Compute 4 "cells" example.

What it does
------------
You can either:

(A) PASS a latent state (ordered marked point pattern)
    latent_x = [(x1,y1,m1), ..., (xn,yn,mn)]  (BACK-to-FRONT order)
    where m=0 is "bad" (white core), m=1 is "good" (black core).
    -> It will render the noiseless pixel array f(latent_x)
    -> Optionally add Gaussian noise
    -> Save:
         - raw array (.npy) UNCLIPPED
         - clipped PNG preview (.png)

(B) GENERATE its own synthetic image like makefake.m
    -> It will sample n, positions, marks, draw cells, add Gaussian noise
    -> It ALSO returns/saves the corresponding latent_x that created the image.

Notes
-----
- Coordinates are 1-based (MATLAB-style): x,y in {1,...,npix}.
- Order matters: the list is BACK-to-FRONT (later cells occlude earlier ones).
- Rendering matches the MATLAB meshgrid + linear indexing behaviour (incl. x/y swap),
  using Fortran-order flatten/reshape.

Dependencies
------------
- numpy
- pillow (for PNG saving): pip install pillow
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, List

import numpy as np

# -----------------------------
# Types
# -----------------------------

# One cell: (x_center, y_center, mark) with 1-based pixel coords
# mark: 0 = "bad" (white core), 1 = "good" (black core)
Cell = Tuple[int, int, int]


# -----------------------------
# Parameters
# -----------------------------

@dataclass(frozen=True)
class RenderParams:
    """
    Rendering parameters for converting a latent marked point pattern into
    a noiseless grayscale image.

    Attributes
    ----------
    npix:
        Image dimension. The output image has shape (npix, npix).
    cellrad:
        Outer radius of each cell (in pixel units, using MATLAB-style coordinates).
    ring_value:
        Intensity assigned to the outer disk/ring region of the cell.
    good_core:
        Intensity assigned to the inner core for a "good" cell (mark m=1).
    bad_core:
        Intensity assigned to the inner core for a "bad" cell (mark m=0).
    background:
        Background intensity used to initialize the image before drawing cells.
    """
    npix: int = 100          # image is npix x npix
    cellrad: float = 9.5     # outer radius
    ring_value: float = 0.5  # grey ring / disk
    good_core: float = 0.0   # black core
    bad_core: float = 1.0    # white core
    background: float = 1.0  # white background


@dataclass(frozen=True)
class MakeFakeParams:
    """
    Parameters matching makefake.m behaviour (plus render params).

    This dataclass bundles:
    - image geometry (npix)
    - cell count control (ncellmax, minimum of 5 is hard-coded in logic)
    - drawing parameters (cellrad and intensity values)
    - noise level (stddev)
    - mark probability (pbad ~ Uniform[pbad_low, pbad_low + pbad_range])

    Attributes
    ----------
    npix:
        Image dimension. Output image has shape (npix, npix).
    ncellmax:
        Maximum number of cells (minimum is 5 by construction in the code).
    cellrad:
        Outer radius of each cell.
    stddev:
        Standard deviation of the additive i.i.d. Gaussian noise.
    pbad_low:
        Lower bound for the uniform draw of pbad.
    pbad_range:
        Range length for the uniform draw of pbad (so upper = pbad_low + pbad_range).
    ring_value:
        Intensity for the outer disk/ring.
    good_core:
        Inner core intensity for good cells (m=1).
    bad_core:
        Inner core intensity for bad cells (m=0).
    background:
        Background intensity.
    """
    npix: int = 100
    ncellmax: int = 10
    cellrad: float = 9.5
    stddev: float = 0.1      # makefake.m draws pbad uniformly from [0.25, 0.75]

    pbad_low: float = 0.25
    pbad_range: float = 0.5

    # pixel intensities (same as RenderParams defaults)
    ring_value: float = 0.5
    good_core: float = 0.0
    bad_core: float = 1.0
    background: float = 1.0


# -----------------------------
# MATLAB-faithful drawing
# -----------------------------

# =====================================================================
# Function: _apply_cell
# =====================================================================
def _apply_cell(
    a: np.ndarray,
    x: int,
    y: int,
    r: float,
    outer_value: float,
    inner_value: float,
) -> np.ndarray:
    """
    Draw a single cell onto an image array in a MATLAB-faithful way.

    This is the shared core for `put_good` and `put_bad`. It matches MATLAB's
    meshgrid + find + linear indexing behaviour, including:
      - meshgrid output shape (n, m) when called as meshgrid(1:m, 1:n)
      - the x/y swap used in the distance formula
      - column-major ("Fortran-order") linear indexing for assignment

    MATLAB reference:
        [mm,nn] = meshgrid(1:m, 1:n);   % gives (n x m)
        a(find((mm - y).^2 + (nn - x).^2 <= r^2)) = outer_value;
        a(find((mm - y).^2 + (nn - x).^2 <= (r/2)^2)) = inner_value;

    Parameters
    ----------
    a:
        Input image array of shape (m, n), interpreted as (rows, cols).
        The returned array is a modified copy with the cell applied.
    x:
        1-based x-coordinate of the cell center (MATLAB-style).
    y:
        1-based y-coordinate of the cell center (MATLAB-style).
    r:
        Outer radius of the cell.
    outer_value:
        Intensity assigned to pixels within the outer radius (the ring/disk).
    inner_value:
        Intensity assigned to pixels within the inner radius (r/2) (the core).

    Returns
    -------
    np.ndarray
        A new array of shape (m, n) containing the updated image with the cell drawn.
    """
    m, n = a.shape  # rows, cols

    xs = np.arange(1, m + 1)   # 1..m
    ys = np.arange(1, n + 1)   # 1..n
    mm, nn = np.meshgrid(xs, ys)  # default indexing='xy' -> shape (n, m)

    outer_mask = (mm - y) ** 2 + (nn - x) ** 2 <= r ** 2
    inner_mask = (mm - y) ** 2 + (nn - x) ** 2 <= (r / 2) ** 2

    aF = np.array(a, order="F", copy=True).ravel(order="F")
    aF[outer_mask.ravel(order="F")] = outer_value
    aF[inner_mask.ravel(order="F")] = inner_value

    return aF.reshape((m, n), order="F")


# =====================================================================
# Function: put_good
# =====================================================================
def put_good(a: np.ndarray, x: int, y: int, r: float, ring_value: float, good_core: float) -> np.ndarray:
    """
    Draw a "good" cell (grey ring/disk with a dark/black core) onto an image.

    Parameters
    ----------
    a:
        Input image array of shape (m, n).
    x:
        1-based x-coordinate of the cell center.
    y:
        1-based y-coordinate of the cell center.
    r:
        Outer radius of the cell.
    ring_value:
        Intensity assigned to the outer disk/ring region.
    good_core:
        Intensity assigned to the inner core region (radius r/2).

    Returns
    -------
    np.ndarray
        Updated image array (copy) with the good cell applied.
    """
    return _apply_cell(a, x=x, y=y, r=r, outer_value=ring_value, inner_value=good_core)


# =====================================================================
# Function: put_bad
# =====================================================================
def put_bad(a: np.ndarray, x: int, y: int, r: float, ring_value: float, bad_core: float) -> np.ndarray:
    """
    Draw a "bad" cell (grey ring/disk with a bright/white core) onto an image.

    Parameters
    ----------
    a:
        Input image array of shape (m, n).
    x:
        1-based x-coordinate of the cell center.
    y:
        1-based y-coordinate of the cell center.
    r:
        Outer radius of the cell.
    ring_value:
        Intensity assigned to the outer disk/ring region.
    bad_core:
        Intensity assigned to the inner core region (radius r/2).

    Returns
    -------
    np.ndarray
        Updated image array (copy) with the bad cell applied.
    """
    return _apply_cell(a, x=x, y=y, r=r, outer_value=ring_value, inner_value=bad_core)


# -----------------------------
# Latent <-> image
# -----------------------------

# =====================================================================
# Function: validate_latent
# =====================================================================
def validate_latent(latent_x: Sequence[Cell], npix: int) -> None:
    """
    Validate that a latent marked point pattern is well-formed.

    Checks:
    - Each (x, y) coordinate lies in the 1..npix range (1-based indexing).
    - Each mark m is in {0, 1} where:
        m = 0 denotes a "bad" cell (white core)
        m = 1 denotes a "good" cell (black core)

    Parameters
    ----------
    latent_x:
        Sequence of cells, each a tuple (x, y, m).
    npix:
        Image dimension used to validate the coordinate bounds.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any coordinate is out of bounds or any mark is not 0 or 1.
    """
    for (x, y, m) in latent_x:
        if not (1 <= int(x) <= npix and 1 <= int(y) <= npix):
            raise ValueError(f"Cell coords must be in 1..{npix}, got (x,y)=({x},{y})")
        if int(m) not in (0, 1):
            raise ValueError(f"Cell mark must be 0 (bad) or 1 (good), got m={m}")


# =====================================================================
# Function: render_noiseless_from_latent
# =====================================================================
def render_noiseless_from_latent(
    latent_x_back_to_front: Sequence[Cell],
    params: RenderParams = RenderParams(),
) -> np.ndarray:
    """
    Render the noiseless pixel array f(x) from a latent state.

    The latent state is an ordered list of marked points (cells). The ordering
    is BACK-to-FRONT: later cells overwrite earlier ones (occlusion).

    Parameters
    ----------
    latent_x_back_to_front:
        Sequence of (x, y, m) tuples in BACK-to-FRONT order.
        - x, y are 1-based pixel coordinates in {1, ..., params.npix}
        - m is the mark: 0 = bad (white core), 1 = good (black core)
    params:
        Rendering parameters controlling image size, radii, and intensity values.

    Returns
    -------
    np.ndarray
        Noiseless rendered image array of shape (params.npix, params.npix),
        with floating-point intensities (not clipped).
    """
    validate_latent(latent_x_back_to_front, params.npix)

    img = np.full((params.npix, params.npix), params.background, dtype=float)

    for (xc, yc, m) in latent_x_back_to_front:
        if int(m) == 1:
            img = put_good(img, int(xc), int(yc), params.cellrad, params.ring_value, params.good_core)
        else:
            img = put_bad(img, int(xc), int(yc), params.cellrad, params.ring_value, params.bad_core)

    return img


# =====================================================================
# Function: add_gaussian_noise
# =====================================================================
def add_gaussian_noise(img: np.ndarray, stddev: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Add i.i.d. Gaussian noise to an image array.

    Implements:
        y = img + stddev * Z
    where Z has i.i.d. standard normal entries.

    Parameters
    ----------
    img:
        Input image array (any shape), typically float-valued.
    stddev:
        Standard deviation of the additive Gaussian noise.
    seed:
        Optional RNG seed for reproducibility. If None, uses an unseeded generator.

    Returns
    -------
    np.ndarray
        Noisy image array with the same shape as `img`.
    """
    rng = np.random.default_rng(seed)
    return img + float(stddev) * rng.standard_normal(img.shape)


# =====================================================================
# Function: make_fake_with_latent
# =====================================================================
def make_fake_with_latent(
    params: MakeFakeParams = MakeFakeParams(),
    seed: Optional[int] = None,
) -> tuple[np.ndarray, List[Cell]]:
    """
    Generate a synthetic "cells" image like makefake.m, and also return the latent state.

    This reproduces the makefake.m behaviour:
    - Initialize a bright background image.
    - Sample number of cells:
          ncell = 5 + ceil((ncellmax - 5) * rand)
    - Sample pbad:
          pbad = 0.25 + 0.5 * rand
      so pbad is uniform on [0.25, 0.75] by default.
    - Sample cell centers:
          xycell = ceil(npix * rand(2, ncell))
      which yields integer coordinates in 1..npix (MATLAB-style).
    - For each cell, sample its mark:
          is_bad = (rand < pbad)
      and draw it onto the image (later cells overwrite earlier ones).
    - Add i.i.d. Gaussian noise with standard deviation `stddev`.

    Parameters
    ----------
    params:
        Parameters controlling image size, max cell count, drawing intensities,
        and noise level.
    seed:
        Optional RNG seed for reproducibility. If None, uses an unseeded generator.

    Returns
    -------
    noisy_slide : np.ndarray
        The noisy observed image after adding Gaussian noise. Shape (npix, npix).
    latent_x_back_to_front : List[Cell]
        List of (x, y, m) in the order they were drawn (BACK-to-FRONT).
        Here m = 0 means bad, m = 1 means good.
    """
    rng = np.random.default_rng(seed)

    # Background
    slide = np.ones((params.npix, params.npix), dtype=float) * params.background

    # Number of cells: ncell = 5 + ceil((ncellmax - 5) * rand)
    u = rng.random()
    ncell = 5 + int(np.ceil((params.ncellmax - 5) * u))

    # pbad = 0.25 + 0.5*rand
    pbad = params.pbad_low + params.pbad_range * rng.random()

    # xycell = ceil(npix*rand(2,ncell)) -> coords in 1..npix
    xycell = np.ceil(params.npix * rng.random((2, ncell))).astype(int)

    latent_x: List[Cell] = []

    # Draw cells in order (back-to-front)
    for icell in range(ncell):
        x = int(xycell[0, icell])
        y = int(xycell[1, icell])

        is_bad = rng.random() < pbad
        m = 0 if is_bad else 1  # 0=bad, 1=good
        latent_x.append((x, y, m))

        if is_bad:
            slide = put_bad(slide, x, y, params.cellrad, params.ring_value, params.bad_core)
        else:
            slide = put_good(slide, x, y, params.cellrad, params.ring_value, params.good_core)

    # Add noise
    slide = slide + params.stddev * rng.standard_normal(slide.shape)

    return slide, latent_x


# -----------------------------
# Saving utilities
# -----------------------------

# =====================================================================
# Function: save_array_png
# =====================================================================
def save_array_png(
    x_array: np.ndarray,
    out_stem: Union[str, Path],
    *,
    png_name: Optional[str] = None,
) -> tuple[Path, Path]:
    """
    Save a clipped PNG preview of an image array.

    This function:
    1) Clips the array to the range [0, 1].
    2) Converts to 8-bit grayscale using:
           x_u8 = (clip(x, 0, 1) * 255 + 0.5).astype(uint8)
       The +0.5 implements rounding to the nearest integer.
    3) Writes a grayscale PNG (mode "L") via Pillow.

    Parameters
    ----------
    x_array:
        Input image array (typically float-valued). Any values outside [0, 1]
        will be clipped for the PNG preview.
    out_stem:
        Output path stem (directory + base filename without extension).
        The PNG will be saved as:
            <out_stem.parent>/<out_stem.name>.png
        unless `png_name` is provided.
    png_name:
        Optional explicit PNG filename (within out_stem.parent). If provided,
        the PNG is saved as:
            <out_stem.parent>/<png_name>

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    x_array = np.asarray(x_array)
    out_stem = Path(out_stem)

    out_dir = out_stem.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = out_stem.name
    png_path = out_dir / (png_name if png_name is not None else f"{stem}.png")

    # (2) clipped preview PNG
    x_clip = np.clip(x_array, 0.0, 1.0)
    x_u8 = (x_clip * 255.0 + 0.5).astype(np.uint8)

    from PIL import Image  # pip install pillow
    Image.fromarray(x_u8, mode="L").save(png_path)

    return png_path


# =====================================================================
# Function: save_array_npy
# =====================================================================
def save_array_npy(x_array: np.ndarray, out_stem: Union[str, Path]) -> Path:
    """
    Save the (unclipped) image array to a NumPy .npy file.

    The array is saved EXACTLY as-is (no clipping, no scaling), using np.save.

    Parameters
    ----------
    x_array:
        Array to be saved (typically the noisy image with floating-point values).
    out_stem:
        Output path stem (directory + base filename without extension).
        The array is saved to:
            <out_stem.parent>/<out_stem.name>.npy

    Returns
    -------
    Path
        Path to the saved .npy file.
    """
    x_array = np.asarray(x_array)
    out_stem = Path(out_stem)

    out_dir = out_stem.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / f"{out_stem.name}.npy"
    np.save(npy_path, x_array)   # unclamped float array
    return npy_path


# =====================================================================
# Function: save_latent_npy
# =====================================================================
def save_latent_npy(latent_x: Sequence[Cell], out_stem: Union[str, Path]) -> Path:
    """
    Save the latent state (cell list) to a NumPy .npy file.

    The latent state is converted to an integer array of shape (n, 3), where each
    row is (x, y, m), and then saved using np.save.

    Parameters
    ----------
    latent_x:
        Sequence of cells, each a tuple (x, y, m) with 1-based coordinates and
        mark m in {0, 1}.
    out_stem:
        Output path stem (directory + base filename without extension).
        The latent array is saved to:
            <out_stem.parent>/<out_stem.name>_latent.npy

    Returns
    -------
    Path
        Path to the saved latent .npy file.
    """
    out_stem = Path(out_stem)
    out_dir = out_stem.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    latent_arr = np.array([(int(x), int(y), int(m)) for (x, y, m) in latent_x], dtype=int)
    npy_path = out_dir / f"{out_stem.name}_latent.npy"
    np.save(npy_path, latent_arr)
    return npy_path

# -----------------------------
# One-call convenience
# -----------------------------

# =====================================================================
# Function: render_or_generate_and_save
# =====================================================================
def render_or_generate_and_save(
    out_stem: Union[str, Path],
    *,
    latent_x: Optional[Sequence[Cell]] = None,
    seed: Optional[int] = None,
    noise_stddev: float = 0.1,
) -> dict:
    """
    Convenience wrapper to (a) render from a provided latent state OR (b) generate
    a synthetic image, and then save outputs to disk.

    Two modes:
    1) If `latent_x` is provided:
       - Render noiseless image from `latent_x` using default RenderParams
       - Add Gaussian noise with standard deviation `noise_stddev`
    2) If `latent_x` is None:
       - Generate a synthetic noisy image and its latent state using makefake logic

    In both modes, the function saves:
    - The noisy (unclipped) image array to "<out_stem>.npy"
    - The latent state to "<out_stem>_latent.npy"
    - A clipped PNG preview to "<out_stem>.png"

    Parameters
    ----------
    out_stem:
        Output path stem (directory + base filename without extension) used to
        construct output filenames.
    latent_x:
        Optional latent state (sequence of (x, y, m)) in BACK-to-FRONT order.
        If provided, the image is rendered from this latent state.
        If None, a new synthetic image is generated.
    seed:
        Optional RNG seed for reproducibility (used in noise addition and/or generation).
    noise_stddev:
        Noise standard deviation used ONLY when `latent_x` is provided (render mode).
        In generation mode, noise level comes from MakeFakeParams.stddev.

    Returns
    -------
    dict
        Dictionary containing:
        - "img_npy_path": Path to saved image array (.npy)
        - "latent_npy_path": Path to saved latent array (.npy)
        - "png_path": Path to saved PNG preview (.png)
        - "array_saved": The in-memory image array that was saved (noisy, unclipped)
        - "latent_x_back_to_front": The latent state used (list/sequence of cells)
    """
    out_stem = Path(out_stem)

    if latent_x is not None:
        # Render from provided latent
        render_params = RenderParams()
        noiseless = render_noiseless_from_latent(latent_x, render_params)
        arr = add_gaussian_noise(noiseless, noise_stddev, seed=seed)
        latent_used = latent_x
    else:
        # makefake returns NOISY slide already + latent
        makefake_params = MakeFakeParams()
        arr, latent_used = make_fake_with_latent(makefake_params, seed=seed)

    # Save ONLY the unclamped pixel array
    img_npy_path = save_array_npy(arr, out_stem)
    # Save latent_x
    latent_npy_path = save_latent_npy(latent_used, out_stem)
    # Save clipped PNG
    png_path = save_array_png(arr, out_stem)

    return {
        "img_npy_path": img_npy_path,
        "latent_npy_path": latent_npy_path,
        "png_path": png_path,
        "array_saved": arr,
        "latent_x_back_to_front": latent_used,
    }

# =====================================================================
# Function: main
# =====================================================================
def main() -> None:
    # latent_x = [
    #     (15, 45, 0),  # bad cell at (15,45) with white core
    #     (30, 35, 1),  # good cell at (30,35) with black core
    #     (1, 1, 1),    # good cell at (1,1) with black core
    # ]
    latent_x = None  # Set to None to generate a new synthetic image instead of rendering from latent

    out = render_or_generate_and_save(
        out_stem="outputs/slide",
        latent_x=latent_x,
        seed=None,
        noise_stddev=0.1,
    )

    print("Saved files:")
    print(f"  Image array (unclipped): {out['img_npy_path']}")
    print(f"  Latent (x,y,m): {out['latent_npy_path']}")
    print(f"  Clipped PNG preview: {out['png_path']}")

if __name__ == "__main__":
    main()
