"""
=============================================================================
STAR-VP SalMap Processor — Saliency Map to S_xyz Conversion
=============================================================================

Paper:  Section 3.2, Equation 1
        Section 4.1.4 (processing parameters: tt=20480, sr=1/160)

Converts 2D saliency maps (from PAVER) into the compact S_xyz representation.
This is a preprocessing step with NO learnable parameters.

Process (per frame):
    1. For every pixel (i, j) in the H×W saliency map, compute its 3D
       coordinate on the unit sphere using the paper's physics spherical
       convention (Eq. 1):
           θ = (2π / W) · (j + 0.5)     — azimuthal angle (longitude)
           φ = (π  / H) · (i + 0.5)     — polar angle (colatitude from +z)
           x = cos(θ) · sin(φ)
           y = sin(θ) · sin(φ)
           z = cos(φ)

    2. Each pixel becomes a 4D tuple (x, y, z, s), where s is its saliency.

    3. Keep only the top tt=20,480 pixels with highest saliency values
       (sorted descending).

    4. Uniformly subsample at rate sr=1/160 (take every 160th point from
       the sorted list), yielding D_P = floor(tt · sr) = 128 points.

    5. Output S_xyz of shape [D_P, 4] = [128, 4] per frame.

COORDINATE CONVENTION — physics spherical (paper's Eq. 1):
    φ is the POLAR angle (colatitude) measured from the +z axis (north pole):
        φ = 0   → north pole  → (x, y, z) = (0, 0, 1)
        φ = π/2 → equator
        φ = π   → south pole  → (x, y, z) = (0, 0, -1)

    This DIFFERS from the geographic convention in muse_vp/salmap_processor.py:
    ┌────────────────────┬──────────────────────┬──────────────────────────┐
    │ Aspect             │ MUSE-VP              │ STAR-VP (this file)      │
    ├────────────────────┼──────────────────────┼──────────────────────────┤
    │ Coordinate conv.   │ Geographic (y = up)  │ Physics (z = polar axis) │
    │ φ definition       │ Latitude from equator│ Polar angle from +z      │
    │ Selection method   │ Direct top-K         │ Top-tt then subsample    │
    │ Number of points   │ K=32                 │ D_P=128                  │
    │ Output format      │ xyz [T,K,3] + w [T,K]│ S_xyz [T,128,4]         │
    │ Area correction    │ cos(φ) applied       │ Not applied (per paper)  │
    │ Weight normaliz.   │ Sum-to-1             │ Raw saliency values      │
    └────────────────────┴──────────────────────┴──────────────────────────┘
=============================================================================
"""

import math
import torch
import torch.nn as nn


# ========================================================================= #
#  SalMap Processor  (Paper Section 3.2, Eq. 1)
# ========================================================================= #

class SalMapProcessor(nn.Module):
    """
    Converts 2D equirectangular saliency maps to 4D sphere+saliency points.

    No learnable parameters — pure geometric transformation + selection.
    Registered as nn.Module so its pre-computed buffers move with .to(device).

    The output D_P points per frame are obtained by:
        1. Selecting the top `top_threshold` (tt) highest-saliency pixels.
        2. Uniformly subsampling every `subsample_stride`-th point.
    Giving D_P = floor(top_threshold × subsample_rate) points per frame.

    With paper defaults (tt=20480, sr=1/160):  D_P = 128.
    """

    def __init__(
        self,
        top_threshold: int = 20_480,
        subsample_rate: float = 1.0 / 160.0,
        height: int = 224,
        width: int = 448,
    ):
        """
        Args:
            top_threshold: Number of highest-saliency pixels to retain before
                           subsampling.  Paper Table 1 / Section 4.1.4: tt=20480.
            subsample_rate: Fraction of retained points to keep after subsampling.
                            Paper: sr = 1/160.
            height: Saliency map height from PAVER.  Default 224.
            width:  Saliency map width from PAVER.   Default 448.
        """
        super().__init__()
        self.top_threshold = top_threshold
        self.subsample_rate = subsample_rate
        self.height = height
        self.width = width

        # D_P = floor(tt × sr)
        self.d_p = int(math.floor(top_threshold * subsample_rate))

        assert self.d_p > 0, (
            f"D_P = floor({top_threshold} × {subsample_rate}) = {self.d_p}; must be > 0"
        )
        assert top_threshold <= height * width, (
            f"top_threshold={top_threshold} exceeds total pixels={height * width}"
        )

        # stride = round(1 / sr) — integer step for uniform subsampling
        self.subsample_stride = int(round(1.0 / subsample_rate))

        # ================================================================= #
        # Pre-compute 3D sphere coordinates for ALL pixels (Eq. 1)
        # ================================================================= #
        i_coords = torch.arange(height, dtype=torch.float32)   # [H]
        j_coords = torch.arange(width,  dtype=torch.float32)   # [W]
        grid_i, grid_j = torch.meshgrid(i_coords, j_coords, indexing="ij")

        # Paper Eq. 1:
        #   θ = (2π / W) · (j + 0.5)   — azimuthal angle  ∈ [0, 2π)
        #   φ = (π  / H) · (i + 0.5)   — polar angle      ∈ (0, π)
        theta = (2.0 * math.pi / width)  * (grid_j + 0.5)   # [H, W]
        phi   = (math.pi       / height) * (grid_i + 0.5)   # [H, W]

        #   (x, y, z) = (cos θ · sin φ,  sin θ · sin φ,  cos φ)
        sin_phi = torch.sin(phi)
        x = torch.cos(theta) * sin_phi     # [H, W]
        y = torch.sin(theta) * sin_phi     # [H, W]
        z = torch.cos(phi)                 # [H, W]

        # Flatten to [H*W, 3] for indexed lookup during forward pass
        sphere_coords_flat = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

        # Pre-compute the subsample indices: [0, stride, 2·stride, …]
        subsample_indices = torch.arange(
            0, top_threshold, self.subsample_stride, dtype=torch.long,
        )[:self.d_p]                        # exactly D_P entries

        # Register as buffers (non-trainable, but travel with .to(device))
        self.register_buffer("sphere_coords_flat", sphere_coords_flat)
        self.register_buffer("subsample_indices", subsample_indices)

    # --------------------------------------------------------------------- #
    #  Forward
    # --------------------------------------------------------------------- #

    def forward(self, saliency_map: torch.Tensor) -> torch.Tensor:
        """
        Convert saliency map(s) to the paper's S_xyz representation.

        Args:
            saliency_map:
                [H, W]      — single frame.
                [T, H, W]   — T frames (batched over time).
                Values should be non-negative (PAVER outputs [0, 1]).

        Returns:
            s_xyz: torch.Tensor
                [D_P, 4]      for a single frame.
                [T, D_P, 4]   for T frames.
                Each row is (x, y, z, s) — unit-sphere coordinate + saliency.
        """
        single_frame = saliency_map.dim() == 2
        if single_frame:
            saliency_map = saliency_map.unsqueeze(0)        # → [1, H, W] //adds a dimension for the batch

        T, H, W = saliency_map.shape
        assert H == self.height and W == self.width, (
            f"Expected saliency map {self.height}×{self.width}, got {H}×{W}"
        )

        sal_flat = saliency_map.reshape(T, -1)              # [T, H*W]

        # --- Step 1: top-tt selection (sorted descending by saliency) ---
        top_values, top_indices = torch.topk(
            sal_flat, k=self.top_threshold, dim=1, sorted=True,
        )                                                    # [T, tt]

        # --- Step 2: uniform subsampling at stride intervals ---
        sub_idx = self.subsample_indices                     # [D_P]  (long)
        sub_values  = top_values[:, sub_idx]                 # [T, D_P]
        sub_indices = top_indices[:, sub_idx]                # [T, D_P]

        # --- Step 3: look up 3D coordinates for the selected pixels ---
        idx_exp = sub_indices.unsqueeze(-1).expand(-1, -1, 3)           # [T, D_P, 3]
        coords  = self.sphere_coords_flat.unsqueeze(0).expand(T, -1, -1)  # [T, H*W, 3]
        xyz = torch.gather(coords, dim=1, index=idx_exp)               # [T, D_P, 3]

        # --- Step 4: concatenate (x, y, z, s) ---
        s_xyz = torch.cat([xyz, sub_values.unsqueeze(-1)], dim=-1)     # [T, D_P, 4]

        if single_frame:
            s_xyz = s_xyz.squeeze(0)                         # [D_P, 4]

        return s_xyz

    # --------------------------------------------------------------------- #
    #  Batch / full-video processing
    # --------------------------------------------------------------------- #

    def process_full_video(
        self,
        saliency_tensor: torch.Tensor,
        chunk_size: int = 500,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Process an entire video's saliency maps in memory-friendly chunks.

        Args:
            saliency_tensor: [num_frames, H, W] from PAVER.
            chunk_size: Frames per chunk (limits peak GPU memory).
            verbose: Print progress.

        Returns:
            s_xyz: [num_frames, D_P, 4] on CPU.
        """
        num_frames = saliency_tensor.shape[0]
        device = self.sphere_coords_flat.device

        if verbose:
            print(f"  Processing {num_frames} frames in chunks of {chunk_size}…")

        chunks_out: list[torch.Tensor] = []

        for start in range(0, num_frames, chunk_size):
            end = min(start + chunk_size, num_frames)
            chunk = saliency_tensor[start:end].to(device)

            with torch.no_grad():
                s_xyz_chunk = self.forward(chunk)

            chunks_out.append(s_xyz_chunk.cpu())

            if verbose and (start // chunk_size) % 10 == 0:
                pct = end / num_frames * 100
                print(f"    {end}/{num_frames} frames ({pct:.1f}%)")

        s_xyz = torch.cat(chunks_out, dim=0)     # [num_frames, D_P, 4]

        if verbose:
            print(f"  Done: s_xyz {tuple(s_xyz.shape)}")

        return s_xyz

    # --------------------------------------------------------------------- #
    #  Repr
    # --------------------------------------------------------------------- #

    def extra_repr(self) -> str:
        return (
            f"tt={self.top_threshold}, sr={self.subsample_rate}, "
            f"stride={self.subsample_stride}, D_P={self.d_p}, "
            f"H={self.height}, W={self.width}"
        )


# ========================================================================= #
#  Standalone utility functions (offline preprocessing)
# ========================================================================= #

def process_single_video(
    input_path: str,
    output_path: str,
    top_threshold: int = 20_480,
    subsample_rate: float = 1.0 / 160.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Process one PAVER *_saliency.pt file → STAR-VP *_salxyz.pt file.

    Saves a dict with keys:
        s_xyz        — [num_frames, D_P, 4]
        d_p          — int (128 by default)
        top_threshold— int (20480)
        subsample_rate— float (1/160)
        input_shape  — tuple (num_frames, H, W)
        source_file  — str
    """
    import os

    print(f"\n{'=' * 70}")
    print(f"STAR-VP SalMap Processor — {os.path.basename(input_path)}")
    print(f"{'=' * 70}")

    print(f"\n[1/3] Loading saliency from: {input_path}")
    saliency = torch.load(input_path, map_location="cpu", weights_only=True)
    print(f"  Shape: {saliency.shape}  Range: [{saliency.min():.4f}, {saliency.max():.4f}]")

    print(f"\n[2/3] Initializing processor (tt={top_threshold}, sr={subsample_rate})…")
    processor = SalMapProcessor(
        top_threshold=top_threshold,
        subsample_rate=subsample_rate,
        height=saliency.shape[1],
        width=saliency.shape[2],
    ).to(device)
    print(f"  D_P = {processor.d_p}")

    print(f"\n[3/3] Processing all frames…")
    s_xyz = processor.process_full_video(saliency, chunk_size=500, verbose=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    output_dict = {
        "s_xyz":           s_xyz,                          # [N, D_P, 4]
        "d_p":             processor.d_p,
        "top_threshold":   top_threshold,
        "subsample_rate":  subsample_rate,
        "input_shape":     tuple(saliency.shape),
        "source_file":     str(input_path),
    }
    torch.save(output_dict, output_path)

    xyz_part = s_xyz[..., :3]
    norms = torch.norm(xyz_part, dim=-1)
    print(f"\n{'=' * 70}")
    print(f"SAVED: {output_path}")
    print(f"  s_xyz shape:  {tuple(s_xyz.shape)}")
    print(f"  xyz range:    [{xyz_part.min():.4f}, {xyz_part.max():.4f}]")
    print(f"  sal range:    [{s_xyz[..., 3].min():.4f}, {s_xyz[..., 3].max():.4f}]")
    print(f"  norm check:   min={norms.min():.6f}, max={norms.max():.6f} (expect ~1.0)")
    print(f"  file size:    {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    print(f"{'=' * 70}")

    return s_xyz


def process_all_videos(
    saliency_dir: str,
    output_dir: str,
    top_threshold: int = 20_480,
    subsample_rate: float = 1.0 / 160.0,
    device: str = "cpu",
) -> None:
    """Batch-process all *_saliency.pt files in a directory."""
    import os
    import glob
    import time

    pattern = os.path.join(saliency_dir, "*_saliency.pt")
    saliency_files = sorted(glob.glob(pattern))

    if not saliency_files:
        print(f"ERROR: No *_saliency.pt files found in {saliency_dir}")
        return

    print(f"\n{'#' * 70}")
    print(f"STAR-VP Batch SalMap Processor — {len(saliency_files)} videos")
    print(f"  Input:  {saliency_dir}")
    print(f"  Output: {output_dir}")
    print(f"  tt={top_threshold}, sr={subsample_rate}, device={device}")
    print(f"{'#' * 70}")

    os.makedirs(output_dir, exist_ok=True)
    total_start = time.time()
    results: list[tuple[str, str]] = []

    for idx, sal_path in enumerate(saliency_files):
        name = os.path.basename(sal_path).replace("_saliency.pt", "")
        out_path = os.path.join(output_dir, f"{name}_salxyz.pt")

        if os.path.exists(out_path):
            print(f"\n[{idx+1}/{len(saliency_files)}] SKIP (exists): {name}")
            results.append((name, "skipped"))
            continue

        print(f"\n[{idx+1}/{len(saliency_files)}] {name}")
        t0 = time.time()
        try:
            process_single_video(
                input_path=sal_path,
                output_path=out_path,
                top_threshold=top_threshold,
                subsample_rate=subsample_rate,
                device=device,
            )
            results.append((name, f"OK ({time.time() - t0:.1f}s)"))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, f"FAILED: {e}"))

    elapsed = time.time() - total_start
    print(f"\n{'#' * 70}")
    print(f"BATCH COMPLETE — {elapsed:.1f}s total")
    for name, status in results:
        print(f"  {name}: {status}")
    print(f"{'#' * 70}")


def inspect_salxyz(filepath: str) -> None:
    """Print diagnostics for a processed *_salxyz.pt file."""
    import os

    print(f"\n{'=' * 70}")
    print(f"Inspecting: {os.path.basename(filepath)}")
    print(f"{'=' * 70}")

    data = torch.load(filepath, map_location="cpu", weights_only=False)
    s_xyz = data["s_xyz"]

    print(f"  Source:       {data.get('source_file', 'N/A')}")
    print(f"  Input shape:  {data.get('input_shape', 'N/A')}")
    print(f"  D_P:          {data.get('d_p', s_xyz.shape[-2])}")
    print(f"  tt:           {data.get('top_threshold', 'N/A')}")
    print(f"  sr:           {data.get('subsample_rate', 'N/A')}")
    print(f"  s_xyz shape:  {tuple(s_xyz.shape)}")
    print(f"  s_xyz dtype:  {s_xyz.dtype}")

    xyz = s_xyz[..., :3]
    sal = s_xyz[..., 3]

    print(f"  x range:      [{xyz[..., 0].min():.4f}, {xyz[..., 0].max():.4f}]")
    print(f"  y range:      [{xyz[..., 1].min():.4f}, {xyz[..., 1].max():.4f}]")
    print(f"  z range:      [{xyz[..., 2].min():.4f}, {xyz[..., 2].max():.4f}]")
    print(f"  sal range:    [{sal.min():.4f}, {sal.max():.4f}]")

    norms = torch.norm(xyz, dim=-1)
    print(f"  norms:        min={norms.min():.6f}, max={norms.max():.6f}")

    print(f"\n  Frame 0 — first 5 points:")
    print(f"  {'#':<4} {'x':>8} {'y':>8} {'z':>8} {'sal':>8}")
    for k in range(min(5, s_xyz.shape[-2])):
        row = s_xyz[0, k] if s_xyz.dim() == 3 else s_xyz[k]
        print(f"  {k:<4} {row[0]:>8.4f} {row[1]:>8.4f} {row[2]:>8.4f} {row[3]:>8.4f}")


# ========================================================================= #
#  Self-Test — exhaustive verification
# ========================================================================= #

def _self_test() -> bool:
    """
    Exhaustive verification of SalMapProcessor for STAR-VP.

    Tests:
      1.  Default D_P = 128
      2.  Output shape — single frame [D_P, 4]
      3.  Output shape — batch [T, D_P, 4]
      4.  Coordinate: north pole (i≈0)          → z ≈ +1
      5.  Coordinate: south pole (i≈H-1)        → z ≈ -1
      6.  Coordinate: equator front (i≈H/2 j=0) → x ≈ +1
      7.  Coordinate: equator right (i≈H/2 j≈W/4) → y ≈ +1
      8.  Unit sphere — xyz norms ≈ 1
      9.  Saliency column — 4th value ≥ 0
     10.  Saliency ordering — first point ≥ last point
     11.  Top-threshold — focused map selects from high-saliency region
     12.  Subsample indices — correct buffer values
     13.  Subsample stride — correct value
     14.  Determinism — same input → identical output
     15.  All-zero saliency — no NaN, s=0 everywhere
     16.  Uniform saliency — correct s value
     17.  Batch vs single frame consistency
     18.  No NaN/Inf in output
     19.  No learnable parameters
     20.  Custom params (tt=1024, sr=1/8 → D_P=128, stride=8)
     21.  Custom params (tt=640, sr=1/10 → D_P=64)
     22.  process_full_video chunking matches direct forward
     23.  Coordinate range — all xyz in [-1, 1]
     24.  Total pixel count — sphere_coords_flat has H*W rows
     25.  Device transfer — buffers follow .to()
    """
    torch.manual_seed(42)

    H, W = 224, 448
    TT = 20_480
    SR = 1.0 / 160.0
    D_P = 128

    all_passed = True
    test_num = 0

    def check(condition: bool, name: str) -> None:
        nonlocal all_passed, test_num
        test_num += 1
        status = "OK" if condition else "FAIL"
        print(f"  [{test_num:>2}] {name}: {status}")
        if not condition:
            all_passed = False

    print("=" * 72)
    print("STAR-VP SalMap Processor Self-Test")
    print("=" * 72)

    proc = SalMapProcessor(top_threshold=TT, subsample_rate=SR, height=H, width=W)

    # =================================================================== #
    #  Test Group A: Basic shapes and D_P
    # =================================================================== #
    print("\n--- A. Basic shapes ---")

    check(proc.d_p == D_P, f"D_P = {proc.d_p} == {D_P}")

    sal_single = torch.rand(H, W)
    out_single = proc(sal_single)
    check(
        tuple(out_single.shape) == (D_P, 4),
        f"Single frame shape {tuple(out_single.shape)}",
    )

    T = 5
    sal_batch = torch.rand(T, H, W)
    out_batch = proc(sal_batch)
    check(
        tuple(out_batch.shape) == (T, D_P, 4),
        f"Batch shape {tuple(out_batch.shape)}",
    )

    # =================================================================== #
    #  Test Group B: Coordinate convention (Paper Eq. 1)
    # =================================================================== #
    print("\n--- B. Coordinate convention (Eq. 1) ---")

    coords_2d = proc.sphere_coords_flat.reshape(H, W, 3)

    # B1: North pole — pixel (0, 0):  φ = π·0.5/224 ≈ 0.007
    #     z = cos(φ) ≈ 1,  x,y ≈ 0
    north_z = coords_2d[0, 0, 2].item()
    check(north_z > 0.999, f"North pole z={north_z:.6f} > 0.999")

    # B2: South pole — pixel (H-1, 0):  φ = π·(223.5)/224 ≈ π − 0.007
    #     z = cos(φ) ≈ −1
    south_z = coords_2d[H - 1, 0, 2].item()
    check(south_z < -0.999, f"South pole z={south_z:.6f} < -0.999")

    # B3: Equator front — pixel (H//2, 0):  φ ≈ π/2,  θ ≈ 0
    #     x = cos(0)·sin(π/2) ≈ 1
    eq_x = coords_2d[H // 2, 0, 0].item()
    check(abs(eq_x) > 0.95, f"Equator front x={eq_x:.6f}, |x|>0.95")

    # B4: Equator right — pixel (H//2, W//4):  φ ≈ π/2,  θ ≈ π/2
    #     y = sin(π/2)·sin(π/2) ≈ 1
    eq_y = coords_2d[H // 2, W // 4, 1].item()
    check(abs(eq_y) > 0.95, f"Equator right y={eq_y:.6f}, |y|>0.95")

    # =================================================================== #
    #  Test Group C: Sphere geometry
    # =================================================================== #
    print("\n--- C. Sphere geometry ---")

    # C1: All pre-computed coords are unit vectors
    all_norms = torch.norm(proc.sphere_coords_flat, dim=-1)
    check(
        torch.allclose(all_norms, torch.ones_like(all_norms), atol=1e-6),
        f"Pre-computed norms: [{all_norms.min():.7f}, {all_norms.max():.7f}]",
    )

    # C2: Output xyz are unit vectors
    xyz_out = out_batch[:, :, :3]
    norms_out = torch.norm(xyz_out, dim=-1)
    check(
        torch.allclose(norms_out, torch.ones_like(norms_out), atol=1e-6),
        f"Output norms: [{norms_out.min():.7f}, {norms_out.max():.7f}]",
    )

    # C3: All xyz components in [-1, 1]
    check(
        (xyz_out >= -1.0 - 1e-6).all() and (xyz_out <= 1.0 + 1e-6).all(),
        f"xyz in [-1,1]: [{xyz_out.min():.6f}, {xyz_out.max():.6f}]",
    )

    # =================================================================== #
    #  Test Group D: Saliency selection logic
    # =================================================================== #
    print("\n--- D. Saliency selection ---")

    # D1: Saliency column ≥ 0
    sal_col = out_batch[:, :, 3]
    check((sal_col >= 0).all().item(), "All saliency values >= 0")

    # D2: Saliency ordering — first subsampled point ≥ last
    # (topk is sorted descending; index 0 is highest, index 160*127 is lower)
    torch.manual_seed(77)
    out_order = proc(torch.rand(H, W))
    check(
        out_order[0, 3].item() >= out_order[-1, 3].item(),
        f"Ordering: first sal={out_order[0, 3]:.4f} >= last={out_order[-1, 3]:.4f}",
    )

    # D3: Focused saliency — high-sal region is selected first
    sal_focused = torch.zeros(H, W)
    sal_focused[100:150, 200:250] = 1.0   # 2500 pixels at 1.0
    out_focused = proc(sal_focused)
    check(
        out_focused[0, 3].item() == 1.0,
        f"Focused: top point sal={out_focused[0, 3]:.4f} == 1.0",
    )

    # D4: Focused — the first few subsampled points should be from the patch
    # Index 0 is the top-1 saliency pixel (in the patch).
    # Index 1 is the 160th highest (still in the 2500-pixel patch).
    # The patch has 2500 pixels; 2500/160 ≈ 15 subsampled points from it.
    n_from_patch = (out_focused[:, 3] == 1.0).sum().item()
    check(
        n_from_patch >= 15,
        f"Focused: {n_from_patch} subsampled points from patch (expect >= 15)",
    )

    # =================================================================== #
    #  Test Group E: Subsample indices
    # =================================================================== #
    print("\n--- E. Subsample indices ---")

    expected_idx = torch.arange(0, TT, 160, dtype=torch.long)[:D_P]
    check(
        torch.equal(proc.subsample_indices, expected_idx),
        f"Subsample indices: len={len(proc.subsample_indices)}, "
        f"last={proc.subsample_indices[-1].item()}",
    )
    check(proc.subsample_stride == 160, f"Stride = {proc.subsample_stride}")

    # =================================================================== #
    #  Test Group F: Robustness
    # =================================================================== #
    print("\n--- F. Robustness ---")

    # F1: Determinism
    torch.manual_seed(55)
    sal_det = torch.rand(H, W)
    out_a = proc(sal_det)
    out_b = proc(sal_det)
    check(torch.equal(out_a, out_b), "Deterministic: identical on repeated call")

    # F2: All-zero saliency
    out_zero = proc(torch.zeros(H, W))
    check(tuple(out_zero.shape) == (D_P, 4), f"Zero sal shape {tuple(out_zero.shape)}")
    check(not torch.isnan(out_zero).any(), "Zero sal: no NaN")
    check((out_zero[:, 3] == 0).all().item(), "Zero sal: all s=0")

    # F3: Uniform saliency
    out_uni = proc(torch.full((H, W), 0.5))
    check(
        torch.allclose(out_uni[:, 3], torch.full((D_P,), 0.5)),
        "Uniform sal: all s=0.5",
    )

    # F4: No NaN/Inf
    check(
        not (torch.isnan(out_batch).any() or torch.isinf(out_batch).any()),
        "No NaN/Inf in batch output",
    )

    # F5: Batch vs single frame
    torch.manual_seed(88)
    sal_f = torch.rand(H, W)
    out_f = proc(sal_f)                              # [D_P, 4]
    out_b_single = proc(sal_f.unsqueeze(0))           # [1, D_P, 4]
    check(
        torch.equal(out_f, out_b_single.squeeze(0)),
        "Single frame == batch[0]",
    )

    # =================================================================== #
    #  Test Group G: Module properties
    # =================================================================== #
    print("\n--- G. Module properties ---")

    # G1: No learnable parameters
    n_params = sum(p.numel() for p in proc.parameters())
    check(n_params == 0, f"Learnable params = {n_params} (expect 0)")

    # G2: Buffer count — sphere_coords_flat + subsample_indices
    n_buffers = sum(1 for _ in proc.buffers())
    check(n_buffers == 2, f"Buffer count = {n_buffers} (expect 2)")

    # G3: Pre-computed coords size
    check(
        proc.sphere_coords_flat.shape == (H * W, 3),
        f"Coords buffer shape {tuple(proc.sphere_coords_flat.shape)}",
    )

    # =================================================================== #
    #  Test Group H: Custom parameters
    # =================================================================== #
    print("\n--- H. Custom parameters ---")

    # H1: tt=1024, sr=1/8 → D_P=128, stride=8
    p1 = SalMapProcessor(top_threshold=1024, subsample_rate=1 / 8, height=H, width=W)
    check(p1.d_p == 128, f"Custom1 D_P={p1.d_p}")
    check(p1.subsample_stride == 8, f"Custom1 stride={p1.subsample_stride}")
    o1 = p1(torch.rand(3, H, W))
    check(tuple(o1.shape) == (3, 128, 4), f"Custom1 shape {tuple(o1.shape)}")

    # H2: tt=640, sr=1/10 → D_P=64, stride=10
    p2 = SalMapProcessor(top_threshold=640, subsample_rate=1 / 10, height=H, width=W)
    check(p2.d_p == 64, f"Custom2 D_P={p2.d_p}")
    o2 = p2(torch.rand(H, W))
    check(tuple(o2.shape) == (64, 4), f"Custom2 shape {tuple(o2.shape)}")

    # H3: Non-standard resolution (64×128, the competitor resolution from paper)
    p3 = SalMapProcessor(top_threshold=1024, subsample_rate=1 / 8, height=64, width=128)
    check(p3.d_p == 128, f"Custom3 D_P={p3.d_p}")
    o3 = p3(torch.rand(64, 128))
    check(tuple(o3.shape) == (128, 4), f"Custom3 shape {tuple(o3.shape)}")

    # =================================================================== #
    #  Test Group I: Chunked processing
    # =================================================================== #
    print("\n--- I. Chunked processing ---")

    torch.manual_seed(42)
    sal_video = torch.rand(20, H, W)
    out_direct  = proc(sal_video)                                       # [20, D_P, 4]
    out_chunked = proc.process_full_video(sal_video, chunk_size=7, verbose=False)

    check(
        torch.allclose(out_direct, out_chunked, atol=1e-6),
        "Chunked == direct forward",
    )
    check(
        tuple(out_chunked.shape) == (20, D_P, 4),
        f"Chunked shape {tuple(out_chunked.shape)}",
    )

    # Chunk size larger than total frames
    out_big = proc.process_full_video(sal_video, chunk_size=999, verbose=False)
    check(torch.allclose(out_direct, out_big, atol=1e-6), "Chunk > N works")

    # =================================================================== #
    #  Summary
    # =================================================================== #
    print(f"\n{'=' * 72}")
    print(f"{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"Total: {test_num} tests")
    print(f"{'=' * 72}")
    return all_passed


# ========================================================================= #
#  CLI
# ========================================================================= #

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="STAR-VP SalMap Processor — saliency map to S_xyz conversion",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
EXAMPLES:
  # Run self-tests:
  python -m star_vp.models.salmap_processor --test

  # Process a single video:
  python -m star_vp.models.salmap_processor \\
      --input path/to/video_saliency.pt \\
      --output path/to/video_salxyz.pt

  # Batch process a directory:
  python -m star_vp.models.salmap_processor --batch \\
      --input-dir ../PAVER/code/qual \\
      --output-dir star_vp/salxyz/

  # Inspect a processed file:
  python -m star_vp.models.salmap_processor --inspect path/to/video_salxyz.pt
        """,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test", action="store_true", help="Run self-tests")
    mode.add_argument("--batch", action="store_true", help="Process all *_saliency.pt in --input-dir")
    mode.add_argument("--inspect", type=str, default=None, help="Inspect a *_salxyz.pt file")
    mode.add_argument("--input", type=str, default=None, help="Single *_saliency.pt to process")

    parser.add_argument("--output", type=str, default=None, help="Output path for single-file mode")
    parser.add_argument(
        "--input-dir", type=str,
        default="/media/user/HDD3/Shini/STAR_VP/PAVER/code/qual",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/media/user/HDD3/Shini/STAR_VP/star_vp/salxyz",
    )
    parser.add_argument("--tt", type=int, default=20_480, help="Top threshold (default: 20480)")
    parser.add_argument("--sr", type=float, default=1.0 / 160.0, help="Subsample rate (default: 1/160)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    args = parser.parse_args()

    if args.test:
        success = _self_test()
        raise SystemExit(0 if success else 1)

    elif args.inspect:
        inspect_salxyz(args.inspect)

    elif args.batch:
        process_all_videos(
            saliency_dir=args.input_dir,
            output_dir=args.output_dir,
            top_threshold=args.tt,
            subsample_rate=args.sr,
            device=args.device,
        )

    elif args.input:
        if args.output is None:
            base = os.path.basename(args.input).replace("_saliency.pt", "_salxyz.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            args.output = os.path.join(args.output_dir, base)
        process_single_video(
            input_path=args.input,
            output_path=args.output,
            top_threshold=args.tt,
            subsample_rate=args.sr,
            device=args.device,
        )

    else:
        parser.print_help()
        print("\nERROR: Specify --test, --batch, --inspect, or --input.")
