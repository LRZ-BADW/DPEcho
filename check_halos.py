#!/usr/bin/env python3
"""Check halo continuity at MPI rank boundaries for dpecho output (no numpy)."""
import struct
import sys

def read_dat(fname):
    with open(fname, 'rb') as f:
        raw = f.read()
    n = len(raw)
    if n % 8 == 0:
        fmt = 'd'
        sz = 8
    elif n % 4 == 0:
        fmt = 'f'
        sz = 4
    else:
        raise ValueError(f"Unknown format: {n} bytes")
    nvals = n // sz
    data = struct.unpack(fmt * nvals, raw)
    return list(data)

# Parameters - Grid uses TOTAL Mx,My,Mz, not decomposed!
Mx, My, Mz = 32, 32, 32
Hx, Hy, Hz = 4, 4, 4
# All ranks have the SAME WH grid dimensions
whx = Mx + 2*Hx  # 40
why = My + 2*Hy  # 40
whz = Mz + 2*Hz  # 40

print(f"Per-rank WH dims: {whx}x{why}x{whz} = {whx*why*whz} cells")

fname = sys.argv[1] if len(sys.argv) > 1 else "out/0009/task_RH_0000.dat"
# With n=[32,32,32], h=[4,4,4], nh=[40,40,40]
# But the decomposition uses physical bounds, not grid dims
# So each rank writes 40*40*40 = 64000 cells for its own subdomain
WH = 40  # full WH dimension per rank (same for all ranks with Mx=32, Hx=4)
data = read_dat(fname)
print(f"Read {len(data)} values from {fname}")

# Check total size
expected = whx * why * whz
if len(data) != expected:
    print(f"WARNING: expected {expected} values, got {len(data)}")

# Reshape: linear index = (z * why + y) * whx + x
def idx3(z, y, x):
    return (z * why + y) * whx + x

def check_continuity(rank_l, rank_r, dir_name):
    """Check that rank_l's right halo matches rank_r's left interior and vice versa."""
    fname_l = fname.replace("_0000", f"_{rank_l:04d}")
    fname_r = fname.replace("_0000", f"_{rank_r:04d}")
    d_l = read_dat(fname_l)
    d_r = read_dat(fname_r)

    # Check right halo of rank_l vs left interior of rank_r
    # Rank_l right halo: x=[whx-Hx, whx), y=[0, why), z=[0, whz)
    # Rank_r left interior: x=[Hx, 2*Hx), y=[0, why), z=[0, whz)
    max_diff = 0.0
    max_pos = None
    for z in range(whz):
        for y in range(why):
            for h in range(Hx):
                val_l = d_l[idx3(z, y, whx - Hx + h)]
                val_r = d_r[idx3(z, y, Hx + h)]
                diff = abs(val_l - val_r)
                if diff > max_diff:
                    max_diff = diff
                    max_pos = (z, y, h, val_l, val_r)

    print(f"  {dir_name}: rank{rank_l} right halo vs rank{rank_r} left interior:", end="")
    if max_diff < 1e-12:
        print(f" OK (max diff {max_diff:.2e})")
    else:
        print(f" MISMATCH (max diff {max_diff:.2e} at [{max_pos[0]},{max_pos[1]},{max_pos[2]}]: "
              f"rank{rank_l}={max_pos[3]:.8e} rank{rank_r}={max_pos[4]:.8e})")

    # Check left halo of rank_r vs right interior of rank_l
    # Rank_r left halo: x=[0, Hx)
    # Rank_l right interior: x=[whx-2*Hx, whx-Hx) (last Hx cells of interior)
    # Interior starts at Hx and has n=Mx cells. Last Hx cells of interior: [whx-2*Hx, whx-Hx)
    # With whx=40, Hx=4: interior = [4:36], last Hx interior = [32:36]
    right_interior_start = whx - 2*Hx
    max_diff2 = 0.0
    max_pos2 = None
    for z in range(whz):
        for y in range(why):
            for h in range(Hx):
                val_r = d_r[idx3(z, y, h)]
                val_l = d_l[idx3(z, y, right_interior_start + h)]
                diff = abs(val_r - val_l)
                if diff > max_diff2:
                    max_diff2 = diff
                    max_pos2 = (z, y, h, val_l, val_r)

    print(f"  {dir_name}: rank{rank_r} left halo vs rank{rank_l} right interior:", end="")
    if max_diff2 < 1e-12:
        print(f" OK (max diff {max_diff2:.2e})")
    else:
        print(f" MISMATCH (max diff {max_diff2:.2e} at [{max_pos2[0]},{max_pos2[1]},{max_pos2[2]}]: "
              f"rank{rank_l}={max_pos2[3]:.8e} rank{rank_r}={max_pos2[4]:.8e})")

    return max_diff, max_diff2

print("\n--- MPI halo continuity checks ---")
d = data

# x-direction: pairs are (0,1), (2,3), (4,5), (6,7)
print("\n--- x-direction (y,z vary, x is MPI dir) ---")
for base in [0, 2, 4, 6]:
    check_continuity(base, base+1, f"x-pair ({base},{base+1})")

print("\n--- y-direction pairs ---")
# y-pairs: (0,2), (1,3), (4,6), (5,7)
# For y-direction check, we'd need the actual decomposition to verify.
# Skip for now since the primary fix was in x-direction.

print("\n--- Physical BC summary (BCOF3) ---")
# Rank0 left halo: physical BC (left edge of domain)
fname_r0 = fname.replace("_0000", "_0000")
d0 = read_dat(fname_r0)
left_vals = [d0[idx3(z, y, h)] for z in range(whz) for y in range(why) for h in range(Hx)]
print(f"  Rank0 left halo: min={min(left_vals):.6e} max={max(left_vals):.6e} mean={sum(left_vals)/len(left_vals):.6e}")

# Rank7 right halo: physical BC (right edge of domain)
fname_r7 = fname.replace("_0000", "_0007")
d7 = read_dat(fname_r7)
right_vals = [d7[idx3(z, y, whx-Hx+h)] for z in range(whz) for y in range(why) for h in range(Hx)]
print(f"  Rank7 right halo: min={min(right_vals):.6e} max={max(right_vals):.6e} mean={sum(right_vals)/len(right_vals):.6e}")

# Check that the leftmost interior cell of rank0 is reasonable
interior_leftmost = [d0[idx3(z, y, Hx)] for z in range(whz) for y in range(why)]
print(f"  Rank0 leftmost interior cell: min={min(interior_leftmost):.6e} max={max(interior_leftmost):.6e}")
