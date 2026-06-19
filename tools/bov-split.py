#!/usr/bin/env python3
"""Split a multi-component BOV dataset into per-variable BOV files.

Reads a .bov header with DATA_COMPONENTS: N and its associated .dat files
(one per MPI rank via %04d expansion), then writes per-variable .dat files
and matching .bov headers.

Usage:
  python tools/bov-split.py out/task_0000.bov
  python tools/bov-split.py out/task_0000.bov --var-names RH VX VY VZ PG BX BY BZ
"""
import argparse, glob, os, re, struct, sys

BOV_DIR = os.path.dirname(os.path.abspath(__file__))

VAR_LABELS = ["RH", "VX", "VY", "VZ", "PG", "BX", "BY", "BZ"]

def parse_bov(path):
    fields = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                fields[k.strip()] = v.strip()
    return fields

def parse_ints(s):
    return [int(x) for x in s.split()]

def parse_floats(s):
    return [float(x) for x in s.split()]

def fmt_size(fmt):
    return 8 if fmt == "DOUBLE" else 4

def fmt_char(fmt):
    return "d" if fmt == "DOUBLE" else "f"

def main():
    ap = argparse.ArgumentParser(description="Split multi-component BOV into per-variable files")
    ap.add_argument("bov", help="Path to .bov header file")
    ap.add_argument("--var-names", nargs="+", default=VAR_LABELS,
                    help="Variable names (default: RH VX VY VZ PG BX BY BZ)")
    args = ap.parse_args()

    bov_path = args.bov
    if not os.path.exists(bov_path):
        print(f"Error: {bov_path} not found", file=sys.stderr)
        return 1

    meta = parse_bov(bov_path)

    data_fmt = meta.get("DATA_FORMAT", "DOUBLE")
    ncomps = int(meta.get("DATA_COMPONENTS", "1"))
    if ncomps < 2:
        print(f"Nothing to split: {bov_path} has only {ncomps} component(s)", file=sys.stderr)
        return 0

    bricklets = parse_ints(meta.get("DATA_BRICKLETS", meta.get("DATA_SIZE", "1 1 1")))
    data_size = parse_ints(meta.get("DATA_SIZE", "1 1 1"))
    brick_origin = parse_floats(meta.get("BRICK_ORIGIN", "0 0 0"))
    brick_size = parse_floats(meta.get("BRICK_SIZE", "1 1 1"))
    time_val = meta.get("TIME", "0")

    data_file_pat = meta.get("DATA_FILE", "")
    if not data_file_pat:
        print("Error: no DATA_FILE in .bov header", file=sys.stderr)
        return 1

    nvars = len(args.var_names)
    if nvars != ncomps:
        print(f"Warning: {nvars} names provided but DATA_COMPONENTS={ncomps}; using indices",
              file=sys.stderr)
        names = [f"v{i}" for i in range(ncomps)]
    else:
        names = args.var_names

    bov_dir = os.path.dirname(bov_path)
    dat_dir = os.path.join(bov_dir, os.path.dirname(data_file_pat))
    dat_pat = os.path.basename(data_file_pat)

    if not os.path.isdir(dat_dir):
        print(f"Error: data directory {dat_dir} not found", file=sys.stderr)
        return 1

    rank_pat = dat_pat.replace("%04d", "[0-9][0-9][0-9][0-9]")
    dat_files = sorted(glob.glob(os.path.join(dat_dir, dat_pat.replace("%04d", "*"))))
    if not dat_files:
        rank_pat2 = dat_pat.replace("%d", "[0-9]+")
        dat_files = sorted(glob.glob(os.path.join(dat_dir, dat_pat.replace("%d", "*"))))

    if not dat_files:
        print(f"Error: no .dat files matching '{dat_pat}' in {dat_dir}", file=sys.stderr)
        return 1

    elem_size = fmt_size(data_fmt)
    ncell = bricklets[0] * bricklets[1] * bricklets[2]

    bov_out_dir = bov_dir
    dat_out_dir = os.path.join(bov_dir, os.path.dirname(data_file_pat))

    for dat_path in dat_files:
        fname = os.path.basename(dat_path)
        rank_m = re.search(r'_(\d+)\.dat$', fname)
        rank_str = rank_m.group(1) if rank_m else "0000"

        with open(dat_path, "rb") as f:
            raw = f.read()

        expected = ncell * ncomps * elem_size
        if len(raw) != expected:
            print(f"Warning: {dat_path}: expected {expected}B, got {len(raw)}B; skipping")
            continue

        data = struct.unpack(fmt_char(data_fmt) * ncell * ncomps, raw)

        for c in range(ncomps):
            comp = data[c::ncomps]
            out_name = fname.replace(".dat", f"_{names[c]}.dat")
            out_path = os.path.join(dat_out_dir, out_name)
            with open(out_path, "wb") as f:
                f.write(struct.pack(fmt_char(data_fmt) * ncell, *comp))
            print(f"  wrote {out_path}")

    bov_out_dir = bov_dir
    for c in range(ncomps):
        bov_name = os.path.basename(bov_path).replace(".bov", f"_{names[c]}.bov")
        bov_path_out = os.path.join(bov_dir, bov_name)
        if data_file_pat:
            new_data_file = os.path.join(
                os.path.dirname(data_file_pat),
                os.path.basename(data_file_pat).replace(".dat", f"_{names[c]}_%04d.dat")
            )
        else:
            new_data_file = ""
        with open(bov_path_out, "w") as f:
            f.write(f"TIME: {time_val}\n")
            f.write(f"DATA_FILE: {new_data_file}\n")
            f.write(f"DATA_SIZE: {' '.join(str(x) for x in data_size)}\n")
            f.write(f"DATA_FORMAT: {data_fmt}\n")
            f.write(f"VARIABLE: {names[c]}\n")
            f.write("DATA_ENDIAN: LITTLE\n")
            f.write("CENTERING: zonal\n")
            f.write(f"BRICK_ORIGIN: {' '.join(str(x) for x in brick_origin)}\n")
            f.write(f"BRICK_SIZE: {' '.join(str(x) for x in brick_size)}\n")
            f.write("DIVIDE_BRICK: false\n")
            f.write(f"DATA_BRICKLETS: {' '.join(str(x) for x in bricklets)}\n")
            f.write("DATA_COMPONENTS: 1\n")
        print(f"  wrote {bov_path_out}")

    print(f"Done: split {len(dat_files)} rank files into {ncomps} components each")

if __name__ == "__main__":
    sys.exit(main())
