#!/usr/bin/env python
"""Build a training index: a JSON list of absolute ``.b2nd`` volume paths.

The training scripts read volumes from a JSON index (``index_json`` in the config).  This
tool collects every file under ``--root`` that matches ``--pattern`` (a recursive glob),
sorts the absolute paths and writes them as a JSON list.  Sorting makes the index, and with
it the data order of a seeded run, independent of the file-system enumeration order.
Hidden files and directories (names starting with a dot) are skipped, as in shell globbing.

Example::

    python tools/make_index.py --root /data/preprocessed --out data/train_index.json
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.data import load_index  # noqa: E402


def make_index(root, pattern="**/*.b2nd"):
    """Sorted absolute paths of the files under ``root`` matching the recursive glob ``pattern``.

    Hidden files and directories (names starting with a dot) are not matched.
    """
    root = os.path.abspath(root)
    hits = glob.glob(os.path.join(root, pattern), recursive=True)
    return sorted(os.path.abspath(p) for p in hits if os.path.isfile(p))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write a JSON list of .b2nd volume paths.")
    ap.add_argument("--root", required=True, help="directory to search")
    ap.add_argument("--pattern", default="**/*.b2nd",
                    help="recursive glob relative to --root (hidden names are skipped)")
    ap.add_argument("--out", required=True, help="output JSON file")
    args = ap.parse_args(argv)

    paths = make_index(args.root, args.pattern)
    if not paths:
        raise SystemExit(f"make_index: no files matching {args.pattern!r} under {args.root}")
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(paths, f, indent=1)
        f.write("\n")
    n = len(load_index(out))          # round trip through the loader's reader
    print(f"[make_index] {n} volumes -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
