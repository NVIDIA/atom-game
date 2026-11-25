# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Put the SPDX header on every source file NVIDIA owns.

The header has to match the approved wording exactly, so the two lines are
written here once and every file gets the same pair. Running this again is
safe: a file that already carries the identifier is left alone.

    PYTHONPATH=. python tools/add_license_headers.py          # report only
    PYTHONPATH=. python tools/add_license_headers.py --write

Only files git tracks are considered, so nothing under outputs/ or analysis/ is
touched. Two files are skipped by name: they carry the attribution of Jonathan
Wurtz of QuEra Computing Inc., so an NVIDIA copyright is not the notice they
should bear, and what they should bear is a question for the review.
"""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COPYRIGHT = (
    "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
    "All rights reserved."
)
IDENTIFIER = "SPDX-License-Identifier: Apache-2.0"
# The marker that makes a second run a no-op.
MARKER = "SPDX-FileCopyrightText"

SKIP = set()

HASH_SUFFIXES = {".py", ".sh", ".yaml", ".yml"}


class Unreadable(Exception):
    """A file this script cannot edit without risking its contents."""


def tracked(suffixes):
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [
        name
        for name in out.split("\0")
        if name and Path(name).suffix in suffixes and name not in SKIP
    ]


def add_hash_header(path):
    """Two comment lines at the top, after a shebang if the file has one."""
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    header = f"# {COPYRIGHT}\n# {IDENTIFIER}\n"
    lines = text.split("\n")
    if lines and lines[0].startswith("#!"):
        # The shebang has to stay on the first line to keep working.
        body = "\n".join(lines[1:])
        path.write_text(lines[0] + "\n" + header + "\n" + body, encoding="utf-8")
    else:
        path.write_text(header + "\n" + text, encoding="utf-8")
    return True


def add_notebook_header(path):
    """A markdown cell of the two lines, first in the notebook.

    A notebook that is not valid JSON is left alone and reported: it cannot be
    edited safely, and it should not ship in that state either.
    """
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError as error:
        raise Unreadable(f"{path.relative_to(ROOT)} is not valid JSON: {error}")
    notebook.setdefault("cells", []).insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [COPYRIGHT + "\n", "\n", IDENTIFIER],
        },
    )
    # nbformat's own layout, so the diff is the inserted cell and nothing else.
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="edit the files rather than report"
    )
    args = parser.parse_args()

    plan = [(name, add_hash_header) for name in tracked(HASH_SUFFIXES)]
    plan += [(name, add_notebook_header) for name in tracked({".ipynb"})]

    added = already = 0
    unreadable = []
    for name, apply in sorted(plan):
        path = ROOT / name
        if MARKER in path.read_text(encoding="utf-8"):
            already += 1
            continue
        try:
            if args.write:
                apply(path)
        except Unreadable as error:
            unreadable.append(str(error))
            continue
        added += 1
        print(f"{'added  ' if args.write else 'would add'} {name}")

    print(f"\n{added} to change, {already} already carrying the header")
    for line in unreadable:
        print(f"UNREADABLE {line}")
    for name in sorted(SKIP):
        print(f"skipped {name}: externally authored, see the docstring")
    if not args.write:
        print("\nnothing written; pass --write")


if __name__ == "__main__":
    main()
