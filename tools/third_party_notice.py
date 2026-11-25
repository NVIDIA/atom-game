# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build THIRD_PARTY_NOTICES.md from the licences the dependencies ship.

The notice has to be auditable, so nothing here is written by hand: every
licence text is copied out of the installed distribution's own metadata, and the
package list is read from ``requirements.txt`` rather than from whatever happens
to be installed. A package whose wheel carries no licence text is reported
rather than guessed at, so that it can be resolved by hand and not silently
mislabelled.

    PYTHONPATH=. python tools/third_party_notice.py

Writes THIRD_PARTY_NOTICES.md next to requirements.txt. Run it in the
environment the release is pinned against; the version each text was taken from
is recorded beside the pinned one so a reviewer can see when they differ.
"""

import importlib.metadata as md
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
NOTICE = ROOT / "THIRD_PARTY_NOTICES.md"

# Where a wheel ships no licence text, or ships one that is not the package's
# own. Each entry had to be read by hand, so each carries why.
BY_HAND = {
    # The only LICENSE file in the wheel is torch_geometric/graphgym/utils/,
    # which covers vendored GraphGym code and is empty in this build. The
    # package's own licence is the MIT one in its classifiers and repository.
    "torch-geometric": (
        "MIT",
        "https://github.com/pyg-team/pytorch_geometric/blob/master/LICENSE",
        "the wheel ships no licence text of its own",
    ),
}
# Metadata that contradicts, or fails to name, the shipped text. The text wins;
# these record the disagreement so that a reviewer reading the metadata is not
# surprised, and give the identifier the table should carry instead.
CONTRADICTED = {
    "bloqade": ("Apache-2.0", "the Python metadata says MIT, the shipped text is Apache-2.0"),
    "torch-ema": ("MIT", "the Python metadata declares no licence; the shipped text is MIT"),
}


def pinned():
    """Package name to pinned specifier, in the order requirements.txt lists."""
    out = {}
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[=><!~\[]", line, maxsplit=1)[0].strip()
        if not name:
            continue
        version = line[len(name):].lstrip("=><!~").strip()
        # A name repeated with and without a pin keeps the pin.
        if name not in out or (version and not out[name]):
            out[name] = version
    return out


def licence_text(name):
    """The licence text the installed distribution ships, if it ships one."""
    for path in md.files(name) or []:
        stem = str(path).split("/")[-1].upper()
        if stem.startswith(("LICENSE", "LICENCE", "COPYING")):
            try:
                # ``PackagePath.read_text`` takes no error handler, so the file
                # is read through its resolved location instead.
                text = Path(path.locate()).read_text(
                    encoding="utf-8", errors="replace"
                )
            except (FileNotFoundError, OSError):
                continue
            if text.strip():
                return str(path), text.rstrip()
    return None, None


def declared(name):
    """The licence to name in the summary table."""
    if name in CONTRADICTED:
        return CONTRADICTED[name][0]
    meta = md.metadata(name)
    for classifier in meta.get_all("Classifier") or []:
        if classifier.startswith("License ::"):
            return classifier.split("::")[-1].strip()
    value = (meta.get("License") or "").strip()
    return value if 0 < len(value) <= 40 else "see text"


def main():
    packages = pinned()
    rows, bodies, unresolved = [], [], []

    for name, version in packages.items():
        try:
            installed = md.version(name)
        except md.PackageNotFoundError:
            unresolved.append(f"{name}: not installed, licence not verified")
            rows.append((name, version or "unpinned", "NOT VERIFIED", ""))
            continue

        if name in BY_HAND:
            identifier, url, why = BY_HAND[name]
            rows.append((name, version or installed, identifier, f"[text]({url})"))
            bodies.append(
                f"## {name} {version or installed}\n\n"
                f"Licence: {identifier}. Taken from {url}, because {why}.\n"
            )
            continue

        source, text = licence_text(name)
        if text is None:
            unresolved.append(f"{name}: installed but ships no licence text")
            rows.append((name, version or installed, "NOT VERIFIED", ""))
            continue

        note = ""
        if version and version != installed:
            note = f" Text read from the installed {installed}; {version} is pinned."
        if name in CONTRADICTED:
            note += f" Note: {CONTRADICTED[name][1]}."
        rows.append((name, version or installed, declared(name), "below"))
        bodies.append(
            f"## {name} {version or installed}\n\n"
            f"Licence text as distributed in `{source}`.{note}\n\n"
            "```\n" + text + "\n```\n"
        )

    width = max(len(r[0]) for r in rows)
    table = [
        f"| {'Package'.ljust(width)} | Version | Licence | Text |",
        f"| {'-' * width} | --- | --- | --- |",
    ]
    for name, version, identifier, where in rows:
        table.append(f"| {name.ljust(width)} | {version} | {identifier} | {where} |")

    header = [
        "# Third party notices",
        "",
        "Atom Game depends on the packages below. It bundles none of them: they are",
        "installed from the Python Package Index by `requirements.txt`, and this file",
        "reproduces their licences as the notice each one requires.",
        "",
        "Generated by `tools/third_party_notice.py`, which copies each text out of the",
        "installed distribution rather than restating it, so that this file can be",
        "checked against the environment it was built from.",
        "",
        *table,
        "",
    ]
    if unresolved:
        header += [
            "## Unresolved",
            "",
            "These could not be read from an installed distribution and need a licence",
            "established by hand before release:",
            "",
            *(f"- {line}" for line in unresolved),
            "",
        ]

    NOTICE.write_text("\n".join(header) + "\n" + "\n".join(bodies))
    print(f"wrote {NOTICE} covering {len(rows)} packages")
    for line in unresolved:
        print(f"  unresolved: {line}")


if __name__ == "__main__":
    main()
