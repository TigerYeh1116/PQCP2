"""Audit/deduplicate root L.txt files, preserving first records and backups.

Default is read-only. Use --apply only when all searches writing these files
are stopped: the writer's Python lock cannot coordinate separate processes.
All selected files are validated before any replacement. Original bytes are
backed up first; malformed/invalid files cause a conservative failure.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import sys
from typing import List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.search_runner import _atomic_write_text
from solver.solution_equivalence import pqcp_equivalence_key
from solver.verifier import verify_pqcp


@dataclass(frozen=True)
class CleanupPlan:
    """Validated before/after snapshot; retained records are byte-for-byte text."""

    path: Path
    original: bytes
    cleaned: str
    original_count: int
    retained_count: int

    @property
    def removed_count(self) -> int:
        return self.original_count - self.retained_count


def plan_cleanup(path: Path) -> CleanupPlan:
    """Reject invalid records; only remove whole equivalent record blocks."""
    path = Path(path)
    if not path.stem.isdecimal() or path.suffix != ".txt" or path.is_symlink():
        raise ValueError("expected a regular numeric L.txt result file")
    L = int(path.stem)
    original = path.read_bytes()
    text = original.decode("utf-8")
    starts = list(re.finditer(r"^L=(\d+)[ \t]*\r?$", text, re.MULTILINE))
    if not starts:
        if text.strip(): raise ValueError("unrecognized nonempty result file: {}".format(path))
        return CleanupPlan(path, original, text, 0, 0)
    prefix = text[:starts[0].start()]
    if prefix.strip(): raise ValueError("unexpected text before first record")
    seen, kept = set(), []
    for i, start in enumerate(starts):
        block = text[start.start():starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        aa = re.findall(r"^a=([01]+)\r?$", block, re.MULTILINE)
        bb = re.findall(r"^b=([01]+)\r?$", block, re.MULTILINE)
        if int(start.group(1)) != L or len(aa) != 1 or len(bb) != 1:
            raise ValueError("malformed record {} in {}".format(i + 1, path))
        a, b = aa[0], bb[0]
        check = verify_pqcp(a, b)
        if len(a) != L or len(b) != L or not check.is_valid:
            raise ValueError("invalid PQCP record {} in {}".format(i + 1, path))
        shifts = re.findall(r"^nonzero shifts=([^\r\n]+)\r?$", block, re.MULTILINE)
        values = re.findall(r"^nonzero PACS=([^\r\n]+)\r?$", block, re.MULTILINE)
        if len(shifts) != 1 or len(values) != 1:
            raise ValueError("missing/duplicate correlation metadata")
        actual_values = tuple(check.profile[u] for u in check.nonzero_shifts)
        if (tuple(map(int, shifts[0].split(","))) != check.nonzero_shifts
                or tuple(map(int, values[0].split(","))) not in
                ((actual_values[0],), actual_values)):
            raise ValueError("incorrect correlation metadata in record {}".format(i + 1))
        key = pqcp_equivalence_key(a, b)
        if key not in seen:
            kept.append(block)
            seen.add(key)
    return CleanupPlan(path, original, prefix + "".join(kept), len(starts), len(kept))


def apply_cleanup(plans: List[CleanupPlan], backup_directory: Path) -> None:
    """Back up every changed file before atomic replacement; never overwrite backups."""
    changed = [p for p in plans if p.removed_count]
    if not changed: return
    backup_directory = Path(backup_directory)
    backup_directory.mkdir(parents=True, exist_ok=False)
    for plan in changed:
        if plan.path.read_bytes() != plan.original:
            raise RuntimeError("file changed during audit; stop searches and retry")
        backup = backup_directory / plan.path.name
        shutil.copy2(plan.path, backup)
        if backup.read_bytes() != plan.original:
            raise RuntimeError("backup consistency check failed")
    for plan in changed:
        if plan.path.read_bytes() != plan.original:
            raise RuntimeError("file changed after backup; refusing replacement")
        _atomic_write_text(plan.path, plan.cleaned)
        audited = plan_cleanup(plan.path)
        if audited.removed_count or audited.original_count != plan.retained_count:
            raise RuntimeError("post-cleanup verification failed; original backup retained")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--L", type=int, action="append", help="repeat to select specific lengths")
    parser.add_argument("--apply", action="store_true", help="replace after backup; stop searches first")
    args = parser.parse_args(argv)
    paths = ([args.root / "{}.txt".format(L) for L in args.L] if args.L else
             sorted((p for p in args.root.glob("*.txt") if p.stem.isdecimal()),
                    key=lambda p: int(p.stem)))
    plans = [plan_cleanup(p) for p in paths]
    backup = args.root / "results/backups" / (
        "equivalence_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    if args.apply:
        apply_cleanup(plans, backup)
    for plan in plans:
        print("{}: {} -> {} ({} equivalent records {})".format(
            plan.path.name, plan.original_count, plan.retained_count,
            plan.removed_count, "removed" if args.apply else "would be removed"))
    if args.apply and any(p.removed_count for p in plans): print("Backup:", backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
