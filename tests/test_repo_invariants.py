"""Repo-level structural invariants that CI must protect.

These are cheap, but each one encodes a mistake that is easy to make and
annoying to diagnose later.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGES = REPO / "packages"
EXPECTED = {"core", "gateway", "training", "eval", "client"}


def test_all_workspace_members_present() -> None:
    found = {p.name for p in PACKAGES.iterdir() if p.is_dir()}
    assert found == EXPECTED


def test_portcullis_is_a_namespace_package() -> None:
    """`portcullis/` must NOT contain __init__.py in any package.

    A stray __init__.py makes one member shadow the whole namespace, and only
    one of the five packages stays importable. The failure looks like a broken
    install rather than a packaging bug, so pin it here.
    """
    offenders = list(PACKAGES.glob("*/src/portcullis/__init__.py"))
    assert offenders == [], f"namespace shadowed by: {offenders}"


def test_no_raw_corpora_committed() -> None:
    """data/ carries manifests and hashes only — never the corpora themselves.

    Guards the licence and reproducibility story: raw rows must be rebuildable
    from a manifest, not vendored into git.
    """
    data = REPO / "data"
    allowed_suffixes = {".json", ".sha256", ".md"}
    bad = [
        p
        for p in data.rglob("*")
        if p.is_file() and p.name != ".gitkeep" and p.suffix not in allowed_suffixes
    ]
    assert bad == [], f"unexpected files under data/: {bad}"


def test_readme_publishes_no_unearned_metrics() -> None:
    """Guards the 'never fabricate a metric' rule mechanically.

    Until `just eval` exists (Milestone 9), every headline metric cell must read
    TBD. This fails loudly the moment someone hand-types a number into the
    benchmark table, which is exactly the failure mode the rule exists to stop.
    """
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    table = [ln for ln in readme.splitlines() if ln.startswith("| **TPR @")]
    assert table, "benchmark table rows not found - did the README structure change?"
    for row in table:
        cells = [c.strip() for c in row.split("|")[2:-1]]
        assert all(c == "TBD" for c in cells), f"non-TBD metric published: {row}"


def test_readme_does_not_report_balanced_accuracy() -> None:
    """Accuracy on a balanced set is barred from the README by project rule.

    It is the vanity metric this project argues against; a detector that blocks
    security researchers is broken regardless of its accuracy.
    """
    readme = (REPO / "README.md").read_text(encoding="utf-8").lower()
    offending = [
        ln for ln in readme.splitlines() if ln.startswith("|") and "accuracy" in ln and "tbd" in ln
    ]
    assert offending == [], f"accuracy row present in benchmark table: {offending}"
