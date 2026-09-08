"""Repo-level structural invariants that CI must protect.

These are cheap, but each one encodes a mistake that is easy to make and
annoying to diagnose later.
"""

import json
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
    """data/ is tracked in git as manifests and hashes only — never the
    corpora themselves.

    Guards the licence and reproducibility story: raw rows must be rebuildable
    from a manifest, not vendored into git. This checks what git actually
    *tracks*, not what exists on disk — `just data` legitimately writes real
    JSONL partitions into data/processed/ as build output, and finding them
    there is correct, not a violation. gitignore is what keeps them out of
    the repo; this test verifies gitignore's promise held, rather than
    re-asserting a stricter rule against the working tree that the pipeline
    itself would violate on every run.
    """
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "data/"],  # noqa: S607 - test-only, the repo's own git, not untrusted input
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    allowed_suffixes = {".json", ".sha256", ".md"}
    bad = [
        p for p in tracked if Path(p).name != ".gitkeep" and Path(p).suffix not in allowed_suffixes
    ]
    assert bad == [], f"unexpected tracked files under data/: {bad}"


def test_readme_metrics_are_traceable_to_a_real_eval_run() -> None:
    """Guards the 'never fabricate a metric' rule mechanically.

    Before Milestone 9 this test required every headline cell to read TBD,
    since `just eval` didn't exist yet. Milestone 9 built it, and the README
    now carries real numbers - so the guard evolves rather than disappears:
    every published number must be traceable to the real JSON `just eval`
    produced, not hand-typed. This fails loudly the moment someone edits a
    number in the README without it coming from a real eval-results.json -
    exactly the failure mode the original TBD check existed to stop.
    """
    results_path = REPO / "docs" / "benchmarks" / "eval-results.json"
    assert results_path.exists(), "eval-results.json missing - run `just eval` first"
    results = json.loads(results_path.read_text(encoding="utf-8"))

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    tpr_rows = [ln for ln in readme.splitlines() if ln.startswith("| **TPR @")]
    assert len(tpr_rows) == 2, "expected exactly two TPR rows - did the README structure change?"
    auprc_row = next(ln for ln in readme.splitlines() if ln.startswith("| AUPRC"))

    def fmt(x: float) -> str:
        return f"{x:.3f}"

    for row, fpr_key, source in (
        (tpr_rows[0], "tpr_at_fpr_0.100%", "headline"),
        (tpr_rows[1], "tpr_at_fpr_1.0%", "headline"),
    ):
        stat = results[source][fpr_key]
        for value in (stat["tpr"], stat["ci_low"], stat["ci_high"]):
            assert fmt(value) in row, f"{fmt(value)} (from eval-results.json) missing in: {row}"

    assert fmt(results["headline"]["auprc"]) in auprc_row, (
        f"headline AUPRC missing from README's AUPRC row: {auprc_row}"
    )

    for baseline_key, fpr_key, row in (
        ("max", "tpr_at_fpr_0.100%", tpr_rows[0]),
        ("max", "tpr_at_fpr_1.0%", tpr_rows[1]),
        ("regex_only", "tpr_at_fpr_0.100%", tpr_rows[0]),
        ("regex_only", "tpr_at_fpr_1.0%", tpr_rows[1]),
    ):
        stat = results["baselines"][baseline_key][fpr_key]
        assert fmt(stat["tpr"]) in row, (
            f"{baseline_key} TPR ({fmt(stat['tpr'])}) missing from: {row}"
        )


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
