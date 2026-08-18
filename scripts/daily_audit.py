#!/usr/bin/env python3
"""100-point daily audit for the cats-dogs-mlops project.

Run this at the end of every working day. It is the enforcement mechanism for
``tracker/GUARDRAILS.md`` — especially the consistency checks that catch the
class of mistake that cost marks on Assignment 1 (artifacts that exist but
don't agree with each other, or don't exist where a grader would look).

    python scripts/daily_audit.py --day 3
    python scripts/daily_audit.py --day 10 --verbose      # full pre-submission
    python scripts/daily_audit.py --day 3 --only CONSISTENCY

Every check declares the day it becomes active, so early days report
NOT-YET instead of drowning you in irrelevant failures. Checks that need human
eyes (screenshots, GitHub UI) report MANUAL with what to look at.

Exit code is 0 only when nothing that is *active* has FAILED.

Stdlib only, on purpose: this must run before any dependency is installed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PASS, FAIL, MANUAL, NOTYET, SKIP = "PASS", "FAIL", "MANUAL", "NOT-YET", "SKIP"

C = {
    PASS: "\033[32m", FAIL: "\033[31m", MANUAL: "\033[33m",
    NOTYET: "\033[90m", SKIP: "\033[90m",
    "bold": "\033[1m", "dim": "\033[2m", "reset": "\033[0m",
}
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    C = {k: "" for k in C}


@dataclass
class Result:
    num: int
    group: str
    title: str
    status: str
    detail: str = ""
    gap: str = ""          # "CV" or "CICD" -> the A1 mark-loss tags


RESULTS: list[Result] = []
_REGISTRY: list[tuple] = []


def check(num: int, group: str, title: str, day: int, gap: str = ""):
    """Register a check. ``day`` is the day it becomes active."""
    def deco(fn):
        _REGISTRY.append((num, group, title, day, gap, fn))
        return fn
    return deco


# ---------------------------------------------------------------- helpers

def exists(*rel: str) -> bool:
    return all((ROOT / r).exists() for r in rel)


def read(rel: str) -> str:
    p = ROOT / rel
    return p.read_text(errors="replace") if p.is_file() else ""


def pins(rel: str) -> dict[str, str]:
    """Parse ``pkg==ver`` lines from a requirements file, lowercased keys."""
    out = {}
    for line in read(rel).splitlines():
        line = line.split("#")[0].strip()
        m = re.match(r"^([A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*==\s*([A-Za-z0-9._+!-]+)$", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20)
        return r.stdout.strip()
    except Exception:
        return ""


def tracked(path: str) -> bool:
    return bool(git("ls-files", "--", path).strip())


def json_at(rel: str) -> dict | None:
    try:
        return json.loads(read(rel))
    except Exception:
        return None


def rows(rel: str) -> list[dict]:
    txt = read(rel)
    if not txt.strip():
        return []
    try:
        return list(csv.DictReader(txt.splitlines()))
    except Exception:
        return []


def ok(cond: bool, good: str = "", bad: str = "") -> tuple[str, str]:
    return (PASS, good) if cond else (FAIL, bad or good)


# ============================================================ A. STRUCTURE (1-10)

@check(1, "STRUCTURE", "Core directories exist", 1)
def _1():
    need = ["src", "api", "tests", "scripts", "models", "k8s", "monitoring",
            "reports/figures", "notebooks", "tracker", ".github/workflows"]
    missing = [d for d in need if not (ROOT / d).is_dir()]
    return ok(not missing, "all present", f"missing: {', '.join(missing)}")


@check(2, "STRUCTURE", "Git repo initialised on branch main", 1)
def _2():
    b = git("rev-parse", "--abbrev-ref", "HEAD")
    return ok(b in ("main", "HEAD"), f"branch={b or 'unborn'}", f"branch={b!r}, expected main")


@check(3, "STRUCTURE", "At least one commit exists", 1)
def _3():
    n = git("rev-list", "--count", "HEAD")
    return ok(n.isdigit() and int(n) > 0, f"{n} commits", "no commits yet")


@check(4, "STRUCTURE", "Git remote configured", 1)
def _4():
    r = git("remote", "-v")
    return ok(bool(r), "remote set", "no remote — GitHub repo not linked")


@check(5, "STRUCTURE", "README.md exists and is substantive", 1)
def _5():
    n = len(read("README.md").splitlines())
    return ok(n >= 30, f"{n} lines", f"only {n} lines")


@check(6, "STRUCTURE", "tracker/ files all present", 1)
def _6():
    need = ["PROGRESS.md", "TASKS.md", "DECISIONS.md", "EVIDENCE.md",
            "DAILY_LOG.md", "GUARDRAILS.md"]
    missing = [f for f in need if not (ROOT / "tracker" / f).is_file()]
    return ok(not missing, "all present", f"missing: {', '.join(missing)}")


@check(7, "STRUCTURE", "ruff.toml / pytest.ini / conftest.py present", 1)
def _7():
    missing = [f for f in ["ruff.toml", "pytest.ini", "conftest.py"] if not exists(f)]
    return ok(not missing, "all present", f"missing: {', '.join(missing)}")


@check(8, "STRUCTURE", ".python-version pins 3.11", 1)
def _8():
    v = read(".python-version").strip()
    return ok(v.startswith("3.11"), f"={v}", f"={v!r}, expected 3.11")


@check(9, "STRUCTURE", "DAILY_LOG.md has an entry for the current day", 1)
def _9():
    return (MANUAL, "confirm today's entry is appended (what worked / blockers / commits)")


@check(10, "STRUCTURE", "Working tree clean (no uncommitted work)", 1)
def _10():
    s = git("status", "--porcelain")
    n = len([x for x in s.splitlines() if x.strip()])
    return ok(n == 0, "clean", f"{n} uncommitted path(s)")


# ============================================ B. ENVIRONMENT & PIN CONSISTENCY (11-22)

@check(11, "CONSISTENCY", "requirements.txt exists", 1)
def _11():
    return ok(exists("requirements.txt"), "present", "missing")


@check(12, "CONSISTENCY", "requirements-serve.txt exists", 6)
def _12():
    return ok(exists("requirements-serve.txt"), "present", "missing")


@check(13, "CONSISTENCY", "Every requirement is version-pinned (==)", 1)
def _13():
    bad = []
    for f in ("requirements.txt", "requirements-serve.txt"):
        if not exists(f):
            continue
        for line in read(f).splitlines():
            line = line.split("#")[0].strip()
            if line and "==" not in line and not line.startswith("-"):
                bad.append(f"{f}:{line}")
    return ok(not bad, "all pinned", f"unpinned: {'; '.join(bad[:5])}")


@check(14, "CONSISTENCY", "TF/numpy/Pillow pins IDENTICAL across both requirements files", 6)
def _14():
    """The ADR-002 trap: a skew here breaks model.h5 loading in the container."""
    a, b = pins("requirements.txt"), pins("requirements-serve.txt")
    if not b:
        return (NOTYET, "requirements-serve.txt not written yet")
    crit = ["numpy", "pillow", "h5py"]
    diffs = []
    for p in crit:
        if p in a and p in b and a[p] != b[p]:
            diffs.append(f"{p}: train={a[p]} serve={b[p]}")
    tf_t = a.get("tensorflow") or a.get("tensorflow-cpu")
    tf_s = b.get("tensorflow-cpu") or b.get("tensorflow")
    if tf_t and tf_s and tf_t != tf_s:
        diffs.append(f"tensorflow: train={tf_t} serve={tf_s}")
    return ok(not diffs, f"aligned (tf={tf_t})", f"SKEW -> {'; '.join(diffs)}")


@check(15, "CONSISTENCY", "tensorflow-cpu version actually exists on PyPI for linux", 6)
def _15():
    """tensorflow-cpu lags tensorflow. Verified on Day 1: cpu maxes at 2.20.0."""
    b = pins("requirements-serve.txt")
    v = b.get("tensorflow-cpu")
    if not v:
        return (NOTYET, "tensorflow-cpu not pinned yet")
    try:
        maj, mnr = (int(x) for x in v.split(".")[:2])
    except Exception:
        return (FAIL, f"unparseable version {v!r}")
    return ok((maj, mnr) <= (2, 20),
              f"tensorflow-cpu=={v} is available for linux",
              f"tensorflow-cpu=={v} does NOT exist (cpu builds stop at 2.20.x)")


@check(16, "CONSISTENCY", "Installed versions match requirements.txt pins", 1)
def _16():
    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        return (NOTYET, ".venv not created yet")
    want = pins("requirements.txt")
    if not want:
        return (NOTYET, "requirements.txt has no pins yet")
    try:
        out = subprocess.run(
            [str(py), "-c",
             "import importlib.metadata as m,json;"
             "print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version "
             "for d in m.distributions() if d.metadata['Name']}))"],
            capture_output=True, text=True, timeout=60)
        have = json.loads(out.stdout or "{}")
    except Exception as e:
        return (FAIL, f"could not introspect venv: {e}")
    drift = [f"{p}: want {v}, have {have[p]}"
             for p, v in want.items() if p in have and have[p] != v]
    absent = [p for p in want if p not in have]
    if drift:
        return (FAIL, f"DRIFT -> {'; '.join(drift[:4])}")
    return ok(not absent, f"{len(want)} pins match installed",
              f"not installed: {', '.join(absent[:5])}")


@check(17, "CONSISTENCY", ".venv exists and imports tensorflow", 1)
def _17():
    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        return (FAIL, ".venv/bin/python missing")
    r = subprocess.run([str(py), "-c", "import tensorflow as tf;print(tf.__version__)"],
                       capture_output=True, text=True, timeout=180)
    v = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    return ok(r.returncode == 0 and v, f"tensorflow {v}", "tensorflow import failed")


@check(18, "CONSISTENCY", "model_metadata.json package versions match requirements.txt", 5)
def _18():
    meta = json_at("models/model_metadata.json")
    if meta is None:
        return (NOTYET, "model not trained yet")
    pv = meta.get("package_versions", {})
    if not pv:
        return (FAIL, "metadata has no package_versions block")
    want = pins("requirements.txt")
    diffs = [f"{k}: meta={v} req={want[k.lower()]}"
             for k, v in pv.items()
             if k.lower() in want and want[k.lower()] != v]
    return ok(not diffs, f"{len(pv)} versions agree",
              f"model trained with different versions -> {'; '.join(diffs[:4])}")


@check(19, "CONSISTENCY", "IMG_SIZE is defined once and is 224x224", 2)
def _19():
    txt = read("src/data.py")
    if not txt:
        return (NOTYET, "src/data.py not written yet")
    m = re.search(r"IMG_SIZE\s*[:=].*?\(?\s*(\d+)\s*,\s*(\d+)", txt)
    if not m:
        return (FAIL, "IMG_SIZE not found in src/data.py")
    got = (int(m.group(1)), int(m.group(2)))
    dupes = [f for f in ("src/preprocess.py", "src/model.py", "api/main.py")
             if re.search(r"^\s*IMG_SIZE\s*=", read(f), re.M)]
    if got != (224, 224):
        return (FAIL, f"IMG_SIZE={got}, spec requires (224, 224)")
    return ok(not dupes, "224x224, single definition",
              f"224x224 but REDEFINED in: {', '.join(dupes)} — import it instead")


@check(20, "CONSISTENCY", "IMG_SIZE in metadata matches src/data.py", 5)
def _20():
    meta = json_at("models/model_metadata.json")
    if meta is None:
        return (NOTYET, "model not trained yet")
    shape = meta.get("input_shape") or meta.get("img_size")
    if not shape:
        return (FAIL, "metadata records no input_shape")
    flat = [int(x) for x in re.findall(r"\d+", json.dumps(shape))]
    return ok(224 in flat, f"input_shape={shape}", f"input_shape={shape} lacks 224")


@check(21, "CONSISTENCY", "CLASS_NAMES consistent between code and metadata", 5)
def _21():
    txt = read("src/data.py")
    meta = json_at("models/model_metadata.json")
    if not txt or meta is None:
        return (NOTYET, "data.py or metadata not ready")
    code = re.findall(r"[\"']([a-z]+)[\"']", re.search(
        r"CLASS_NAMES\s*=\s*\[([^\]]*)\]", txt).group(1)) if re.search(
        r"CLASS_NAMES\s*=\s*\[([^\]]*)\]", txt) else []
    mc = meta.get("class_names") or list((meta.get("class_indices") or {}).keys())
    if not code or not mc:
        return (FAIL, f"code={code}, metadata={mc} — one is empty")
    return ok([str(x).lower() for x in code] == [str(x).lower() for x in mc],
              f"{code}", f"MISMATCH code={code} metadata={mc}")


@check(22, "CONSISTENCY", "Preprocessing shared between train and serve (no duplication)", 6)
def _22():
    pred, api = read("src/predict.py"), read("api/main.py")
    if not pred:
        return (NOTYET, "src/predict.py not written yet")
    shares = "preprocess" in pred or "preprocess" in api
    return ok(shares, "predict path imports src.preprocess",
              "serving does NOT reuse src/preprocess — train/serve skew risk")


# ============================================================ C. SECRETS (23-30)

@check(23, "SECRETS", ".gitignore exists", 1)
def _23():
    return ok(exists(".gitignore"), "present", "missing")


@check(24, "SECRETS", ".gitignore covers Kaggle credentials", 1)
def _24():
    g = read(".gitignore")
    need = ["access_token", "kaggle.json"]
    missing = [n for n in need if n not in g]
    return ok(not missing, "access_token + kaggle.json ignored",
              f"missing from .gitignore: {', '.join(missing)}")


@check(25, "SECRETS", ".gitignore covers .venv / mlruns / data / dvc cache", 1)
def _25():
    g = read(".gitignore")
    missing = [n for n in [".venv", "mlruns", "data/raw", ".dvc/cache"] if n not in g]
    return ok(not missing, "all ignored", f"missing: {', '.join(missing)}")


@check(26, "SECRETS", "No credential files tracked by git", 1)
def _26():
    bad = [p for p in git("ls-files").splitlines()
           if re.search(r"(access_token|kaggle\.json|\.pem$|id_rsa|\.env$)", p)]
    return ok(not bad, "none tracked", f"TRACKED CREDENTIALS: {', '.join(bad)}")


@check(27, "SECRETS", "No Kaggle token literal in the repo", 1)
def _27():
    hits = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or any(x in p.parts for x in
                                  (".git", ".venv", "node_modules", "mlruns", ".dvc")):
            continue
        if p.suffix in (".py", ".sh", ".yml", ".yaml", ".md", ".txt", ".json", ".cfg", ".ini"):
            if re.search(r"KGAT_[A-Za-z0-9]{16,}", p.read_text(errors="replace")):
                hits.append(str(p.relative_to(ROOT)))
    return ok(not hits, "no token literals", f"TOKEN LEAKED IN: {', '.join(hits)}")


@check(28, "SECRETS", "No AWS/generic secret literals committed", 1)
def _28():
    pat = re.compile(r"(AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY)")
    hits = []
    for rel in git("ls-files").splitlines():
        p = ROOT / rel
        if p.is_file() and p.stat().st_size < 2_000_000:
            if pat.search(p.read_text(errors="replace")):
                hits.append(rel)
    return ok(not hits, "clean", f"SECRETS IN: {', '.join(hits)}")


@check(29, "SECRETS", "API does not log image contents (spec: exclude sensitive data)", 6)
def _29():
    txt = read("api/main.py")
    if not txt:
        return (NOTYET, "api/main.py not written yet")
    bad = re.search(r"log(?:ger)?\.\w+\([^)]*\b(contents|raw_bytes|image_bytes|body)\b", txt)
    has_safe = re.search(r"(size|len\(|hash|sha)", txt)
    if bad:
        return (FAIL, "logging call appears to include raw image data")
    return ok(bool(has_safe), "logs size/hash only",
              "no evidence of size/hash logging — confirm manually")


@check(30, "SECRETS", "README documents the no-sensitive-data logging policy", 10)
def _30():
    r = read("README.md").lower()
    return ok(("sensitive" in r or "no image" in r) and "log" in r,
              "policy stated", "README does not state the logging policy (spec asks for it)")


# ============================================================ D. DATA & DVC (31-42)

@check(31, "DATA", "Kaggle token installed at ~/.kaggle/access_token", 1)
def _31():
    p = Path.home() / ".kaggle" / "access_token"
    if not p.exists():
        return (FAIL, "~/.kaggle/access_token missing")
    mode = oct(p.stat().st_mode)[-3:]
    return ok(mode == "600", f"present, mode {mode}", f"present but mode {mode}, want 600")


@check(32, "DATA", "scripts/download.sh exists and is executable", 1)
def _32():
    p = ROOT / "scripts" / "download.sh"
    if not p.exists():
        return (NOTYET, "not written yet")
    return ok(os.access(p, os.X_OK), "executable", "not executable (chmod +x)")


@check(33, "DATA", "data/raw holds the REAL dataset, not synthetic fixtures", 3)
def _33():
    """Day 1-2 legitimately run on synthetic smoke data; from day 3 it must be real.

    Split out from the docstring-counts check because shipping metrics derived
    from synthetic images would be a far worse failure than a thin docstring —
    it would put fabricated accuracy in model_metadata.json and the README.
    """
    if not (ROOT / "data" / "raw").exists():
        return (FAIL, "data/raw missing — run bash scripts/download.sh")
    if (ROOT / "data" / "raw" / ".synthetic").exists():
        return (FAIL, "data/raw is SYNTHETIC smoke data — any metrics from it are "
                      "meaningless. Run: bash scripts/download.sh")
    n = len([p for p in (ROOT / "data" / "raw").rglob("*")
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    return ok(n > 1000, f"{n:,} real images",
              f"only {n} images — expected ~25k for this dataset")


@check(34, "DATA", "Corrupt-image audit ran and produced a report", 1)
def _34():
    if not exists("data/corrupt_files.txt"):
        return (NOTYET, "audit not run yet")
    n = len([x for x in read("data/corrupt_files.txt").splitlines() if x.strip()])
    return (PASS, f"audit report present ({n} corrupt files listed)")


@check(35, "DATA", "DVC initialised", 2)
def _35():
    return ok(exists(".dvc/config"), "present", ".dvc/config missing — dvc init not run")


@check(36, "DATA", "DVC remote configured", 2)
def _36():
    cfg = read(".dvc/config")
    if not cfg:
        return (NOTYET, "dvc not initialised yet")
    return ok("remote" in cfg and "url" in cfg, "remote set", "no remote in .dvc/config")


@check(37, "DATA", "Raw data tracked by DVC", 2)
def _37():
    found = list(ROOT.glob("data/*.dvc")) + list(ROOT.glob("*.dvc"))
    return ok(bool(found) or "outs" in read("dvc.yaml"),
              f"{len(found)} .dvc file(s)", "no .dvc files and no dvc.yaml outs")


@check(38, "DATA", "dvc.yaml declares pipeline stages", 3)
def _38():
    """Activates on day 3, not day 2.

    The `train` stage cannot exist until src/train.py does, and declaring a stage
    whose script is missing breaks `dvc repro` outright — so requiring two stages
    on day 2 would push toward writing a broken manifest to satisfy an audit.
    """
    y = read("dvc.yaml")
    if not y:
        return (NOTYET, "dvc.yaml not written yet")
    have = [s for s in ("preprocess", "train", "cross_validate")
            if re.search(rf"^\s+{s}\s*:", y, re.MULTILINE)]
    return ok(len(have) >= 2, f"stages: {', '.join(have)}",
              f"only found: {have or 'none'} — want at least preprocess + train")


@check(39, "DATA", "dvc.lock committed", 2)
def _39():
    if not exists("dvc.lock"):
        return (NOTYET, "dvc repro not run yet")
    return ok(tracked("dvc.lock"), "tracked", "exists but NOT committed")


@check(40, "DATA", "No large binaries committed to git", 1)
def _40():
    big = []
    for rel in git("ls-files").splitlines():
        p = ROOT / rel
        if p.is_file() and p.stat().st_size > 15_000_000:
            big.append(f"{rel} ({p.stat().st_size // 1_000_000}MB)")
    return ok(not big, "none over 15MB", f"LARGE FILES COMMITTED: {', '.join(big)}")


@check(41, "DATA", "dvc status clean", 2)
def _41():
    """Runs dvc with .venv/bin prepended to PATH, deliberately.

    DVC stages use bare ``cmd: python -m ...``, which resolves against PATH at
    run time. On this machine an unactivated shell resolves ``python`` to Apple's
    system Python 2.7, so the stage dies with a SyntaxError on modern type hints.
    Prepending the venv here makes this check reflect the documented workflow
    (README says to activate the venv) rather than the ambient shell.
    """
    if not exists(".dvc/config"):
        return (NOTYET, "dvc not initialised")
    dvc = ROOT / ".venv" / "bin" / "dvc"
    if not dvc.exists():
        return (MANUAL, "run `dvc status` — dvc not in .venv")
    venv_bin = ROOT / ".venv" / "bin"
    env = {**os.environ, "PATH": f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"}
    r = subprocess.run([str(dvc), "status"], cwd=ROOT, capture_output=True,
                       text=True, timeout=300, env=env, check=False)
    clean = "up to date" in (r.stdout + r.stderr).lower()
    return ok(clean, "up to date", f"drift: {r.stdout.strip()[:200]}")


@check(42, "DATA", "Split manifests exist for train/val/test", 2)
def _42():
    missing = [s for s in ("train", "val", "test")
               if not exists(f"data/processed/{s}.csv")]
    return (NOTYET, "preprocessing not run yet") if len(missing) == 3 else \
        ok(not missing, "all three present", f"missing: {', '.join(missing)}")


# ============================================================ E. PREPROCESSING (43-52)

@check(43, "PREPROCESS", "src/preprocess.py exists", 2)
def _43():
    return ok(exists("src/preprocess.py"), "present", "missing")


@check(44, "PREPROCESS", "Splits are 80/10/10 within tolerance", 2)
def _44():
    counts = {s: len(rows(f"data/processed/{s}.csv")) for s in ("train", "val", "test")}
    if not all(counts.values()):
        return (NOTYET, "manifests not built yet")
    tot = sum(counts.values())
    pct = {s: 100 * n / tot for s, n in counts.items()}
    bad = (abs(pct["train"] - 80) > 2 or abs(pct["val"] - 10) > 2
           or abs(pct["test"] - 10) > 2)
    desc = ", ".join(f"{s}={pct[s]:.1f}%" for s in pct)
    return ok(not bad, f"{desc} (n={tot})", f"OFF TARGET: {desc}")


@check(45, "PREPROCESS", "Splits are disjoint (no file in two splits)", 2)
def _45():
    sets = {}
    for s in ("train", "val", "test"):
        rr = rows(f"data/processed/{s}.csv")
        if not rr:
            return (NOTYET, "manifests not built yet")
        key = "filepath" if "filepath" in rr[0] else list(rr[0])[0]
        sets[s] = {r[key] for r in rr}
    overlaps = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n = len(sets[a] & sets[b])
        if n:
            overlaps.append(f"{a}&{b}={n}")
    return ok(not overlaps, "disjoint", f"LEAKAGE: {', '.join(overlaps)}")


@check(46, "PREPROCESS", "Stratification holds across splits (class balance within 2%)", 2)
def _46():
    """Tolerance must scale with split size.

    A split of n rows can only express class proportions in steps of 100/n
    percent, so on a small split a "failure" can be arithmetically unavoidable
    rather than a stratification bug (the 120-image synthetic fixture has an
    11-row val split, where one image is 9.1%). The gate is therefore the larger
    of 2% and the coarsest achievable granularity, so this check stays meaningful
    on the real ~25k-image dataset without crying wolf on the smoke fixture.
    """
    bal, sizes = {}, {}
    for s in ("train", "val", "test"):
        rr = rows(f"data/processed/{s}.csv")
        if not rr or "label" not in rr[0]:
            return (NOTYET, "manifests or label column not ready")
        labs = [r["label"] for r in rr]
        sizes[s] = len(labs)
        bal[s] = 100 * sum(1 for x in labs if str(x) in ("1", "dog")) / len(labs)
    spread = max(bal.values()) - min(bal.values())
    granularity = 100.0 / min(sizes.values())
    tol = max(2.0, granularity)
    desc = ", ".join(f"{s}={bal[s]:.1f}% (n={sizes[s]})" for s in bal)
    return ok(spread <= tol,
              f"spread {spread:.1f}% <= tol {tol:.1f}%; {desc}",
              f"IMBALANCED: spread {spread:.1f}% > tol {tol:.1f}%; {desc}")


@check(47, "PREPROCESS", "Corrupt files excluded from the manifests", 2)
def _47():
    corrupt = {x.strip() for x in read("data/corrupt_files.txt").splitlines() if x.strip()}
    if not corrupt:
        return (NOTYET, "no corrupt list (run the Day 1 audit)")
    listed = set()
    for s in ("train", "val", "test"):
        for r in rows(f"data/processed/{s}.csv"):
            listed.add(next(iter(r.values())))
    names = {Path(c).name for c in corrupt}
    leaked = [n for n in names if any(n in entry for entry in listed)]
    return ok(not leaked, f"{len(corrupt)} excluded",
              f"CORRUPT FILES IN MANIFESTS: {len(leaked)}")


@check(48, "PREPROCESS", "Augmentation applied to train only", 2)
def _48():
    txt = read("src/preprocess.py")
    if not txt:
        return (NOTYET, "not written yet")
    return ok("augment" in txt and re.search(r"if\s+augment", txt),
              "gated on an augment flag",
              "no `if augment:` gate — augmentation may leak into val/test")


@check(49, "PREPROCESS", "Images forced to RGB (handles greyscale/alpha)", 2)
def _49():
    txt = read("src/preprocess.py")
    if not txt:
        return (NOTYET, "not written yet")
    return ok('"RGB"' in txt or "'RGB'" in txt or "channels=3" in txt,
              "RGB conversion present", "no RGB conversion — greyscale/RGBA will break shapes")


@check(50, "PREPROCESS", "RANDOM_STATE=42 defined and used", 2)
def _50():
    txt = read("src/data.py")
    if not txt:
        return (NOTYET, "not written yet")
    return ok(re.search(r"RANDOM_STATE\s*[:=]\s*42", txt), "=42",
              "RANDOM_STATE not set to 42 — reproducibility at risk")


@check(51, "PREPROCESS", "EDA figures produced", 2)
def _51():
    have = [f for f in ("class_balance.png", "sample_grid.png", "augmentation_grid.png")
            if exists(f"reports/figures/{f}")]
    return ok(len(have) >= 3, f"{len(have)}/3 present",
              f"only {len(have)}/3: have {have}")


@check(52, "PREPROCESS", "notebooks/01_eda.ipynb exists (M1 wants notebooks in Git)", 2)
def _52():
    if not exists("notebooks/01_eda.ipynb"):
        return (NOTYET, "not created yet")
    return ok(tracked("notebooks/01_eda.ipynb"), "tracked", "exists but not committed")


# ============================================================ F. MODEL (53-62)

@check(53, "MODEL", "src/model.py defines two architectures", 3)
def _53():
    txt = read("src/model.py")
    if not txt:
        return (NOTYET, "not written yet")
    have = re.findall(r"def\s+(build_\w+)", txt)
    return ok(len(have) >= 2, f"{have}", f"only {have} — need baseline + transfer")


@check(54, "MODEL", "src/train.py exists", 3)
def _54():
    return ok(exists("src/train.py"), "present", "missing")


@check(55, "MODEL", "models/model.h5 exists", 3)
def _55():
    if not exists("models/model.h5"):
        return (NOTYET, "not trained yet")
    mb = (ROOT / "models/model.h5").stat().st_size / 1e6
    return (PASS, f"present ({mb:.1f}MB)")


@check(56, "MODEL", "model.h5 is COMMITTED to git (grader can't dvc pull)", 3)
def _56():
    if not exists("models/model.h5"):
        return (NOTYET, "not trained yet")
    return ok(tracked("models/model.h5"), "tracked",
              "NOT committed — the submission zip would ship without a model")


@check(57, "MODEL", "model.h5 reloads and predicts", 3)
def _57():
    py = ROOT / ".venv" / "bin" / "python"
    if not exists("models/model.h5") or not py.exists():
        return (NOTYET, "model or venv not ready")
    code = ("import keras,numpy as np;m=keras.models.load_model('models/model.h5');"
            "p=m.predict(np.random.rand(1,224,224,3).astype('float32'),verbose=0);"
            "print('OK',p.shape)")
    r = subprocess.run([str(py), "-c", code], cwd=ROOT, capture_output=True,
                       text=True, timeout=300)
    return ok("OK" in r.stdout, "loads and predicts",
              f"LOAD FAILED: {(r.stderr or '')[-200:]}")


@check(58, "MODEL", "models/model_metadata.json exists", 3)
def _58():
    return (PASS, "present") if exists("models/model_metadata.json") \
        else (NOTYET, "not trained yet")


@check(59, "MODEL", "Metadata records hyperparameters and metrics", 3)
def _59():
    meta = json_at("models/model_metadata.json")
    if meta is None:
        return (NOTYET, "not trained yet")
    missing = [k for k in ("hyperparameters", "metrics", "package_versions")
               if k not in meta]
    return ok(not missing, "complete", f"missing keys: {', '.join(missing)}")


@check(60, "MODEL", "Test accuracy beats chance by a clear margin", 3)
def _60():
    meta = json_at("models/model_metadata.json")
    if meta is None:
        return (NOTYET, "not trained yet")
    nums = re.findall(r'"(?:test_)?accuracy"\s*:\s*([0-9.]+)', json.dumps(meta))
    if not nums:
        return (FAIL, "no accuracy in metadata")
    acc = max(float(n) for n in nums)
    return ok(acc > 0.70, f"accuracy={acc:.4f}", f"accuracy={acc:.4f} too close to chance")


@check(61, "MODEL", "Training figures produced (loss + accuracy curves)", 3)
def _61():
    have = [f for f in ("loss_curves.png", "accuracy_curves.png",
                        "confusion_matrix.png", "roc_curve.png")
            if exists(f"reports/figures/{f}")]
    return ok(len(have) >= 4, f"{len(have)}/4", f"{len(have)}/4: have {have}")


@check(62, "MODEL", "Per-epoch wall-clock recorded (sizes the CV budget)", 3)
def _62():
    return (MANUAL, "confirm epoch timing is noted in DAILY_LOG.md (drives the Day 4 CV protocol)")


# ================================================ G. CROSS-VALIDATION [GAP-CV] (63-74)

@check(63, "CROSS-VAL", "src/cross_validate.py exists as its OWN module", 4, gap="CV")
def _63():
    return ok(exists("src/cross_validate.py"), "present",
              "MISSING — CV must not be a side effect of tuning (A1's mistake)")


@check(64, "CROSS-VAL", "Uses StratifiedKFold with 5 splits", 4, gap="CV")
def _64():
    txt = read("src/cross_validate.py")
    if not txt:
        return (NOTYET, "not written yet")
    return ok("StratifiedKFold" in txt and re.search(r"n_splits\s*=\s*5", txt),
              "StratifiedKFold(n_splits=5)", "no StratifiedKFold(n_splits=5)")


@check(65, "CROSS-VAL", "Model rebuilt AND recompiled per fold (no leakage)", 4, gap="CV")
def _65():
    txt = read("src/cross_validate.py")
    if not txt:
        return (NOTYET, "not written yet")
    return ok("compile" in txt and re.search(r"build_\w+\(", txt),
              "rebuild + recompile present",
              "no per-fold rebuild/recompile — weights would leak across folds")


@check(66, "CROSS-VAL", "reports/cv_results.csv exists", 4, gap="CV")
def _66():
    return (PASS, "present") if exists("reports/cv_results.csv") \
        else (NOTYET, "CV not run yet")


@check(67, "CROSS-VAL", "CSV has one row PER FOLD (not just aggregates)", 4, gap="CV")
def _67():
    rr = rows("reports/cv_results.csv")
    if not rr:
        return (NOTYET, "CV not run yet")
    folds = [r for r in rr if re.match(r"^\d+$", str(
        r.get("fold", "")).strip())]
    return ok(len(folds) >= 10, f"{len(folds)} fold rows",
              f"only {len(folds)} fold rows — want 10 (2 models x 5 folds). "
              "Aggregates alone are what made A1's CV invisible")


@check(68, "CROSS-VAL", "Fold scores DIFFER (identical values mean a broken loop)", 4, gap="CV")
def _68():
    rr = rows("reports/cv_results.csv")
    if not rr:
        return (NOTYET, "CV not run yet")
    col = next((c for c in ("accuracy", "val_accuracy", "acc") if rr[0].get(c)), None)
    if not col:
        return (FAIL, "no accuracy column in cv_results.csv")
    vals = []
    for r in rr:
        try:
            vals.append(round(float(r[col]), 6))
        except Exception:
            pass
    return ok(len(set(vals)) > 1, f"{len(set(vals))} distinct values",
              "ALL FOLD SCORES IDENTICAL — the CV loop is not actually refitting")


@check(69, "CROSS-VAL", "mean/std rows recompute correctly from the fold rows", 4, gap="CV")
def _69():
    rr = rows("reports/cv_results.csv")
    if not rr:
        return (NOTYET, "CV not run yet")
    col = next((c for c in ("accuracy", "val_accuracy", "acc") if rr[0].get(c)), None)
    folds, means = [], []
    for r in rr:
        tag = str(r.get("fold", "")).strip().lower()
        try:
            v = float(r[col])
        except Exception:
            continue
        (means if tag in ("mean", "avg") else folds).append(v)
    if not means:
        return (FAIL, "no mean row in cv_results.csv")
    if not folds:
        return (FAIL, "no fold rows to recompute from")
    calc = sum(folds) / len(folds)
    close = any(abs(calc - m) < 0.02 for m in means)
    return ok(close, f"mean row agrees (recomputed {calc:.4f})",
              f"MEAN MISMATCH: recomputed {calc:.4f} vs reported {means}")


@check(70, "CROSS-VAL", "Both CV figures exist", 4, gap="CV")
def _70():
    have = [f for f in ("cv_comparison.png", "cv_fold_scores.png")
            if exists(f"reports/figures/{f}")]
    return ok(len(have) == 2, "both present", f"have {have}, want both")


@check(71, "CROSS-VAL", "README has a literal '## Cross-validation' heading", 4, gap="CV")
def _71():
    r = read("README.md")
    if not r:
        return (NOTYET, "README not written yet")
    return ok(re.search(r"^##+\s*Cross[- ]validation", r, re.M | re.I),
              "heading present",
              "NO '## Cross-validation' heading — Rule 4. This is the exact A1 failure")


@check(72, "CROSS-VAL", "README contains the fold table (not just a mean)", 4, gap="CV")
def _72():
    r = read("README.md")
    if not re.search(r"^##+\s*Cross[- ]validation", r, re.M | re.I):
        return (NOTYET, "CV section not written yet")
    sec = re.split(r"^##+\s*Cross[- ]validation", r, flags=re.M | re.I)[1]
    sec = re.split(r"^##\s", sec, flags=re.M)[0]
    return ok(sec.count("|") >= 12, "table present",
              "CV section has no table — per-fold numbers must be visible (Rule 3)")


@check(73, "CROSS-VAL", "Test split provably untouched during CV", 4, gap="CV")
def _73():
    txt = read("src/cross_validate.py")
    if not txt:
        return (NOTYET, "not written yet")
    uses_test = re.search(r"test\.csv|test_manifest", txt)
    return ok(not uses_test, "CV pools train+val only",
              "cross_validate.py references the test split — it must stay held out")


@check(74, "CROSS-VAL", "ADR-005 records selection BY CV MEAN", 4, gap="CV")
def _74():
    d = read("tracker/DECISIONS.md")
    m = re.search(r"ADR-005.*?(?=###\s*ADR-006|\Z)", d, re.S)
    if not m:
        return (FAIL, "ADR-005 not found")
    body = m.group(0)
    if "Proposed" in body and "_to be recorded" in body:
        return (NOTYET, "ADR-005 still a placeholder — fill it in on Day 4")
    return ok(re.search(r"cv|cross", body, re.I) and "Accepted" in body,
              "recorded and accepted", "ADR-005 does not justify selection by CV mean")


# ============================================================ H. API (75-82)

@check(75, "API", "api/main.py exists", 5)
def _75():
    return ok(exists("api/main.py"), "present", "missing")


@check(76, "API", "GET /health defined", 5)
def _76():
    txt = read("api/main.py")
    if not txt:
        return (NOTYET, "not written yet")
    return ok('"/health"' in txt or "'/health'" in txt, "present", "no /health route")


@check(77, "API", "POST /predict defined", 5)
def _77():
    txt = read("api/main.py")
    if not txt:
        return (NOTYET, "not written yet")
    return ok(re.search(r'post\(\s*["\']/predict', txt), "present", "no POST /predict")


@check(78, "API", "/predict returns class probabilities (spec requirement)", 5)
def _78():
    txt = read("api/main.py")
    if not txt:
        return (NOTYET, "not written yet")
    return ok("probabilit" in txt.lower(), "probabilities in response model",
              "no probabilities — spec asks for class probabilities/label")


@check(79, "API", "Model loaded once at startup (lifespan), not per request", 5)
def _79():
    txt = read("api/main.py")
    if not txt:
        return (NOTYET, "not written yet")
    return ok("lifespan" in txt, "lifespan startup load",
              "no lifespan — model may reload per request")


@check(80, "API", "Structured JSON request logging present", 6)
def _80():
    txt = read("api/main.py")
    if not txt:
        return (NOTYET, "not written yet")
    return ok("json" in txt.lower() and "middleware" in txt.lower(),
              "JSON logging middleware", "no JSON logging middleware")


@check(81, "API", "Latency tracked per request", 6)
def _81():
    txt = read("api/main.py")
    if not txt:
        return (NOTYET, "not written yet")
    return ok("latency" in txt.lower() or "perf_counter" in txt,
              "latency measured", "no latency measurement (M5 requirement)")


@check(82, "API", "/metrics exposed for Prometheus", 6)
def _82():
    txt = read("api/main.py")
    if not txt:
        return (NOTYET, "not written yet")
    return ok("Instrumentator" in txt or "/metrics" in txt, "exposed", "no /metrics")


# ============================================================ I. DOCKER (83-88)

@check(83, "DOCKER", "Dockerfile exists", 6)
def _83():
    return ok(exists("Dockerfile"), "present", "missing")


@check(84, "DOCKER", ".dockerignore excludes data/mlruns/notebooks", 6)
def _84():
    d = read(".dockerignore")
    if not d:
        return (NOTYET, "not written yet")
    missing = [n for n in ("data", "mlruns", "notebooks", "tests") if n not in d]
    return ok(not missing, "context trimmed", f"not excluded: {', '.join(missing)}")


@check(85, "DOCKER", "Container runs as a non-root user", 6)
def _85():
    txt = read("Dockerfile")
    if not txt:
        return (NOTYET, "not written yet")
    return ok(re.search(r"^USER\s+(?!root)", txt, re.M), "USER set", "runs as root")


@check(86, "DOCKER", "HEALTHCHECK defined", 6)
def _86():
    txt = read("Dockerfile")
    if not txt:
        return (NOTYET, "not written yet")
    return ok("HEALTHCHECK" in txt, "present", "no HEALTHCHECK")


@check(87, "DOCKER", "Dockerfile copies the model artifact", 6)
def _87():
    txt = read("Dockerfile")
    if not txt:
        return (NOTYET, "not written yet")
    return ok(re.search(r"COPY.*model\.(h5|keras)", txt), "model copied",
              "model.h5 not COPYed — container would start without a model")


@check(88, "DOCKER", "Image builds and serves a prediction", 6)
def _88():
    return (MANUAL, "confirm: docker build, docker run, then curl -F file=@... /predict")


# ================================================ J. TESTS & CI [GAP-CICD] (89-94)

@check(89, "TESTS", "Required test files exist (preprocess + model + api)", 7, gap="CICD")
def _89():
    missing = [f for f in ("tests/test_preprocess.py", "tests/test_model.py",
                           "tests/test_api.py") if not exists(f)]
    return ok(not missing, "all three present", f"missing: {', '.join(missing)}")


@check(90, "TESTS", "Test fixtures committed (tests run without the dataset)", 7, gap="CICD")
def _90():
    n = len(list((ROOT / "tests" / "fixtures").glob("*"))) if \
        (ROOT / "tests" / "fixtures").is_dir() else 0
    return ok(n >= 3, f"{n} fixtures", f"only {n} fixtures — CI has no data/")


@check(91, "TESTS", "pytest passes", 7, gap="CICD")
def _91():
    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists() or not exists("tests"):
        return (NOTYET, "venv or tests not ready")
    if not any((ROOT / "tests").glob("test_*.py")):
        return (NOTYET, "no tests written yet")
    r = subprocess.run([str(py), "-m", "pytest", "-q", "--no-header"],
                       cwd=ROOT, capture_output=True, text=True, timeout=900)
    tail = (r.stdout or "").strip().splitlines()
    return ok(r.returncode == 0, f"{tail[-1] if tail else 'passed'}",
              f"FAILING: {tail[-1] if tail else r.returncode}")


@check(92, "TESTS", "ruff clean", 7, gap="CICD")
def _92():
    ruff = ROOT / ".venv" / "bin" / "ruff"
    if not ruff.exists():
        return (NOTYET, "ruff not installed yet")
    r = subprocess.run([str(ruff), "check", "."], cwd=ROOT,
                       capture_output=True, text=True, timeout=180)
    return ok(r.returncode == 0, "clean",
              f"{len((r.stdout or '').splitlines())} issue lines")


@check(93, "CI", "ci.yml has gated jobs (build needs test)", 7, gap="CICD")
def _93():
    y = read(".github/workflows/ci.yml")
    if not y:
        return (NOTYET, "not written yet")
    jobs = re.findall(r"^  ([a-z][\w-]*):", y, re.M)
    return ok("needs:" in y and len(jobs) >= 3,
              f"{len(jobs)} jobs, gating present",
              f"jobs={jobs}, needs:={'needs:' in y} — 'thin CI/CD' risk")


@check(94, "CI", "ci.yml pushes to GHCR with a SHA tag", 7, gap="CICD")
def _94():
    y = read(".github/workflows/ci.yml") + read(".github/workflows/cd.yml")
    if not y:
        return (NOTYET, "not written yet")
    has_ghcr = "ghcr.io" in y
    has_sha = re.search(r"(sha|github\.sha)", y)
    return ok(has_ghcr and bool(has_sha), "GHCR + SHA tagging",
              f"ghcr={has_ghcr} sha_tag={bool(has_sha)} — M3 artifact publishing")


# ================================================ K. CD & DEPLOY [GAP-CICD] (95-97)

@check(95, "CD", "k8s Deployment + Service manifests exist", 8, gap="CICD")
def _95():
    missing = [f for f in ("k8s/deployment.yaml", "k8s/service.yaml") if not exists(f)]
    return ok(not missing, "both present", f"missing: {', '.join(missing)}")


@check(96, "CD", "Deployment references a SHA tag, not :latest (traceability)", 8, gap="CICD")
def _96():
    y = read("k8s/deployment.yaml")
    if not y:
        return (NOTYET, "not written yet")
    m = re.search(r"image:\s*(\S+)", y)
    if not m:
        return (FAIL, "no image: line")
    img = m.group(1)
    return ok(not img.endswith(":latest"), f"image={img}",
              f"image={img} — :latest breaks commit->pod traceability (Rule 7)")


@check(97, "CD", "Argo CD Application + post-deploy smoke test exist", 8, gap="CICD")
def _97():
    have_argo = exists("argocd/application.yaml")
    have_smoke = exists("scripts/smoke_test_deployed.py")
    return ok(have_argo and have_smoke,
              "both present",
              f"argocd={have_argo} smoke_test={have_smoke} — M4 core")


# ================================================ L. MONITORING & SUBMISSION (98-100)

@check(98, "MONITORING", "Grafana dashboard + monitoring README exported", 9)
def _98():
    missing = [f for f in ("monitoring/grafana_dashboard.json", "monitoring/README.md")
               if not exists(f)]
    return ok(not missing, "both present", f"missing: {', '.join(missing)}")


@check(99, "MONITORING", "Post-deployment tracking artifacts exist", 10)
def _99():
    have = [f for f in ("scripts/replay_batch.py", "reports/post_deployment_report.md",
                        "reports/figures/post_deploy_confusion_matrix.png")
            if exists(f)]
    return ok(len(have) >= 3, "all present", f"{len(have)}/3: have {have}")


@check(100, "SUBMISSION", "README maps every module to real file paths (Rule 1)", 10)
def _100():
    r = read("README.md")
    if not r:
        return (NOTYET, "README not written yet")
    has_map = re.search(r"maps?\s+to\s+the\s+rubric", r, re.I)
    mods = sum(1 for m in ("M1", "M2", "M3", "M4", "M5") if m in r)
    return ok(bool(has_map) and mods >= 5,
              f"rubric map present, {mods}/5 modules referenced",
              f"rubric_map={bool(has_map)} modules={mods}/5 — the 60-Second Grader Test")


# ---------------------------------------------------------------- runner

def main() -> int:
    ap = argparse.ArgumentParser(description="100-point daily audit")
    ap.add_argument("--day", type=int, default=10,
                    help="current plan day (1-10); gates which checks are active")
    ap.add_argument("--only", help="filter to one group (e.g. CONSISTENCY, CROSS-VAL)")
    ap.add_argument("--verbose", action="store_true", help="show NOT-YET checks too")
    a = ap.parse_args()

    for num, group, title, day, gap, fn in sorted(_REGISTRY):
        if a.only and a.only.upper() not in group.upper():
            continue
        if day > a.day:
            RESULTS.append(Result(num, group, title, NOTYET,
                                  f"activates on day {day}", gap))
            continue
        try:
            status, detail = fn()
        except Exception as e:
            status, detail = FAIL, f"check errored: {type(e).__name__}: {e}"
        RESULTS.append(Result(num, group, title, status, detail, gap))

    print(f"\n{C['bold']}100-Point Daily Audit{C['reset']} "
          f"{C['dim']}— day {a.day}/10 · cats-dogs-mlops{C['reset']}\n")

    last = None
    for r in RESULTS:
        if r.status == NOTYET and not a.verbose:
            continue
        if r.group != last:
            print(f"{C['bold']}{r.group}{C['reset']}")
            last = r.group
        tag = f" {C['bold']}[GAP-{r.gap}]{C['reset']}" if r.gap else ""
        print(f"  {C[r.status]}{r.status:<7}{C['reset']} "
              f"{r.num:>3}. {r.title}{tag}")
        if r.detail and r.status != PASS:
            print(f"          {C['dim']}{r.detail}{C['reset']}")
        elif r.detail and a.verbose:
            print(f"          {C['dim']}{r.detail}{C['reset']}")

    tally = {s: sum(1 for r in RESULTS if r.status == s)
             for s in (PASS, FAIL, MANUAL, NOTYET)}
    active = tally[PASS] + tally[FAIL] + tally[MANUAL]
    fails = [r for r in RESULTS if r.status == FAIL]
    gapfails = [r for r in fails if r.gap]

    print(f"\n{C['bold']}Summary{C['reset']}  "
          f"{C[PASS]}{tally[PASS]} pass{C['reset']} · "
          f"{C[FAIL]}{tally[FAIL]} fail{C['reset']} · "
          f"{C[MANUAL]}{tally[MANUAL]} manual{C['reset']} · "
          f"{C[NOTYET]}{tally[NOTYET]} not-yet{C['reset']}   "
          f"{C['dim']}({active} active of 100){C['reset']}")

    if gapfails:
        print(f"\n{C[FAIL]}{C['bold']}⚠ {len(gapfails)} failure(s) on the "
              f"Assignment-1 mark-loss axes:{C['reset']}")
        for r in gapfails:
            print(f"    {r.num}. [{r.gap}] {r.title}")

    if tally[MANUAL]:
        print(f"\n{C[MANUAL]}Manual confirmation needed:{C['reset']}")
        for r in RESULTS:
            if r.status == MANUAL:
                print(f"    {r.num}. {r.title} — {C['dim']}{r.detail}{C['reset']}")

    if fails:
        print(f"\n{C[FAIL]}FAILED — fix the {len(fails)} item(s) above "
              f"before closing day {a.day}.{C['reset']}\n")
        return 1
    print(f"\n{C[PASS]}Day {a.day} audit clean.{C['reset']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
