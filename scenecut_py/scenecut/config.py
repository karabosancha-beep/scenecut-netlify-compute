"""Default configuration & paths."""
from __future__ import annotations

import os
from pathlib import Path

# Project root is the parent of scenecut_py/
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent  # /home/z/my-project

# Workspace directory for uploads, projects, downloads.
# Override via SCENECUT_WORKDIR env var.
WORKDIR = Path(os.environ.get("SCENECUT_WORKDIR", PROJECT_ROOT / "upload"))
WORKDIR.mkdir(parents=True, exist_ok=True)

# Projects subdir (each project = one folder with project.json + clips + thumbs)
PROJECTS_DIR = WORKDIR / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# Downloads (yt-dlp output lands here by default)
DOWNLOADS_DIR = WORKDIR / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Default detection parameters (schema 1.1 — JS-divergence scale for
# dissolve/group thresholds; see .agents/DECISIONS.md D3/D11)
DEFAULTS = {
    "algo": "all",
    "threshold": 27.0,
    "min_scene_len": 0.6,
    "fade_threshold": 12,
    "fade_ceiling": 243,
    "fade_min_len": 0.3,
    "dissolve_windows": [8, 16, 32],
    "dissolve_dissim": 0.15,     # normalized JS endpoint floor (D19: SBD T p5
                                  # 0.177 / negatives p75 0.108 — the notch)
    "hard_cut_mad_min": 40.0,    # pixel MAD above which a window contains a hard
                                  # cut (D19: cuts p25 53.8 / whip 56+ vs external
                                  # blends p75 20 / p90 ~30; was 20 = fixture-era)
    "consec_js_min": 0.03,       # window-mean consec-JS floor (D19: external
                                  # blends p25 0.030 / p50 0.046; sunset-drift
                                  # impostors measure 0.026-0.028 — the floor
                                  # sits exactly between them; was 0.15 =
                                  # fixture-era, rejected ~95% of real blends)
    "consec_js_frame_min": 0.03,   # D19: frame-level elevation floor (sustained-
                                    # elevation gate; single-cut spikes fail it)
    "consec_js_sustain_frac": 0.35,  # D19: fraction of window frames >= frame_min
    "consec_js_ratio_min": 1.8,     # D19: window mean must exceed clip-baseline
                                     # median by this factor (global motion like
                                     # pans has center ~= base -> rejected; blends
                                     # are localized elevation -> pass)
    "hash_threshold": 0.4,       # HashDetector firing distance (was hardcoded)
    "hash_corroboration": True,  # D16 v4.3: solo-hash feature-evidence gate
    "hash_corroboration_spike": 3.0,   # mad[p]/max(median(mad[p-30..p-5]), 0.5)
    "hash_corroboration_js": 0.15,     # D16-pinned consec_js[p-1] floor for solo-hash
                                       # corroboration (DECOUPLED from consec_js_min
                                       # by D19 — do NOT retune it with the dissolve
                                       # gates; 0.15 is the measured 56:1 value)
    "hash_corroboration_luma": 8.0,    # |mean_v[p]-mean_v[p-1]| floor
    "spatial_dissolve": True,     # D18 (Phase 1.5): tier-2 match-dissolve detector
    "spatial_dissim": 14.0,       # 8x8 block-max MAD endpoint floor (MD 17-84 vs sunset <=10)
    "spatial_consec_lo": 1.0,     # per-frame blend-band floor (MD med 3.86; sunset 0.18)
    "spatial_motion_max": 8.0,    # band ceiling (pan 36 / fades 18.8-19.9 sit ABOVE)
    "motion_gate": False,
    "motion_gate_js": 0.05,      # consecutive JS below this => motion, not content
    "group_scenes": False,
    "group_threshold": 0.15,     # normalized JS scale (was Pearson 0.6)
    "thumb_size": 320,
    "proxy_height": 360,
    "proxy_crf": 28,
    "proxy_preset": "veryfast",
    "cut_crf": 20,
    "cut_preset": "veryfast",
    "format": "mp4",
}

SCENE_FILE_VERSION = "1.0"


def project_dir(project_id: str) -> Path:
    """Return (and create) the directory for a project."""
    p = PROJECTS_DIR / project_id
    p.mkdir(parents=True, exist_ok=True)
    (p / "clips").mkdir(exist_ok=True)
    (p / "thumbs").mkdir(exist_ok=True)
    return p


def project_file(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"
