"""Interactive audit REPL — add/remove/move cuts, merge/split scenes."""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from .project import apply_overrides, load_project, save_project, add_override
from .timecode import hmsms_to_seconds, seconds_to_hmsms


HELP = """Audit commands:
  list                       list current cuts
  add <time>                 add a cut at time (e.g. 12.5 or 00:00:12.500)
  del <index>                remove cut at index (1-based from `list`)
  move <index> <time>        move a cut
  merge <i> <j>              merge scenes i and j (1-based scene indices)
  split <scene> <time>       split scene at time (1-based scene index, time relative)
  label <scene> <text>       set scene label
  tags <scene> <tag,tag>     set scene tags
  save [path]                save project (default: in-place)
  show                       show full JSON of current project
  help                       this message
  quit / exit                exit without saving (use 'save' first)
"""


def _parse_time(s: str) -> float:
    s = s.strip()
    if ":" in s:
        return hmsms_to_seconds(s)
    return float(s)


def audit(project_path: str, input_video: str | None = None) -> int:
    """Run the interactive audit REPL. Returns 0 on clean exit."""
    project = load_project(project_path)
    print(f"Loaded project: {project.get('source', {}).get('path', '?')}")
    print(f"Cuts: {len(project.get('cuts', []))} | Scenes: {len(project.get('scenes', []))}")
    print(HELP)

    while True:
        try:
            line = input("scenecut> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        try:
            parts = shlex.split(line)
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd in ("quit", "exit"):
                print("Exiting without save (use 'save' to persist).")
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "list":
                p = apply_overrides(project)
                for c in p["cuts"]:
                    print(f"  [{c['index']}] {c['timecode']} ({c['seconds']:.3f}s) type={c['type']} conf={c.get('confidence', 0):.2f}")
            elif cmd == "show":
                p = apply_overrides(project)
                print(json.dumps(p, indent=2)[:4000])
            elif cmd == "add":
                if not args:
                    print("usage: add <time>")
                    continue
                t = _parse_time(args[0])
                add_override(project, "added_cuts", {"seconds": t, "type": "cut"})
                print(f"Added cut at {seconds_to_hmsms(t)} (will be applied on next save/list)")
            elif cmd == "del":
                if not args:
                    print("usage: del <cut-index>")
                    continue
                idx = int(args[0])
                add_override(project, "removed_cuts", idx)
                print(f"Marked cut {idx} for removal")
            elif cmd == "move":
                if len(args) < 2:
                    print("usage: move <index> <time>")
                    continue
                idx = int(args[0])
                t = _parse_time(args[1])
                add_override(project, "moved_cuts", (idx, t))
                print(f"Marked cut {idx} moved to {seconds_to_hmsms(t)}")
            elif cmd == "merge":
                if len(args) < 2:
                    print("usage: merge <i> <j>")
                    continue
                i, j = int(args[0]), int(args[1])
                add_override(project, "merged_scenes", [i, j])
                print(f"Marked scenes {i} and {j} for merge (applies on save)")
            elif cmd == "split":
                if len(args) < 2:
                    print("usage: split <scene> <time>")
                    continue
                s, t = int(args[0]), _parse_time(args[1])
                add_override(project, "split_scenes", {"scene": s, "at": t})
                print(f"Marked scene {s} for split at {seconds_to_hmsms(t)}")
            elif cmd == "label":
                if len(args) < 2:
                    print("usage: label <scene> <text>")
                    continue
                s = int(args[0])
                project.setdefault("labels", {})[str(s)] = " ".join(args[1:])
                print(f"Set label for scene {s}")
            elif cmd == "tags":
                if len(args) < 2:
                    print("usage: tags <scene> <tag,tag>")
                    continue
                s = int(args[0])
                project.setdefault("tags", {})[str(s)] = args[1].split(",")
                print(f"Set tags for scene {s}")
            elif cmd == "save":
                out_path = args[0] if args else project_path
                save_project(project, out_path)
                print(f"Saved to {out_path}")
            else:
                print(f"Unknown command: {cmd}. Type 'help'.")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
    return 0
