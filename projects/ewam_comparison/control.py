#!/usr/bin/env python3
"""Submit a reviewable command to the current-window planner's running episode."""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import argparse
import json
import os
from pathlib import Path
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--note", default="", help="Brief observable progress/subgoal summary for the trace")
    sub = parser.add_subparsers(dest="op", required=True)
    sub.add_parser("observe")
    sub.add_parser("finish")
    ew = sub.add_parser("ewam")
    ew.add_argument("--instruction")
    ew.add_argument("--chunks", type=int, default=1)
    ew.add_argument("--model-actions", type=int, default=16)
    code = sub.add_parser("python")
    code.add_argument("--file", type=Path, required=True)
    args = parser.parse_args()
    identifier = str(uuid.uuid4())
    inbox = args.session / "commands"
    inbox.mkdir(exist_ok=True)
    command = {"id": identifier, "op": args.op, "issued_at": time.time(), "note": args.note}
    if args.op == "ewam":
        command.update(instruction=args.instruction, chunks=args.chunks,
                       model_actions=args.model_actions)
    if args.op == "python":
        command["code"] = args.file.read_text()
    temporary = inbox / (identifier + ".tmp")
    temporary.write_text(json.dumps(command, ensure_ascii=False) + "\n")
    os.replace(temporary, inbox / (identifier + ".json"))
    response = args.session / "responses" / (identifier + ".json")
    deadline = time.monotonic() + 1200
    while not response.exists():
        if (args.session / "result.json").exists():
            print((args.session / "result.json").read_text())
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Command pending: {identifier}")
        time.sleep(0.25)
    print(response.read_text())


if __name__ == "__main__":
    main()
