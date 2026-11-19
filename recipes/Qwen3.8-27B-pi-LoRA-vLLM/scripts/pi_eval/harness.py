#!/usr/bin/env python3
"""A pi-shaped agent loop for A/B testing the pi LoRA against its base.

Replays pi's coding-agent system prompt and its four default tools (read, bash,
edit, write) with the descriptions and JSON schemas from pi's source
(earendil-works/pi main @ 5a14e69, packages/coding-agent/src/core), over the
OpenAI chat-completions API (pi's `openai-completions` mode). bash runs in a
throwaway container with no network, as the invoking user; read/edit/write are
confined to the task directory. Hidden checks are copied in only after the
agent stops.

Usage:
  ./harness.py                         # both suites, 5 reps, base vs adapter, port 8014
  ./harness.py --suite hard --reps 3
  ./analyze.py <run dir>/results.jsonl

Results and per-run transcripts go to OUT_DIR/pi-eval/<timestamp>/ (default
~/.local/state/dgx-spark-recipes/Qwen3.8-27B-pi-LoRA-vLLM/pi-eval/), never into
the recipe directory.
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid

sys.dont_write_bytecode = True  # keep __pycache__ out of the recipe directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import TASKS  # noqa: E402

IMAGE = "vllm/vllm-openai@sha256:3016f449f5bf9bbe171406b7715822acb49026a4a404ba830f482575c02302d6"
WORK = "/work"
STATE_DIR = os.environ.get("OUT_DIR") or os.path.join(
    os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state"),
    "dgx-spark-recipes", "Qwen3.8-27B-pi-LoRA-vLLM")
SHIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shim")
MAX_LINES, MAX_BYTES = 2000, 50 * 1024

# --- pi's system prompt (buildSystemPromptSections with the default tools) ---
SYSTEM_PROMPT = "\n\n".join([
    "You are a coding assistant running inside pi, a coding agent harness. Help users inspect files, run commands, and modify code.",
    "<tools>\n- read: Read file contents\n- bash: Execute bash commands (ls, grep, find, etc.)\n"
    "- edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call\n"
    "- write: Create or overwrite files\n\nAdditional tools may be available.\n</tools>",
    "<rules>\n- Use bash for file operations like ls, rg, find\n- Use read to examine files instead of cat or sed.\n"
    "- Use edit for targeted changes. Batch changes to one file in a single call.\n"
    "- Each edits[].oldText must exactly and uniquely match the original file. Keep matches short and non-overlapping; merge nearby changes without including large unchanged regions.\n"
    "- Use write only for new files or complete rewrites.\n- Be concise in your responses\n- Show file paths clearly when working with files\n</rules>",
    "<docs>\nPi documentation (read only when the user asks about the harness itself, its SDK, extensions, themes, skills, or TUI):\n"
    "- Main documentation: /usr/lib/node_modules/@earendil-works/pi/README.md\n- Additional docs: /usr/lib/node_modules/@earendil-works/pi/docs\n"
    "- Examples: /usr/lib/node_modules/@earendil-works/pi/examples (extensions, custom tools, SDK)\n</docs>",
    f"<cwd>\n{WORK}\n</cwd>",
])

TOOLS = [
    {"type": "function", "function": {
        "name": "read",
        "description": f"Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to {MAX_LINES} lines or {MAX_BYTES // 1024}KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
        "parameters": {"type": "object", "required": ["path"], "properties": {
            "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
            "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
            "limit": {"type": "number", "description": "Maximum number of lines to read"}}}}},
    {"type": "function", "function": {
        "name": "bash",
        "description": f"Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last {MAX_LINES} lines or {MAX_BYTES // 1024}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
        "parameters": {"type": "object", "required": ["command"], "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"}}}}},
    {"type": "function", "function": {
        "name": "edit",
        "description": "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.",
        "parameters": {"type": "object", "required": ["path", "edits"], "properties": {
            "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
            "edits": {"type": "array", "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.",
                      "items": {"type": "object", "required": ["oldText", "newText"], "properties": {
                          "oldText": {"type": "string", "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."},
                          "newText": {"type": "string", "description": "Replacement text for this targeted edit."}}}}}}}},
    {"type": "function", "function": {
        "name": "write",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
        "parameters": {"type": "object", "required": ["path", "content"], "properties": {
            "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
            "content": {"type": "string", "description": "Content to write to the file"}}}}},
]


class ToolError(Exception):
    pass


class Sandbox:
    def __init__(self, root):
        self.root = os.path.realpath(root)
        self.name = f"pi-eval-{uuid.uuid4().hex[:10]}"
        subprocess.run(["docker", "run", "-d", "--rm", "--name", self.name, "--network", "none",
                        "--memory", "2g", "--cpus", "2", "--pids-limit", "256",
                        "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
                        "-v", f"{self.root}:{WORK}", "-w", WORK,
                        # The image has python3 but no `python`; the shim supplies it.
                        "-v", f"{SHIM}:/opt/shim:ro",
                        "-e", "PATH=/opt/shim:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                        "--entrypoint", "sleep", IMAGE, "infinity"],
                       check=True, capture_output=True)

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)

    def host_path(self, path):
        p = path if path.startswith("/") else os.path.join(WORK, path)
        p = os.path.normpath(p)
        if not (p == WORK or p.startswith(WORK + "/")):
            raise ToolError(f"Path is outside the working directory: {path}")
        real = os.path.realpath(os.path.join(self.root, os.path.relpath(p, WORK)))
        if not (real == self.root or real.startswith(self.root + "/")):
            raise ToolError(f"Path is outside the working directory: {path}")
        return real

    def bash(self, command, timeout=None):
        t = min(float(timeout or 120), 300)
        r = subprocess.run(["docker", "exec", "-w", WORK, self.name, "timeout", "-k", "5", str(int(t)),
                            "bash", "-c", command], capture_output=True)
        out = (r.stdout + r.stderr).decode(errors="replace")
        lines = out.splitlines()
        if len(lines) > MAX_LINES or len(out) > MAX_BYTES:
            out = "\n".join(lines[-MAX_LINES:])[-MAX_BYTES:]
        if r.returncode == 124:
            out += f"\n\nCommand timed out after {int(t)} seconds"
        if r.returncode != 0:
            out += f"\n\nCommand exited with code {r.returncode}"
            return out.strip() or "(no output)", True
        return out.strip() or "(no output)", False


def run_tool(sb, name, args):
    """Returns (text, is_error)."""
    try:
        if name == "bash":
            return sb.bash(args["command"], args.get("timeout"))
        if name == "read":
            p = sb.host_path(args["path"])
            if not os.path.isfile(p):
                raise ToolError(f"File not found: {args['path']}")
            lines = open(p, errors="replace").read().split("\n")
            start = max(int(args.get("offset") or 1), 1) - 1
            if start >= len(lines) and lines != [""]:
                raise ToolError(f"Offset {start + 1} is beyond end of file ({len(lines)} lines total)")
            limit = int(args["limit"]) if args.get("limit") else None
            chunk = lines[start:start + limit] if limit else lines[start:]
            text = "\n".join(chunk[:MAX_LINES])[:MAX_BYTES]
            end = start + min(len(chunk), MAX_LINES)
            if end < len(lines) and (limit is None or len(chunk) > MAX_LINES):
                text += f"\n\n[Showing lines {start + 1}-{end} of {len(lines)}. Use offset={end + 1} to continue.]"
            elif limit and start + limit < len(lines):
                text += f"\n\n[{len(lines) - start - limit} more lines in file. Use offset={start + limit + 1} to continue.]"
            return text, False
        if name == "write":
            p = sb.host_path(args["path"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(args["content"])
            return f"Successfully wrote to {args['path']}", False
        if name == "edit":
            p = sb.host_path(args["path"])
            if not os.path.isfile(p):
                raise ToolError(f"File not found: {args['path']}")
            original = open(p).read()
            edits = args["edits"]
            if isinstance(edits, str):
                edits = json.loads(edits)
            spans = []
            for e in edits:
                n = original.count(e["oldText"])
                if n == 0:
                    raise ToolError(f"Could not find the exact text in {args['path']}. The old text must match exactly including all whitespace and newlines.")
                if n > 1:
                    raise ToolError(f"Found {n} occurrences of the text in {args['path']}. The text must be unique. Please provide more context to make it unique.")
                i = original.index(e["oldText"])
                spans.append((i, i + len(e["oldText"]), e["newText"]))
            spans.sort()
            for a, b in zip(spans, spans[1:]):
                if b[0] < a[1]:
                    raise ToolError("Edits overlap. Merge them into one edit.")
            out, pos = [], 0
            for s, t, new in spans:
                out += [original[pos:s], new]
                pos = t
            out.append(original[pos:])
            with open(p, "w") as f:
                f.write("".join(out))
            return f"Successfully replaced {len(spans)} block(s) in {args['path']}.", False
        raise ToolError(f"Tool {name} not found")
    except ToolError as e:
        return str(e), True
    except (KeyError, TypeError, ValueError) as e:
        return f"Invalid arguments for {name}: {e}", True


def chat(base, model, messages, temperature, timeout=900):
    payload = {"model": model, "messages": messages, "tools": TOOLS, "max_tokens": 4096,
               "temperature": temperature, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run_one(base, model, task, rep, temperature, max_turns, keep_dir):
    root = tempfile.mkdtemp(prefix=f"{task['name']}-", dir=keep_dir)
    for rel, content in task["files"].items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb" if isinstance(content, bytes) else "w").write(content)
    sb = Sandbox(root)
    rec = dict(model=model, task=task["name"], rep=rep, tool_calls=0, bad_calls=0, tool_errors=0,
               turns=0, completion_tokens=0, prompt_tokens=0, stop="turn_limit", raw_tool_text=0, tools={})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task["prompt"]}]
    t0 = time.monotonic()
    try:
        for _ in range(max_turns):
            body = chat(base, model, messages, temperature)
            rec["turns"] += 1
            rec["completion_tokens"] += body["usage"]["completion_tokens"]
            rec["prompt_tokens"] = body["usage"]["prompt_tokens"]
            msg = body["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            content = msg.get("content") or ""
            if "<tool_call>" in content or "<function=" in content:
                rec["raw_tool_text"] += 1
            messages.append({"role": "assistant", "content": content or None,
                             **({"tool_calls": calls} if calls else {})})
            if not calls:
                rec["stop"] = "finished" if body["choices"][0]["finish_reason"] == "stop" else body["choices"][0]["finish_reason"]
                rec["final"] = content[-400:]
                break
            for c in calls:
                rec["tool_calls"] += 1
                name = c["function"]["name"]
                rec["tools"][name] = rec["tools"].get(name, 0) + 1
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                    if not isinstance(args, dict):
                        raise ValueError
                except ValueError:
                    rec["bad_calls"] += 1
                    text, err = f"Invalid JSON arguments for {name}", True
                else:
                    text, err = run_tool(sb, name, args)
                    if err and (text.startswith("Invalid arguments") or text.startswith("Tool ")):
                        rec["bad_calls"] += 1
                rec["tool_errors"] += int(err)
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": text})
    except Exception as e:  # noqa: BLE001
        rec["stop"] = f"error: {type(e).__name__}: {e}"[:200]
    rec["seconds"] = round(time.monotonic() - t0, 1)
    for rel, content in task["hidden"].items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb" if isinstance(content, bytes) else "w").write(content)
    out, err = sb.bash(task["verify"], 120)
    rec["solved"] = not err
    rec["verify_tail"] = out[-300:]
    sb.close()
    with open(os.path.join(root, ".transcript.json"), "w") as f:
        json.dump(messages, f, indent=1)
    rec["dir"] = root
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=f"http://127.0.0.1:{os.environ.get('PORT', '8014')}")
    ap.add_argument("--models", default="qwen3.8-27b,qwen3.8-pi")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--suite", default="", help="easy, hard, or empty for all")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--run-dir", default=None,
                    help="where results.jsonl and per-run directories go (default: OUT_DIR/pi-eval/<timestamp>)")
    a = ap.parse_args()
    a.run_dir = a.run_dir or os.path.join(STATE_DIR, "pi-eval", time.strftime("%Y%m%dT%H%M%S"))
    a.keep_dir = os.path.join(a.run_dir, "runs")
    a.out = os.path.join(a.run_dir, "results.jsonl")
    os.makedirs(a.keep_dir, exist_ok=True)
    tasks = [t for t in TASKS if (not a.tasks or t["name"] in a.tasks.split(","))
             and (not a.suite or t["suite"] == a.suite)]
    jobs = [(m, t, r) for r in range(a.reps) for t in tasks for m in a.models.split(",")]
    print(f"{len(jobs)} runs, {a.parallel} at a time -> {a.out}", flush=True)
    with cf.ThreadPoolExecutor(a.parallel) as ex, open(a.out, "a") as out:
        futs = {ex.submit(run_one, a.base, m, t, r, a.temperature, a.max_turns, a.keep_dir): (m, t["name"], r) for m, t, r in jobs}
        for f in cf.as_completed(futs):
            rec = f.result()
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(f"{rec['model']:12s} {rec['task']:20s} r{rec['rep']} solved={rec['solved']!s:5s} calls={rec['tool_calls']:2d} "
                  f"bad={rec['bad_calls']} turns={rec['turns']:2d} tok={rec['completion_tokens']:5d} {rec['seconds']:6.1f}s {rec['stop']}", flush=True)


if __name__ == "__main__":
    main()
