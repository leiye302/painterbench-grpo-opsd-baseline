"""Native PainterBench state transitions; only transport/storage are adapted."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import re


def split_stimuli():
    from painterbench.stimuli import STIMULI
    train, heldout = [], []
    for kind, n in [("incomplete_shape", 4), ("object_transformation", 2)]:
        names = sorted((k for k, v in STIMULI.items() if v["kind"] == kind),
                       key=lambda k: hashlib.sha256(("painterbench-route-v1:" + k).encode()).hexdigest())
        heldout.extend(names[:n])
        train.extend(names[n:])
    return sorted(train), sorted(heldout)


def processor_messages(messages):
    """Convert transport image objects without changing text or image pixels."""
    from PIL import Image
    messages = copy.deepcopy(messages)
    for message in messages:
        if isinstance(message.get("content"), list):
            for i, item in enumerate(message["content"]):
                if item["type"] == "image_url":
                    url = item["image_url"]["url"]
                    if not url.startswith("data:image/png;base64,"):
                        raise ValueError("Native environment images must be local PNG data")
                    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
                    message["content"][i] = {"type": "image", "image": image}
    return messages


def parse_qwen_tools(text, schemas):
    """Qwen3.5 native qwen3_coder function/parameter wire format.

    Parse only complete calls AFTER the reasoning close. No repair, retries,
    regex extraction of arbitrary JSON, constrained decoding, or Python eval.
    """
    if "</think>" not in text:
        return [], "", "reasoning_not_closed"
    reasoning, body = text.split("</think>", 1)
    properties = {s["function"]["name"]: s["function"]["parameters"].get("properties", {})
                  for s in schemas}
    calls = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", body, re.S):
        match = re.fullmatch(r"\s*<function=([A-Za-z_]\w*)>(.*?)</function>\s*", block, re.S)
        if not match:
            return [], reasoning, "malformed_tool_call"
        name, content = match.groups()
        args, end = {}, 0
        for param in re.finditer(r"<parameter=([A-Za-z_]\w*)>(.*?)</parameter>", content, re.S):
            if content[end:param.start()].strip():
                return [], reasoning, "malformed_parameter"
            key, value = param.groups()
            if key in args:
                return [], reasoning, "duplicate_parameter"
            kind = properties.get(name, {}).get(key, {}).get("type", "string")
            try:
                args[key] = value.strip() if kind == "string" else json.loads(value)
            except (ValueError, TypeError):
                return [], reasoning, "invalid_parameter_json"
            end = param.end()
        if content[end:].strip():
            return [], reasoning, "malformed_parameter"
        calls.append({"id": f"call_{len(calls)}", "type": "function",
                      "function": {"name": name, "arguments": args}})
    if body.count("<tool_call>") != len(calls) or body.count("</tool_call>") != len(calls):
        return [], reasoning, "incomplete_tool_call"
    return calls, reasoning, "" if calls else "no_tool_call"


class DrawingEpisode:
    def __init__(self, directory, stimulus, max_tokens=16384):
        from painterbench.canvas import Canvas
        from painterbench.config import ModelSpec, RunSettings, TaskSpec
        from painterbench.agents.drawing import build_session_inputs
        from painterbench.core.loop import build_initial_user_message
        from painterbench.core.tool_schema import build_tool_specs
        from painterbench.stimuli import STIMULI
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stimulus = stimulus
        self.canvas = Canvas(self.directory / "current.png", stimulus=stimulus, stroke_width=5)
        self.canvas.reset()
        self.canvas.load().save(self.directory / "initial.png")
        self.snapshots = self.directory / "snapshots"
        self.snapshots.mkdir()
        run = RunSettings(max_rounds=0, max_tokens=max_tokens, tool_history_window=10,
                          stitched_snapshot_window=3, stroke_width=5)
        inputs = build_session_inputs(
            model=ModelSpec(name="actor", model_id="Qwen/Qwen3.5-4B"),
            task=TaskSpec(name=stimulus, stimulus=stimulus,
                          prompt="Create the most original drawing you can think of.",
                          category=STIMULI[stimulus]["kind"]),
            run=run, canvas=self.canvas, snapshot_dir=self.snapshots,
            final_painting_path=self.directory / "final_painting.png")
        self.tools = build_tool_specs(inputs.tools)
        self.dispatch = {f.__name__: f for f in inputs.tools}
        self.preamble = [{"role": "system", "content": inputs.system_prompt},
                         build_initial_user_message(self.canvas, inputs.initial_user_text, include_canvas=True)]
        self.history = []
        self.records = []
        self.finished = False

    def messages(self):
        from painterbench.core.loop import build_turn_context_message
        kept = self.history[-10:]
        messages = copy.deepcopy(self.preamble)
        for group in kept:
            messages.extend(group)
        context = build_turn_context_message(self.canvas, self.snapshots, 3,
                                              include_canvas=True, include_stitched=True,
                                              omitted_rounds=len(self.history)-len(kept))
        if context is not None:
            messages.append(context)
        return messages

    def step(self, text):
        calls, reasoning, parse_error = parse_qwen_tools(text, self.tools)
        before = hashlib.sha256(self.canvas.load().tobytes()).hexdigest()
        # Match official dispatch: complete non-finish calls before finish.
        order = sorted(range(len(calls)), key=lambda i: calls[i]["function"]["name"] == "drawing_finished")
        results = {}
        errors = 0
        for i in order:
            fn = calls[i]["function"]
            try:
                if fn["name"] not in self.dispatch:
                    raise ValueError("unknown tool")
                value = self.dispatch[fn["name"]](**fn["arguments"])
                content = "" if value is None else str(value)
                if fn["name"] == "drawing_finished":
                    self.finished = True
            except Exception as exc:
                content = f"ERROR: {type(exc).__name__}: {exc}"
                errors += 1
            results[i] = {"role": "tool", "tool_call_id": calls[i]["id"],
                          "name": fn["name"], "content": content}
        assistant = {"role": "assistant", "content": ""}
        if calls:
            assistant["tool_calls"] = calls
        self.history.append([assistant, *[results[i] for i in range(len(calls))]])
        record = {"raw_text": text, "reasoning_present": bool(reasoning.strip()),
                  "parse_error": parse_error, "tool_calls": calls,
                  "tool_results": [results[i] for i in range(len(calls))],
                  "tool_error_count": errors, "finished": self.finished,
                  "canvas_before_sha256": before,
                  "canvas_after_sha256": hashlib.sha256(self.canvas.load().tobytes()).hexdigest()}
        self.records.append(record)
        return self.finished or not calls

    def finish(self, stop_reason):
        from scripts.reward_probe import atomic_json
        final = self.directory / "final_painting.png"
        # Always score the actual final state, even no-tool/length/turn caps.
        self.canvas.load().save(final)
        atomic_json(self.directory / "trajectory.json", {
            "stimulus": self.stimulus, "finished": self.finished,
            "stop_reason": stop_reason, "turns": self.records})
        return final
