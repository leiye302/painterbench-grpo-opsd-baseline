import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from painter_rl.actor import (
    CaptureSampleLogProbs, extend_inputs, score_tokens, native_eos_ids, save_turn_audit,
)
from painter_rl.checkpoints import save_checkpoint, verify_checkpoint, restore_optimizer, prune_checkpoints
from painter_rl.environment import parse_qwen_tools, processor_messages
from painter_rl.training import validate_config


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 5)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.head = torch.nn.Linear(5, 11)

    def get_output_embeddings(self):
        return self.head

    def save_pretrained(self, path, **kwargs):
        path.mkdir()
        torch.save(self.state_dict(), path / "toy.pt")


class TinyProcessor:
    def save_pretrained(self, path):
        (path / "processor.json").write_text("{}")


class RuntimeContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_chunked_head_exact_causal_alignment_and_gradient(self):
        model = TinyModel()
        inputs = {"input_ids": torch.tensor([[1, 2, 3]]),
                  "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        response = torch.tensor([4, 5, 6])
        values, prefix = extend_inputs(inputs, response, torch.device("cpu"))
        logits = model.head(model.model(**values).last_hidden_state)[:, prefix-1:prefix+2]
        probs = logits.log_softmax(-1)
        expected = probs.gather(-1, response.reshape(1, -1, 1)).squeeze(-1)
        expected_entropy = -(probs.exp() * probs).sum(-1)
        actual, entropy = score_tokens(model, inputs, response, 2, with_entropy=True)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(entropy, expected_entropy)
        g1 = torch.autograd.grad((expected + .1*expected_entropy).sum(), tuple(model.parameters()))
        g2 = torch.autograd.grad((actual + .1*entropy).sum(), tuple(model.parameters()))
        for a, b in zip(g1, g2):
            torch.testing.assert_close(a, b)

    def test_sampling_tracker_does_not_modify_distribution(self):
        tracker = CaptureSampleLogProbs()
        scores = torch.randn(1, 11)
        ids = torch.tensor([[1, 2]])
        self.assertIs(tracker(ids, scores), scores)
        next_scores = torch.randn(1, 11)
        tracker(torch.tensor([[1, 2, 7]]), next_scores)
        actual = tracker.finish(torch.tensor(4))
        expected = torch.stack([scores.log_softmax(-1)[0, 7],
                                next_scores.log_softmax(-1)[0, 4]])
        torch.testing.assert_close(actual, expected)

    def test_image_token_types_only_extend_with_text(self):
        inputs = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2),
                  "mm_token_type_ids": torch.tensor([[0, 1]])}
        extended, p = extend_inputs(inputs, torch.tensor([3, 4]), torch.device("cpu"))
        self.assertEqual(p, 2)
        self.assertEqual(extended["mm_token_type_ids"].tolist(), [[0, 1, 0, 0]])
        self.assertEqual(inputs["input_ids"].shape[1], 2)

    def test_native_tools_require_closed_reasoning_and_complete_calls(self):
        schema = [{"function": {"name": "draw", "parameters": {
            "properties": {"operations": {"type": "array"}, "erase": {"type": "boolean"}}}}}]
        text = ('A plan</think><tool_call><function=draw>'
                '<parameter=operations>[{"x": 10}]</parameter>'
                '<parameter=erase>false</parameter></function></tool_call><|im_end|>')
        calls, reasoning, error = parse_qwen_tools(text, schema)
        self.assertEqual(error, "")
        self.assertEqual(reasoning, "A plan")
        self.assertEqual(calls[0]["function"]["arguments"], {"operations": [{"x": 10}], "erase": False})
        self.assertFalse(parse_qwen_tools(text.replace("</think>", ""), schema)[0])
        self.assertFalse(parse_qwen_tools(text.replace("</tool_call>", ""), schema)[0])
        self.assertFalse(parse_qwen_tools(text.replace("false", "garbage"), schema)[0])

    def test_duplicate_parameter_cannot_silently_override(self):
        t = ('r</think><tool_call><function=draw><parameter=x>1</parameter>'
             '<parameter=x>2</parameter></function></tool_call>')
        self.assertEqual(parse_qwen_tools(t, [])[2], "duplicate_parameter")

    def test_missing_generation_config_still_stops_at_native_chat_eos(self):
        tokenizer = SimpleNamespace(eos_token_id=248046,
                                    convert_tokens_to_ids=lambda s: 248046,
                                    convert_ids_to_tokens=lambda i: "<|im_end|>")
        self.assertEqual(native_eos_ids(tokenizer, 248044), [248044, 248046])
        self.assertEqual(native_eos_ids(tokenizer, [248044, 248046]), [248044, 248046])
        self.assertEqual(native_eos_ids(tokenizer, None), [248046])

    def test_rollout_evidence_preserves_pixels_and_tokens_without_expanded_tensors(self):
        from PIL import Image
        picture = io.BytesIO()
        Image.new("RGB", (8, 8), "black").save(picture, format="PNG")
        raw = picture.getvalue()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "draw"},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(raw).decode()}},
        ]}]
        values = {"input_ids": torch.tensor([[1, 2, 3]]),
                  "attention_mask": torch.ones(1, 3, dtype=torch.long),
                  "pixel_values": torch.randn(1024, 128),
                  "image_grid_thw": torch.tensor([[1, 2, 2]])}
        pixels = values["pixel_values"].clone()
        response, logp = torch.tensor([4, 5]), torch.tensor([-.3, -.7])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in (0, 1):
                save_turn_audit(root, index, messages, values, values, response, logp)
            record = json.loads((root / "turn_000.json").read_text())
            self.assertFalse(record["expanded_image_tensors_saved"])
            self.assertFalse(list(root.glob("*.pt")))
            self.assertEqual(len(list((root / "observed_images").iterdir())), 1)
            reference = record["messages"][0]["content"][1]
            self.assertEqual((root / reference["path"]).read_bytes(), raw)
            self.assertEqual(reference["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(record["inputs"]["input_ids"]["values"], [[1, 2, 3]])
            self.assertNotIn("values", record["inputs"]["pixel_values"])
            digest = hashlib.sha256(pixels.view(torch.uint8).numpy().tobytes()).hexdigest()
            self.assertEqual(record["inputs"]["pixel_values"]["sha256"], digest)
            self.assertEqual(record["response_token_ids"], response.tolist())
            self.assertEqual(record["sampled_log_probs"], logp.tolist())
            self.assertTrue(messages[0]["content"][1]["image_url"]["url"].startswith("data:image/"))
            torch.testing.assert_close(values["pixel_values"], pixels, rtol=0, atol=0)
            self.assertLess((root / "turn_000.json").stat().st_size, pixels.numel())

    def test_configs_share_fixed_algorithm_contract(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("baseline.json", "smoke.json"):
            validate_config(json.loads((root / "configs" / name).read_text()))

    def test_checkpoint_is_atomic_complete_restorable_and_detects_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "step1"
            model = TinyModel()
            optimizer = torch.optim.AdamW(model.parameters())
            model.head(model.model(torch.tensor([[1]])).last_hidden_state).sum().backward()
            optimizer.step()
            probe = {"inputs": {"input_ids": torch.tensor([[1]])}, "expected": torch.tensor([.2])}
            save_checkpoint(destination, model, TinyProcessor(), optimizer, 1, {"fixed": 42},
                            restore_probe=probe)
            self.assertFalse(destination.with_name("step1.partial").exists())
            manifest = verify_checkpoint(destination, {"fixed": 42})
            self.assertEqual(manifest["update"], 1)
            self.assertIn("restore_probe.pt", manifest["files"])
            saved_probe = torch.load(destination / "restore_probe.pt", weights_only=True)
            torch.testing.assert_close(saved_probe["expected"], probe["expected"])
            other = TinyModel()
            other.load_state_dict(torch.load(destination / "model/toy.pt", weights_only=True))
            opt2 = torch.optim.AdamW(other.parameters())
            self.assertEqual(restore_optimizer(destination, opt2), 1)
            self.assertTrue(all(s["step"] == 1 for s in opt2.state.values()))
            with self.assertRaises(ValueError):
                verify_checkpoint(destination, {"fixed": 43})
            (destination / "model/processor.json").write_text('{"bad": true}')
            with self.assertRaises(ValueError):
                verify_checkpoint(destination, {"fixed": 42})

    def _three_checkpoints(self, root):
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters())
        for update in (10, 20, 30):
            save_checkpoint(root / f"update_{update:04d}", model, TinyProcessor(),
                            optimizer, update, {"fixed": 42})

    def test_retention_keeps_latest_two_and_never_deletes_foreign_or_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._three_checkpoints(root)
            (root / "update_0040.partial").mkdir()
            (root / "update_0040.partial/unfinished").write_text("incomplete")
            model = TinyModel()
            save_checkpoint(root / "update_0099", model, TinyProcessor(),
                            torch.optim.AdamW(model.parameters()), 99, {"foreign": True})
            (root / "user-note.txt").write_text("keep")
            result = prune_checkpoints(root, {"fixed": 42}, 2)
            self.assertEqual(result["retained"], ["update_0020", "update_0030"])
            self.assertEqual([r["update"] for r in result["removed"]], [10])
            self.assertFalse((root / "update_0010").exists())
            self.assertTrue((root / "update_0099").is_dir())
            self.assertTrue((root / "update_0040.partial/unfinished").is_file())
            self.assertTrue((root / "user-note.txt").is_file())
            self.assertEqual(prune_checkpoints(root, {"fixed": 42}, 2)["removed"], [])
            with self.assertRaises(ValueError):
                prune_checkpoints(root, {"fixed": 42}, 1)

    def test_corrupt_newest_checkpoint_never_authorizes_retention(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._three_checkpoints(root)
            (root / "update_0030/model/processor.json").write_text('{"bad": true}')
            with self.assertRaises(ValueError):
                prune_checkpoints(root, {"fixed": 42}, 2)
            self.assertTrue((root / "update_0010").exists())
            self.assertTrue((root / "update_0020").exists())

    def test_retention_rejects_untracked_files_in_deletion_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._three_checkpoints(root)
            (root / "update_0010/personal.txt").write_text("do not remove")
            with self.assertRaises(ValueError):
                prune_checkpoints(root, {"fixed": 42}, 2)
            self.assertTrue((root / "update_0010/personal.txt").is_file())

    def test_checkpoint_symlink_cannot_escape_retention_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "checkpoints"
            root.mkdir()
            self._three_checkpoints(root)
            other = Path(folder) / "not-a-checkpoint"
            other.mkdir()
            (other / "preserve.txt").write_text("keep")
            try:
                (root / "update_0001").symlink_to(other, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"OS does not allow symlinks: {exc}")
            prune_checkpoints(root, {"fixed": 42}, 2)
            self.assertTrue((other / "preserve.txt").exists())
            self.assertTrue((root / "update_0001").is_symlink())


if __name__ == "__main__":
    unittest.main()
