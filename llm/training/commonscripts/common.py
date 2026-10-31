from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from torch.utils.data import Dataset

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
SOURCE_EXTENSIONS = {".java", ".ts", ".js", ".html", ".css", ".json"}
SKIP_DIRECTORIES = {".git", "node_modules", "target", "dist", "build"}
CHUNK_CHARACTERS = 6000


def make_messages(prompt: str, answer: str) -> list[dict[str, str]]:
	return [
		{"role": "user", "content": prompt.strip()},
		{"role": "assistant", "content": answer.strip()},
	]


def jsonl_messages(record: dict[str, Any]) -> list[dict[str, str]] | None:
	messages = record.get("messages")
	if isinstance(messages, list):
		valid_messages = [
			{"role": message["role"], "content": message["content"]}
			for message in messages
			if isinstance(message, dict)
			and message.get("role") in {"user", "assistant"}
			and isinstance(message.get("content"), str)
		]
		if len(valid_messages) >= 2 and valid_messages[-1]["role"] == "assistant":
			return valid_messages

	title = record.get("title")
	body = record.get("body")
	patch = record.get("fix_patch")
	if isinstance(title, str) and isinstance(patch, str) and patch.strip():
		prompt_parts = [f"Implement the requested code change: {title.strip()}"]
		if isinstance(body, str) and body.strip():
			prompt_parts.append(body.strip())
		resolved_issues = record.get("resolved_issues")
		if isinstance(resolved_issues, list):
			for issue in resolved_issues:
				if isinstance(issue, dict):
					issue_title = issue.get("title")
					issue_body = issue.get("body")
					if isinstance(issue_title, str) and issue_title.strip():
						prompt_parts.append(f"Related issue: {issue_title.strip()}")
					if isinstance(issue_body, str) and issue_body.strip():
						prompt_parts.append(issue_body.strip())
		return make_messages("\n\n".join(prompt_parts), patch)

	prompt = record.get("instruction", record.get("prompt", record.get("question")))
	answer = record.get(
		"output", record.get("completion", record.get("response", record.get("answer")))
	)
	if isinstance(prompt, str) and isinstance(answer, str) and answer.strip():
		context = record.get("input", record.get("context"))
		if isinstance(context, str) and context.strip():
			prompt = f"{prompt.strip()}\n\n{context.strip()}"
		return make_messages(prompt, answer)
	return None


def split_text(text: str, limit: int = CHUNK_CHARACTERS) -> list[str]:
	chunks: list[str] = []
	current_lines: list[str] = []
	current_length = 0
	for line in text.splitlines():
		while len(line) > limit:
			if current_lines:
				chunks.append("\n".join(current_lines))
				current_lines = []
				current_length = 0
			chunks.append(line[:limit])
			line = line[limit:]
		if current_length + len(line) + 1 > limit and current_lines:
			chunks.append("\n".join(current_lines))
			current_lines = []
			current_length = 0
		current_lines.append(line)
		current_length += len(line) + 1
	if current_lines:
		chunks.append("\n".join(current_lines))
	return [chunk.strip() for chunk in chunks if chunk.strip()]


class ChatDataset(Dataset):
	def __init__(self, examples: list[list[dict[str, str]]], tokenizer: Any, max_length: int):
		self.samples: list[dict[str, list[int]]] = []
		for messages in examples:
			prompt_ids = tokenizer.apply_chat_template(
				messages[:-1], tokenize=True, add_generation_prompt=True
			)
			all_ids = tokenizer.apply_chat_template(
				messages, tokenize=True, add_generation_prompt=False
			)
			if all_ids[: len(prompt_ids)] != prompt_ids:
				continue
			answer_ids = all_ids[len(prompt_ids) :]
			if len(answer_ids) > max_length:
				answer_ids = answer_ids[:max_length]
			prompt_budget = max_length - len(answer_ids)
			if prompt_budget <= 0 or not answer_ids:
				continue
			kept_prompt = prompt_ids[-prompt_budget:]
			input_ids = kept_prompt + answer_ids
			self.samples.append(
				{
					"input_ids": input_ids,
					"attention_mask": [1] * len(input_ids),
					"labels": [-100] * len(kept_prompt) + answer_ids,
				}
			)

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int) -> dict[str, list[int]]:
		return self.samples[index]

class CausalLanguageModelCollator:
	def __init__(self, pad_token_id: int):
		self.pad_token_id = pad_token_id

	def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
		import torch

		max_length = max(len(feature["input_ids"]) for feature in features)
		input_ids = []
		attention_masks = []
		labels = []
		for feature in features:
			padding = max_length - len(feature["input_ids"])
			input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
			attention_masks.append(feature["attention_mask"] + [0] * padding)
			labels.append(feature["labels"] + [-100] * padding)
		return {
			"input_ids": torch.tensor(input_ids, dtype=torch.long),
			"attention_mask": torch.tensor(attention_masks, dtype=torch.long),
			"labels": torch.tensor(labels, dtype=torch.long),
		}

def collect_examples(data_dir: Path, max_files: int | None = None) -> list[list[dict[str, str]]]:
	examples: list[list[dict[str, str]]] = []
	files_seen = 0
	jsonl_dir = data_dir / "automationexamples"
	if jsonl_dir.exists():
		for path in sorted(jsonl_dir.rglob("*.jsonl")):
			if max_files is not None and files_seen >= max_files:
				break
			files_seen += 1
			with path.open(encoding="utf-8") as source:
				for line_number, line in enumerate(source, start=1):
					if not line.strip():
						continue
					try:
						record = json.loads(line)
					except json.JSONDecodeError as exc:
						print(f"Skipping invalid JSON at {path}:{line_number}: {exc}")
						continue
					if isinstance(record, dict):
						messages = jsonl_messages(record)
						if messages:
							examples.append(messages)

	for folder_name in ("documentation", "trainingrepos"):
		folder = data_dir / folder_name
		if not folder.exists():
			continue
		for path in sorted(folder.rglob("*")):
			if not path.is_file() or any(part in SKIP_DIRECTORIES for part in path.parts):
				continue
			if folder_name == "documentation":
				if path.suffix.casefold() != ".data":
					continue
			else:
				if path.suffix.casefold() not in SOURCE_EXTENSIONS:
					continue
			if max_files is not None and files_seen >= max_files:
				break
			files_seen += 1
			try:
				text = path.read_text(encoding="utf-8")
			except (OSError, UnicodeDecodeError) as exc:
				print(f"Skipping unreadable file {path}: {exc}")
				continue
			for chunk_number, chunk in enumerate(split_text(text), start=1):
				if folder_name == "documentation":
					prompt = "Study this technical reference passage and reproduce it accurately."
				else:
					prompt = (
						f"Write the contents of this {path.suffix.lstrip('.')} source-code example "
						f"(part {chunk_number}) accurately."
					)
				examples.append(make_messages(prompt, chunk))
	return examples

def parse_args(continuing: bool = False) -> argparse.Namespace:
	data_dir = Path(__file__).resolve().parent.parent
	parser = argparse.ArgumentParser(description=__doc__)
	if continuing:
		parser.add_argument("--checkpoint", type=Path, required=True)
	parser.add_argument("--model", default=MODEL_NAME)
	parser.add_argument("--data-dir", type=Path, default=data_dir)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None
		if continuing
		else data_dir.parent / "running" / "qwen2.5-coder-3b-lora",
	)
	parser.add_argument("--max-seq-length", type=int, default=2048)
	parser.add_argument("--epochs", type=float, default=1.0)
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
	parser.add_argument("--learning-rate", type=float, default=2e-4)
	parser.add_argument("--max-files", type=int, default=None)
	parser.add_argument("--use-4bit", action="store_true")
	return parser.parse_args()

def ensure_training_data(data_dir: Path) -> None:
	data_dir = data_dir.resolve()
	required_folders = ("trainingrepos", "automationexamples")
	missing_folders = [name for name in required_folders if not (data_dir / name).is_dir()]
	if not missing_folders:
		return

	download_script = Path(__file__).resolve().parent.parent / "download_data.py"
	print(f"Missing training data folders: {', '.join(missing_folders)}. Downloading data...")
	subprocess.run([sys.executable, str(download_script)], check=True)

	missing_folders = [name for name in required_folders if not (data_dir / name).is_dir()]
	if missing_folders:
		raise SystemExit(
			"Data download completed without creating: " + ", ".join(missing_folders)
		)