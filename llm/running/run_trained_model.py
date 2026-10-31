"""Run the locally fine-tuned Qwen2.5-Coder LoRA adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parent / "qwen2.5-coder-3b-lora"
DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."
MAX_HISTORY_TURNS = 8


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--adapter-dir",
		type=Path,
		default=DEFAULT_ADAPTER_DIR,
		help="Path to the saved LoRA adapter directory.",
	)
	parser.add_argument(
		"--base-model",
		default=None,
		help="Override the base model recorded in adapter_config.json.",
	)
	parser.add_argument(
		"--device",
		choices=("auto", "cpu", "cuda"),
		default="auto",
		help="Choose where to run the model (default: use CUDA when available).",
	)
	parser.add_argument(
		"--prompt",
		default=None,
		help="Run one prompt and exit; omit to start interactive chat.",
	)
	parser.add_argument(
		"--system-prompt",
		default=DEFAULT_SYSTEM_PROMPT,
		help="System message used to start the conversation.",
	)
	parser.add_argument("--max-new-tokens", type=int, default=512)
	parser.add_argument("--temperature", type=float, default=0.7)
	parser.add_argument("--top-p", type=float, default=0.9)
	return parser.parse_args()


def load_model(
	adapter_dir: Path,
	base_model_override: str | None,
	requested_device: str,
) -> tuple[Any, Any, Any, str]:
	try:
		import torch
		from peft import PeftConfig, PeftModel
		from transformers import AutoModelForCausalLM, AutoTokenizer
	except ImportError as exc:
		raise SystemExit(
			"Install dependencies with: python -m pip install torch transformers peft accelerate"
		) from exc

	if not adapter_dir.is_dir():
		raise SystemExit(f"Adapter directory does not exist: {adapter_dir}")
	if requested_device == "cuda" and not torch.cuda.is_available():
		raise SystemExit("CUDA was requested, but PyTorch cannot detect a CUDA GPU.")
	device = (
		"cuda"
		if requested_device == "auto" and torch.cuda.is_available()
		else "cpu"
		if requested_device == "auto"
		else requested_device
	)

	adapter_config = PeftConfig.from_pretrained(str(adapter_dir))
	base_model = base_model_override or adapter_config.base_model_name_or_path
	if not base_model:
		raise SystemExit(
			"Could not determine the base model. Pass it with --base-model."
		)

	model_kwargs: dict[str, Any] = {}
	if device == "cuda":
		model_kwargs["torch_dtype"] = (
			torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
		)

	print(f"Loading base model: {base_model}")
	tokenizer = AutoTokenizer.from_pretrained(base_model)
	if tokenizer.pad_token_id is None:
		if tokenizer.eos_token is None:
			raise SystemExit("The tokenizer has neither a pad token nor an EOS token.")
		tokenizer.pad_token = tokenizer.eos_token

	model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
	model = PeftModel.from_pretrained(model, str(adapter_dir))
	model.to(device)
	model.eval()
	model.config.pad_token_id = tokenizer.pad_token_id
	print(f"Model ready on {device}.")
	return torch, model, tokenizer, device


def generate_reply(
	torch: Any,
	model: Any,
	tokenizer: Any,
	device: str,
	messages: list[dict[str, str]],
	max_new_tokens: int,
	temperature: float,
	top_p: float,
) -> str:
	encoded = tokenizer.apply_chat_template(
		messages,
		tokenize=True,
		add_generation_prompt=True,
		return_tensors="pt",
		return_dict=True,
	)
	encoded = {name: value.to(device) for name, value in encoded.items()}
	generation_options: dict[str, Any] = {
		"max_new_tokens": max_new_tokens,
		"pad_token_id": tokenizer.pad_token_id,
		"do_sample": temperature > 0,
	}
	if temperature > 0:
		generation_options["temperature"] = temperature
		generation_options["top_p"] = top_p

	with torch.inference_mode():
		output = model.generate(**encoded, **generation_options)
	new_tokens = output[0, encoded["input_ids"].shape[-1] :]
	return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
	args = parse_args()
	if args.max_new_tokens < 1:
		raise SystemExit("--max-new-tokens must be at least 1.")
	if args.temperature < 0:
		raise SystemExit("--temperature cannot be negative.")
	if not 0 < args.top_p <= 1:
		raise SystemExit("--top-p must be greater than 0 and at most 1.")

	torch, model, tokenizer, device = load_model(
		args.adapter_dir, args.base_model, args.device
	)
	messages = [{"role": "system", "content": args.system_prompt}]
	if args.prompt is not None:
		messages.append({"role": "user", "content": args.prompt})
		print(
			generate_reply(
				torch,
				model,
				tokenizer,
				device,
				messages,
				args.max_new_tokens,
				args.temperature,
				args.top_p,
			)
		)
		return

	print("Chat started. Type /reset to clear history or /exit to quit.")
	while True:
		try:
			prompt = input("You> ").strip()
		except (EOFError, KeyboardInterrupt):
			print()
			break
		if not prompt:
			continue
		if prompt.casefold() in {"/exit", "/quit"}:
			break
		if prompt.casefold() == "/reset":
			messages = [{"role": "system", "content": args.system_prompt}]
			print("Conversation history cleared.")
			continue

		messages.append({"role": "user", "content": prompt})
		reply = generate_reply(
			torch,
			model,
			tokenizer,
			device,
			messages,
			args.max_new_tokens,
			args.temperature,
			args.top_p,
		)
		messages.append({"role": "assistant", "content": reply})
		print(f"Assistant> {reply}\n")
		if len(messages) > 1 + MAX_HISTORY_TURNS * 2:
			del messages[1:3]


if __name__ == "__main__":
	main()
