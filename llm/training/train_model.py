"""Fine-tune Qwen2.5-Coder on the local training data with LoRA."""

from typing import Any

from llm.training.commonscripts.common import CHUNK_CHARACTERS, CausalLanguageModelCollator, ChatDataset, \
    collect_examples, ensure_training_data, parse_args


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


def main() -> None:
	args = parse_args()
	ensure_training_data(args.data_dir)
	try:
		import torch
		from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
		from transformers import (
			AutoModelForCausalLM,
			AutoTokenizer,
			Trainer,
			TrainingArguments,
		)
	except ImportError as exc:
		raise SystemExit(
			"Install dependencies with: python -m pip install torch transformers peft accelerate"
		) from exc

	examples = collect_examples(args.data_dir, args.max_files)
	if not examples:
		raise SystemExit(f"No usable training examples found under {args.data_dir}")
	print(f"Loaded {len(examples)} examples from {args.data_dir}")

	tokenizer = AutoTokenizer.from_pretrained(args.model)
	if tokenizer.pad_token_id is None:
		if tokenizer.eos_token is None:
			raise SystemExit("The tokenizer has neither a pad token nor an EOS token.")
		tokenizer.pad_token = tokenizer.eos_token
	pad_token_id = tokenizer.pad_token_id
	if pad_token_id is None:
		raise SystemExit("Could not determine the tokenizer pad token ID.")
	tokenizer.padding_side = "right"

	device = "cuda" if torch.cuda.is_available() else "cpu"
	model_kwargs: dict[str, Any] = {}
	if device == "cuda":
		model_kwargs["torch_dtype"] = (
			torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
		)
	if args.use_4bit:
		if device != "cuda":
			raise SystemExit("4-bit loading requires a CUDA GPU and bitsandbytes.")
		try:
			from transformers import BitsAndBytesConfig
		except ImportError as exc:
			raise SystemExit("Install bitsandbytes to use --use-4bit.") from exc
		model_kwargs["quantization_config"] = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
			bnb_4bit_use_double_quant=True,
		)
		model_kwargs["device_map"] = {"": 0}

	model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
	if args.use_4bit:
		model = prepare_model_for_kbit_training(model)
	model.config.use_cache = False
	model = get_peft_model(
		model,
		LoraConfig(
			r=16,
			lora_alpha=32,
			lora_dropout=0.05,
			bias="none",
			task_type="CAUSAL_LM",
			target_modules=[
				"q_proj",
				"k_proj",
				"v_proj",
				"o_proj",
				"gate_proj",
				"up_proj",
				"down_proj",
			],
		),
	)
	model.print_trainable_parameters()

	dataset = ChatDataset(examples, tokenizer, args.max_seq_length)
	if not len(dataset):
		raise SystemExit("No examples fit the selected sequence length.")
	training_args = TrainingArguments(
		output_dir=str(args.output_dir),
		num_train_epochs=args.epochs,
		per_device_train_batch_size=args.batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		learning_rate=args.learning_rate,
		weight_decay=0.01,
		logging_steps=10,
		save_strategy="epoch",
		fp16=device == "cuda" and not torch.cuda.is_bf16_supported(),
		bf16=device == "cuda" and torch.cuda.is_bf16_supported(),
		gradient_checkpointing=True,
		optim="adamw_torch",
		report_to="none",
		remove_unused_columns=False,
	)
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=dataset,
		data_collator=CausalLanguageModelCollator(pad_token_id),
	)
	trainer.train()
	trainer.save_model(str(args.output_dir))
	tokenizer.save_pretrained(args.output_dir)
	print(f"LoRA adapter and tokenizer saved to: {args.output_dir}")


if __name__ == "__main__":
	main()