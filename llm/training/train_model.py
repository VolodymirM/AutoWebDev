"""Fine-tune Qwen2.5-Coder on the local training data with LoRA."""

from typing import Any

if __package__:
	from .commonscripts.common import (
		CausalLanguageModelCollator,
		ChatDataset,
		collect_examples,
		ensure_training_data,
		parse_args,
	)
else:
	from commonscripts.common import (
		CausalLanguageModelCollator,
		ChatDataset,
		collect_examples,
		ensure_training_data,
		parse_args,
	)


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
		save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
		fp16=device == "cuda" and not torch.cuda.is_bf16_supported(),
		bf16=device == "cuda" and torch.cuda.is_bf16_supported(),
		gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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