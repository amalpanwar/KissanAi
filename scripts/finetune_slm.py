from __future__ import annotations

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from app.config import load_config


def format_row(row: dict) -> dict:
    text = f"### Instruction:\n{row['prompt']}\n\n### Response:\n{row['response']}"
    return {"text": text}


def main() -> None:
    cfg = load_config()
    model_name = cfg.generator_model

    ds = load_dataset("json", data_files=cfg.paths["finetune_data"], split="train")
    ds = ds.map(format_row)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    args = TrainingArguments(
        output_dir="models/western-up-lora",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        fp16=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model("models/western-up-lora")
    print("Saved LoRA adapter at models/western-up-lora")


if __name__ == "__main__":
    main()
