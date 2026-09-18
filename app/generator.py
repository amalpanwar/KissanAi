from __future__ import annotations

import os


class LocalGenerator:
    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        online = os.getenv("KISAANAI_HF_ONLINE", "").strip().lower() in {"1", "true", "yes"}
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=not online)
        device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else -1
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            local_files_only=not online,
            torch_dtype=torch.float32 if device == -1 else torch.float16,
        )
        self.pipe = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer,
            max_new_tokens=int(os.getenv("KISAANAI_MAX_NEW_TOKENS", "512")),
            do_sample=False, device=device,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def generate(self, prompt: str) -> str:
        # Instruct checkpoints must receive the conversation template used in training.
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        ) if self.tokenizer.chat_template else prompt
        result = self.pipe(formatted, return_full_text=False)[0]["generated_text"]
        return result.strip()
