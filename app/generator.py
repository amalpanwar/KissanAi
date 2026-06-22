from __future__ import annotations

import os


class LocalGenerator:
    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        online = os.getenv("KISAANAI_HF_ONLINE", "").strip().lower() in {"1", "true", "yes"}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=not online)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            local_files_only=not online,
        )
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=160,
            do_sample=False,
            temperature=0.2,
            device=device,
        )

    def generate(self, prompt: str) -> str:
        result = self.pipe(prompt, return_full_text=False)[0]["generated_text"]
        if result.startswith(prompt):
            return result[len(prompt) :].strip()
        return result.strip()
