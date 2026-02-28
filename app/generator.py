from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LocalGenerator:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=450,
            do_sample=False,
        )

    def generate(self, prompt: str) -> str:
        result = self.pipe(prompt)[0]["generated_text"]
        if result.startswith(prompt):
            return result[len(prompt) :].strip()
        return result.strip()
