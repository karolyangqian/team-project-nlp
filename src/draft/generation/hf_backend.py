"""Hugging Face Transformers + optional PEFT LoRA backend for Stage 3 generation."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from draft.generation.prompts import PromptMessages


class HFPeftGenerator:
    """Local or Colab causal LLM generator with optional QLoRA adapter loading."""

    def __init__(
        self,
        base_model_name: str,
        adapter_dir: Path | None = None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        default_max_new_tokens: int = 384,
        default_temperature: float = 0.2,
    ) -> None:
        self.model_name = str(adapter_dir) if adapter_dir is not None else base_model_name
        self.default_max_new_tokens = default_max_new_tokens
        self.default_temperature = default_temperature

        tok_source = (
            str(adapter_dir)
            if adapter_dir is not None and (adapter_dir / "tokenizer_config.json").exists()
            else base_model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tok_source)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict = {}
        if load_in_4bit and torch.cuda.is_available():
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model_kwargs["device_map"] = device_map
        else:
            model_kwargs["torch_dtype"] = (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                model_kwargs["device_map"] = device_map

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        if adapter_dir is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        else:
            self.model = base_model
        self.model.eval()

    def generate(
        self,
        messages: PromptMessages,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Render chat template, generate continuation tokens, and decode reply text."""
        prompt_text = self.tokenizer.apply_chat_template(
            messages.to_chat_messages(),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len = int(inputs["input_ids"].shape[1])

        temp = self.default_temperature if temperature is None else temperature
        max_tok = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens
        gen_kwargs = {
            "max_new_tokens": max_tok,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": temp > 0.0,
        }
        if temp > 0.0:
            gen_kwargs["temperature"] = temp

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][input_len:]
        return str(self.tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()
