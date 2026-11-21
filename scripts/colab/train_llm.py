"""Colab QLoRA/LoRA fine-tuning and Ollama Modelfile/GGUF prep script for Qwen2.5-3B-Instruct."""

import argparse
import gc
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from draft.config import AppConfig, load_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Colab Stage 3 QLoRA/LoRA fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune Stage 3 Draft Generator LLM.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--sft-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--merge-fp16", action="store_true")
    return parser.parse_args()


def _write_ollama_modelfile(cfg: AppConfig, target_path: Path) -> None:
    """Generate an Ollama Modelfile using Qwen2.5 ChatML formatting and system.j2."""
    system_prompt = (cfg.paths.prompts_dir / "system.j2").read_text(encoding="utf-8").strip()
    template = (
        'TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}\n'
        "{{ .Content }}<|im_end|>\n"
        '{{ end }}<|im_start|>assistant\n"""'
    )
    modelfile = "\n".join(
        [
            f"FROM ./{cfg.generation.ollama_model}-q8_0.gguf",
            template,
            f'SYSTEM """{system_prompt}"""',
            f"PARAMETER temperature {cfg.generation.temperature}",
            f"PARAMETER num_predict {cfg.generation.max_new_tokens}",
            'PARAMETER stop "<|im_end|>"',
        ]
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(modelfile + "\n", encoding="utf-8")


def _build_sft_config(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    use_bf16: bool,
    use_4bit: bool,
) -> SFTConfig:
    """Construct TRL SFTConfig cleanly across TRL version parameter variations."""
    sig_params = inspect.signature(SFTConfig.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(checkpoint_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 30,
        "logging_steps": 25,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "bf16": use_bf16,
        "fp16": torch.cuda.is_available() and not use_bf16,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit" if use_4bit else "adamw_torch",
        "report_to": "none",
    }
    if "max_length" in sig_params:
        kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in sig_params:
        kwargs["max_seq_length"] = args.max_seq_length
    return SFTConfig(**kwargs)


def _merge_adapter_fp16(base_model_name: str, adapter_dir: Path, merged_dir: Path) -> None:
    """Merge trained LoRA adapter into FP16 base weights for GGUF export."""
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged = peft_model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(merged_dir))


def main() -> None:
    """Run QLoRA/LoRA SFT training on Colab and emit metrics, Modelfile, and GGUF steps."""
    args = parse_args()
    cfg = load_config(args.config)
    sft_dir = args.sft_dir or (cfg.paths.processed_dir / "sft")
    output_dir = args.output_dir or (cfg.paths.artifacts_dir / "llm" / "lora_adapter")
    checkpoint_dir = output_dir.parent / "checkpoints"

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(sft_dir / "sft_train.jsonl"),
            "eval": str(sft_dir / "sft_eval.jsonl"),
        },
    )
    train_ds = dataset["train"].select_columns(["messages"])
    eval_ds = dataset["eval"].select_columns(["messages"])

    tokenizer = AutoTokenizer.from_pretrained(cfg.generation.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_4bit = not args.no_4bit

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.generation.base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.generation.base_model,
            torch_dtype=compute_dtype,
            device_map="auto",
        )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
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
    )
    sft_config = _build_sft_config(args, checkpoint_dir, use_bf16, use_4bit=use_4bit)

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    tok_kwarg = "processing_class" if "processing_class" in trainer_params else "tokenizer"
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        **{tok_kwarg: tokenizer},
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    modelfile_path = cfg.paths.artifacts_dir / "llm" / "Modelfile"
    _write_ollama_modelfile(cfg, modelfile_path)

    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.paths.reports_dir / "llm_training_metrics.json"
    report = {
        "base_model": cfg.generation.base_model,
        "use_4bit": use_4bit,
        "lora_r": args.lora_r,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "log_history": trainer.state.log_history,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"LoRA adapter saved to: {output_dir}")
    print(f"Ollama Modelfile saved to: {modelfile_path}")
    print(f"Metrics report saved to: {report_path}")

    merged_dir = cfg.paths.artifacts_dir / "llm" / "merged_fp16"
    if args.merge_fp16:
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _merge_adapter_fp16(cfg.generation.base_model, output_dir, merged_dir)
        print(f"Merged FP16 model saved to: {merged_dir}")

    print("\nNext steps for local Ollama GGUF deployment:")
    print(
        "  1) Pure-Python Q8_0 GGUF conversion (~3.2 GB, no C++ build required):\n"
        f"     python llama.cpp/convert_hf_to_gguf.py {merged_dir} "
        f"--outfile artifacts/llm/{cfg.generation.ollama_model}-q8_0.gguf --outtype q8_0"
    )
    print(
        f"     ollama create {cfg.generation.ollama_model} -f {modelfile_path}"
    )
    print(
        "  2) Optional smaller Q4_K_M (~2.0 GB) via f16 GGUF + Ollama quantize:\n"
        f"     python llama.cpp/convert_hf_to_gguf.py {merged_dir} "
        f"--outfile artifacts/llm/{cfg.generation.ollama_model}-f16.gguf --outtype f16\n"
        f"     # Update FROM in {modelfile_path} to ./{cfg.generation.ollama_model}-f16.gguf\n"
        f"     ollama create --quantize q4_k_m {cfg.generation.ollama_model} -f {modelfile_path}"
    )


if __name__ == "__main__":
    main()
