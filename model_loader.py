"""Loads the base language model and tokenizer for GRPO training.

Uses Unsloth on CUDA (HuggingFace Space T4) when compatible.
Falls back to standard HuggingFace + PEFT on CPU/MPS or on version mismatch.
"""
# Import unsloth FIRST before transformers/peft to apply its patches correctly
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import os
import torch

MODEL_NAME = os.getenv("BASE_MODEL", "unsloth/Qwen2.5-1.5B-Instruct")
MAX_SEQ_LENGTH = 2048

_model = None
_tokenizer = None


def _get_resume_adapter_path() -> str | None:
    """Return adapter path to resume from, if configured and present."""
    resume_path = os.getenv("RESUME_ADAPTER_PATH", "").strip()
    if not resume_path:
        return None
    if os.path.isdir(resume_path):
        return resume_path
    return None


def _resolve_hf_model_name(name: str) -> str:
    """Resolve a Hugging Face model id from BASE_MODEL.

    Examples:
    - unsloth/Qwen2.5-1.5B-Instruct -> Qwen/Qwen2.5-1.5B-Instruct
    - unsloth/SomeOrg/Model          -> SomeOrg/Model
    - Qwen/Qwen2.5-1.5B-Instruct     -> Qwen/Qwen2.5-1.5B-Instruct
    """
    if not name.startswith("unsloth/"):
        return name

    raw = name[len("unsloth/") :]
    # If already org/model after removing unsloth, keep it.
    if "/" in raw:
        return raw

    # Common unsloth shorthand for Qwen models.
    if raw.startswith("Qwen"):
        return f"Qwen/{raw}"

    return raw


def get_model_and_tokenizer():
    """Load and cache the model + tokenizer. Safe to call multiple times."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    resume_adapter = _get_resume_adapter_path()
    use_unsloth = (os.getenv("USE_UNSLOTH", "1").lower() in ("1", "true", "yes", "on"))

    # If resuming from an existing adapter, always use HF+PEFT load path
    # for predictable adapter restore behavior.
    if resume_adapter:
        print(f"[MODEL] Found existing adapter, resuming from: {resume_adapter}", flush=True)
        _load_with_hf(resume_adapter_path=resume_adapter)
    elif torch.cuda.is_available() and use_unsloth:
        try:
            _load_with_unsloth()
        except Exception as e:
            # Keep first-time setup simple and robust on Spaces:
            # if Unsloth/TRL versions drift, continue with plain HF+PEFT.
            print(f"[MODEL] Unsloth load failed, falling back to HF+PEFT: {e}", flush=True)
            _load_with_hf()
    else:
        _load_with_hf()

    print(f"[MODEL] Loaded: {MODEL_NAME} | CUDA={torch.cuda.is_available()}", flush=True)
    return _model, _tokenizer


def _load_with_unsloth():
    """Load with Unsloth (CUDA path — used on HuggingFace Space T4)."""
    global _model, _tokenizer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    # Add LoRA adapters — only these weights get updated during GRPO
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    _model = model
    _tokenizer = tokenizer


def _load_with_hf(resume_adapter_path: str | None = None):
    """Load with standard HuggingFace + PEFT (CPU/MPS path — local dev)."""
    global _model, _tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, get_peft_model, LoraConfig, TaskType

    hf_name = _resolve_hf_model_name(MODEL_NAME)
    print(f"[MODEL] HF fallback model id: {hf_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    torch_dtype = torch.float32
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        dtype=torch_dtype,      # `torch_dtype` is deprecated in new transformers
        device_map="auto",
    )

    if resume_adapter_path:
        # Load previously saved adapter and keep it trainable for continued GRPO.
        _model = PeftModel.from_pretrained(
            model,
            resume_adapter_path,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        _model = get_peft_model(model, lora_config)
    _tokenizer = tokenizer
