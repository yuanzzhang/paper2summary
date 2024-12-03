from dataclasses import dataclass, field
from typing import List

@dataclass
class GlobalConfig:
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "4"
    use_wandb: bool = True
    project: str = "paper2summary"

    # Data splits
    train_split: str = "train[:10%]"
    val_split: str = "validation[:10%]"

    # Context window
    padding: str = "max_length"
    truncation: bool = True
    context_length: int = 10182
    summary_length: int = 350

    # Output paths
    log_file: str = "train.log"
    output_dir: str = "./output/lora"


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.01


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 4
    weight_decay: float = 5e-3
    warmup_ratio: float = 0.01
    bf16: bool = True

    # Checkpointing
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    eval_steps: int = 20
    save_steps: int = 500
    save_total_limit: int = 2
    logging_steps: int = 10
    run_name: str = "lora"


@dataclass
class DataLoaderConfig:
    train_batch_size: int = 1
    eval_batch_size: int = 1
    drop_last: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2