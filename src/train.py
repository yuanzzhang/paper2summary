import os
from datetime import datetime
from dataclasses import asdict

import wandb
import torch
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    TrainingArguments,
    Trainer, 
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, PeftModel

from .paper_dataset import get_paper_dataset
from .utils import setup_logger
from .config.lora_config import (
    GlobalConfig,
    LoRAConfig,
    TrainingConfig,
    DataLoaderConfig
)


class LoggingCallback(TrainerCallback):
    def __init__(self, logger, wandb_logger=None):
        self.logger = logger
        self.wandb_logger = wandb_logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.info(f"Training Metrics: {logs}")
            if self.wandb_logger:
                self.wandb_logger.log(logs)


def main():
    global_config = GlobalConfig()
    lora_config = LoRAConfig()
    training_config = TrainingConfig()
    dataloader_config = DataLoaderConfig()

    # Set visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = global_config.device
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if global_config.use_wandb:
        wandb.init(project=global_config.project)

    # Setup directories
    output_dir = global_config.output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = setup_logger(os.path.join(output_dir, global_config.log_file))

    try:
        logger.info("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(global_config.model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Resume from existing checkpoint
        latest_checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
            if checkpoints:
                latest_dir = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
                latest_checkpoint = os.path.join(checkpoint_dir, latest_dir)
                logger.info(f"Found checkpoint: {latest_checkpoint}")

        model = LlamaForCausalLM.from_pretrained(
            global_config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"
        )

        if latest_checkpoint:
            logger.info(f"Loading model from checkpoint: {latest_checkpoint}")
            model = PeftModel.from_pretrained(model, latest_checkpoint, is_trainable=True)
        else:
            logger.info("Initializing new model...")
            
            # Print total parameters before LoRA
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Total parameters before LoRA: {total_params:,}")

            # Apply LoRA
            peft_config = LoraConfig(**asdict(lora_config))
            model = get_peft_model(model, peft_config)

        trainable_params, all_param = model.get_nb_trainable_parameters()
        logger.info(
            f"{global_config.model_name} - "
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )

        logger.info("Loading datasets...")
        train_dataset = get_paper_dataset(global_config, tokenizer, global_config.train_split)
        val_dataset  = get_paper_dataset(global_config, tokenizer, global_config.val_split)

        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            logging_dir=os.path.join(output_dir, "logs"),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            resume_from_checkpoint=latest_checkpoint,
            **asdict(training_config)
        )
        training_args.set_dataloader(**asdict(dataloader_config))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            callbacks=[LoggingCallback(logger, wandb)]
        )

        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=latest_checkpoint)

        # Save the final model
        final_output_dir = os.path.join(output_dir, f"finetuned_model_{datetime.now().strftime('%Y-%m-%d_%H%M')}")
        os.makedirs(final_output_dir, exist_ok=True)
        model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        logger.info(f"Training completed. Model saved to {final_output_dir}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
        if global_config.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()