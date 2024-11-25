import os
import sys
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel

from paper_dataset import PreprocessedDataset
from logger import setup_logger  # Import the logger setup function


class LoggingCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.info(f"Training Metrics: {logs}")


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Setup logger
    logger = setup_logger(config['output']['log_file'])

    # Set visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = config['model']['device']

    # Setup directories
    output_dir = config['output']['base_dir']
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Check for existing checkpoint
        latest_checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
            if checkpoints:
                latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
                logger.info(f"Found checkpoint: {latest_checkpoint}")

        # Initialize model
        if latest_checkpoint:
            logger.info(f"Loading model from checkpoint: {latest_checkpoint}")
            model = LlamaForCausalLM.from_pretrained(config['model']['name'], device_map="auto")
            model = PeftModel.from_pretrained(model, latest_checkpoint)
        else:
            logger.info("Initializing new model...")
            model = LlamaForCausalLM.from_pretrained(config['model']['name'], device_map="auto")
            
            # Print total parameters before LoRA
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Total parameters before LoRA: {total_params:,}")

            # Apply LoRA
            lora_config = LoraConfig(
                r=config['lora']['r'],
                lora_alpha=config['lora']['alpha'],
                target_modules=config['lora']['target_modules'],
                lora_dropout=config['lora']['dropout'],
                bias=config['lora']['bias'],
                task_type=config['lora']['task_type'],
            )
            model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Percentage of parameters being trained: {100 * trainable_params / total_params:.2f}%")

        # Load datasets
        logger.info("Loading datasets...")
        train_dataset = PreprocessedDataset(config['data']['train_path'], tokenizer, config['training']['max_seq_length'])
        val_dataset = PreprocessedDataset(config['data']['val_path'], tokenizer, config['training']['max_seq_length'])

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            prefetch_factor=config['data']['prefetch_factor'],
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            prefetch_factor=config['data']['prefetch_factor'],
            pin_memory=True,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            eval_strategy="steps",
            eval_steps=config['checkpoint']['eval_steps'],
            save_strategy="steps",
            save_steps=config['checkpoint']['save_steps'],
            learning_rate=config['training']['learning_rate'],
            per_device_train_batch_size=config['training']['batch_size'],
            per_device_eval_batch_size=config['training']['batch_size'],
            num_train_epochs=config['training']['num_epochs'],
            weight_decay=config['training']['weight_decay'],
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=config['checkpoint']['logging_steps'],
            save_total_limit=config['checkpoint']['save_total_limit'],
            push_to_hub=False,
            fp16=config['training']['fp16'],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            resume_from_checkpoint=latest_checkpoint,
        )

        # Data collator
        def data_collator(batch):
            return {
                "input_ids": torch.stack([item["input_ids"] for item in batch]),
                "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
                "labels": torch.stack([item["labels"] for item in batch]),
            }

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            callbacks=[LoggingCallback(logger)],  # Add custom callback
        )

        # Train the model
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=latest_checkpoint)

        # Save the final model
        final_output_dir = os.path.join(output_dir, f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(final_output_dir, exist_ok=True)
        model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        logger.info(f"Training completed. Model saved to {final_output_dir}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

    finally:
        # Clean up old checkpoints if needed
        if os.path.exists(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
            if len(checkpoints) > config['checkpoint']['save_total_limit']:
                checkpoints.sort()
                for checkpoint in checkpoints[:-config['checkpoint']['save_total_limit']]:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                    try:
                        import shutil
                        shutil.rmtree(checkpoint_path)
                        logger.info(f"Cleaned up old checkpoint: {checkpoint}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up checkpoint {checkpoint}: {str(e)}")


if __name__ == "__main__":
    main()
