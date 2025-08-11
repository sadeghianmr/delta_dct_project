# src/finetuner.py
import torch
import evaluate
import numpy as np
from datasets import load_dataset, get_dataset_split_names
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from typing import Dict, Any

def fine_tune_model(config: Dict[str, Any]):
    """
    A general-purpose function to fine-tune a model on a classification task.
    """
    print(f"--- Starting Fine-Tuning Job ---")
    print(f"Base Model: {config['base_model_id']}")
    print(f"Dataset: {config['dataset_name']}")

    # ... (Dataset loading and preprocessing logic remains the same) ...
    try:
        splits = get_dataset_split_names(config['dataset_name'], config.get('dataset_config'))
        print(f"Dataset found. Available splits: {splits}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    dataset = load_dataset(config['dataset_name'], config.get('dataset_config'))
    is_image_model = "image" in config['model_class'].__name__.lower()
    processor, tokenizer = None, None
    if is_image_model:
        processor = AutoImageProcessor.from_pretrained(config['base_model_id'])
        def transform(examples):
            inputs = processor([img.convert("RGB") for img in examples[config['image_column']]], return_tensors="pt")
            inputs['labels'] = examples[config['label_column']]
            return inputs
        dataset = dataset.with_transform(transform)
        data_collator = DefaultDataCollator()
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['base_model_id'])
        def tokenize(examples):
            return tokenizer(examples[config['text_column']], padding="max_length", truncation=True)
        dataset = dataset.map(tokenize, batched=True).rename_column(config['label_column'], 'labels')
        data_collator = None
    train_dataset = dataset['train']
    eval_dataset = dataset[config['validation_split']]
    num_labels = train_dataset.features['labels'].num_classes

    # --- Model Loading remains the same ---
    model = config['model_class'].from_pretrained(config['base_model_id'], num_labels=num_labels)

    # --- Trainer Configuration ---
    accuracy_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    # --- SIMPLIFIED TrainingArguments ---
    # We have removed all the problematic, non-essential arguments.
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config.get('num_epochs', 3),
        per_device_train_batch_size=config.get('batch_size', 16),
        per_device_eval_batch_size=config.get('batch_size', 16),
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- Start Fine-Tuning ---
    print("\nStarting training...")
    trainer.train()

    # --- Save the Fine-Tuned Model ---
    print(f"Training complete. Saving model to {config['output_dir']}")
    trainer.save_model(config['output_dir'])
    if tokenizer:
        tokenizer.save_pretrained(config['output_dir'])
    if processor:
        processor.save_pretrained(config['output_dir'])

    print("--- Fine-Tuning Job Finished ---")