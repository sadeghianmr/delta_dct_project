import torch
import gc
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForImageClassification, AutoImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple

from pipeline import compress_model, decompress_model
from utils import calculate_delta_parameters, calculate_parameters_size, calculate_compressed_size
from evaluation import evaluate_accuracy

# --- Helper Functions ---

def _load_models(model_class: torch.nn.Module, base_id: str, finetuned_id: str) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Loads the pre-trained and fine-tuned models from Hugging Face."""
    print(f"Loading models: {base_id} and {finetuned_id}")
    finetuned_model = model_class.from_pretrained(finetuned_id)
    num_labels = finetuned_model.config.num_labels
    
    pretrained_model = model_class.from_pretrained(
        base_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    return pretrained_model, finetuned_model

# In src/runner.py

def _prepare_dataloader(task_info: Dict[str, Any], model_class: torch.nn.Module, base_model_id: str) -> DataLoader:
    """Prepares the appropriate dataloader for a given text or image classification task."""
    print(f"Preparing dataloader for dataset: {task_info['name']}")
    split_name = task_info.get('split', 'validation')
    eval_dataset = load_dataset(task_info['name'], task_info.get('config'), split=f"{split_name}[:200]")

    if model_class == AutoModelForImageClassification:
        processor = AutoImageProcessor.from_pretrained(base_model_id)
        def transform(examples):
            inputs = processor([img.convert("RGB") for img in examples['img']], return_tensors="pt")
            
            # --- DEBUGGING PRINT STATEMENT ---
            # This line will tell us the exact dtype of the pixel values tensor.
            print(f"DEBUG: Dtype of pixel_values from processor is {inputs['pixel_values'].dtype}")
            
            # This line ensures the tensor is float before being used.
            inputs['pixel_values'] = inputs['pixel_values'].float()
            inputs['label'] = examples['label']
            return inputs
            
        eval_dataset.set_transform(transform)
        return DataLoader(eval_dataset, batch_size=32)
    else:  # Text models
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        def tokenize(e): return tokenizer(e[task_info['text_column']], padding="max_length", truncation=True, max_length=128)
        tokenized_eval = eval_dataset.map(tokenize, batched=True)
        tokenized_eval.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return DataLoader(tokenized_eval, batch_size=32)

def _perform_evaluations(ft_model, recon_model, pt_model, compressed_data, dataloader, device):
    """Runs all evaluations (accuracy and storage) and returns a metrics dictionary."""
    print("Performing evaluations...")
    original_accuracy = evaluate_accuracy(ft_model, dataloader, device)
    reconstructed_accuracy = evaluate_accuracy(recon_model, dataloader, device)

    uncompressed_delta = calculate_delta_parameters(pt_model, ft_model)
    original_delta_size = calculate_parameters_size(uncompressed_delta)
    compressed_delta_size = calculate_compressed_size(compressed_data)
    compression_ratio = original_delta_size / compressed_delta_size if compressed_delta_size > 0 else float('inf')

    return {
        "original_accuracy": original_accuracy,
        "reconstructed_accuracy": reconstructed_accuracy,
        "accuracy_drop": original_accuracy - reconstructed_accuracy,
        "original_delta_mb": original_delta_size,
        "compressed_delta_mb": compressed_delta_size,
        "compression_ratio": compression_ratio
    }

# --- Main Runner Function ---

def run_classification_experiment(config: Dict[str, Any], device: str = "cpu") -> Dict[str, Any]:
    """Runs a complete compression and classification experiment."""
    print(f"\n{'='*20} Starting Experiment: {config['finetuned_model_id'].split('/')[-1]} {'='*20}")
    
    pt_model, ft_model = _load_models(config['model_class'], config['pretrained_model_id'], config['finetuned_model_id'])
    dataloader = _prepare_dataloader(config['task_info'], config['model_class'], config['pretrained_model_id'])

    compressed_data = compress_model(pt_model, ft_model, config['patch_size'], config['bit_strategy'])
    reconstructed_model = decompress_model(pt_model, compressed_data)
    
    metrics = _perform_evaluations(ft_model, reconstructed_model, pt_model, compressed_data, dataloader, torch.device(device))
    
    print("Releasing models from memory...")
    del pt_model, ft_model, reconstructed_model, compressed_data
    gc.collect()

    results = {
        "model_name": config['finetuned_model_id'].split('/')[-1],
        "patch_size": config['patch_size'],
        "bit_strategy": str(config['bit_strategy']),
        **metrics
    }
    
    print(f"{'='*20} Experiment Finished {'='*20}")
    return results