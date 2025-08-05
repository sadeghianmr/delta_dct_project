import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from tqdm import tqdm

def evaluate_accuracy(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """
    Evaluates the classification accuracy of a model on a given dataloader.
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Accuracy"):
            labels = batch['label'].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor) and k != 'label'}
            
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    return accuracy