import torch
import numpy as np
from sklearn.metrics import roc_curve

def compute_face_similarity(face_model, img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute cosine similarity between face embeddings."""
    with torch.no_grad():
        emb1 = face_model(img1)
        emb2 = face_model(img2)

        if isinstance(emb1, tuple):
            emb1 = emb1[0]
        if isinstance(emb2, tuple):
            emb2 = emb2[0]

        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

        similarity = (emb1 * emb2).sum(dim=1).mean().item()

    return similarity


def find_optimal_threshold(similarities: np.ndarray, labels: np.ndarray) -> float:
    """Find optimal threshold that maximizes accuracy."""
    thresholds = np.arange(-1.0, 1.0, 0.01)
    best_acc = 0.0
    best_thresh = 0.5

    for thresh in thresholds:
        predictions = (similarities >= thresh).astype(int)
        acc = np.mean(predictions == labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    return best_thresh


def evaluate_verification(similarities: list, labels: list, threshold: float = None) -> dict:
    """Evaluate face verification performance."""
    similarities = np.array(similarities)
    labels = np.array(labels)

    # Find optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(similarities, labels)

    predictions = (similarities >= threshold).astype(int)

    true_positives = np.sum((predictions == 1) & (labels == 1))
    true_negatives = np.sum((predictions == 0) & (labels == 0))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))

    accuracy = (true_positives + true_negatives) / len(labels)
    tar = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    far = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

    return {
        'accuracy': float(accuracy),
        'tar': float(tar),
        'far': float(far),
        'threshold': float(threshold),
        'true_positives': int(true_positives),
        'true_negatives': int(true_negatives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'total_pairs': len(labels)
    }


def compute_tar_at_far(labels, similarities, target_far):
    """Compute TAR at specific FAR using interpolation."""
    fpr, tpr, _ = roc_curve(labels, similarities)
    
    if target_far <= fpr[-1]:
        return float(np.interp(target_far, fpr, tpr))
    else:
        return float(tpr[-1])
