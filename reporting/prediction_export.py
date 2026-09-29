"""Consistent score exports without changing the existing argmax/majority decision."""
import numpy as np


def probabilities_from_logits(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Ожидается непустая конечная матрица logits с >=2 классами")
    scores = np.exp(values - values.max(axis=1, keepdims=True))
    return scores / scores.sum(axis=1, keepdims=True)


def add_segment_scores(frame, logits, probabilities):
    if len(frame) != len(logits) or probabilities.shape != logits.shape:
        raise ValueError("Размеры метаданных и выходов модели различаются")
    result = frame.copy()
    for index in range(logits.shape[1]):
        result[f"logit_{index}"] = logits[:, index]
        result[f"probability_{index}"] = probabilities[:, index]
    return result


def aggregate_scores(logits, probabilities, predictions):
    counts = np.bincount(predictions, minlength=logits.shape[1])
    result = {}
    for index in range(logits.shape[1]):
        result[f"mean_logit_{index}"] = float(logits[:, index].mean())
        result[f"mean_probability_{index}"] = float(probabilities[:, index].mean())
        result[f"vote_share_{index}"] = float(counts[index] / len(predictions))
    return result


def export_schema(classes):
    return {"version": 2, "class_index": {str(i): name for i, name in enumerate(classes)},
            "segment_scores": "logit_i and softmax probability_i in class_index order",
            "record_scores": "arithmetic means of segment scores; vote_share_i is a vote fraction",
            "record_decision": "majority of segment argmax; smallest class index on ties",
            "probability_note": "softmax scores; empirical calibration is not asserted"}
