"""Training, retrieval evaluation, and checkpoints."""

from aerial_geoloc.engine.checkpoint import load_checkpoint, model_from_checkpoint, save_checkpoint
from aerial_geoloc.engine.evaluator import run_eval
from aerial_geoloc.engine.metrics import evaluate_ranks, mean_average_precision, recall_at_k
from aerial_geoloc.engine.retrieval import cosine_scores, evaluate_model, retrieve
from aerial_geoloc.engine.trainer import resolve_device, run_training

__all__ = [
    "cosine_scores",
    "evaluate_model",
    "evaluate_ranks",
    "load_checkpoint",
    "mean_average_precision",
    "model_from_checkpoint",
    "recall_at_k",
    "resolve_device",
    "retrieve",
    "run_eval",
    "run_training",
    "save_checkpoint",
]
