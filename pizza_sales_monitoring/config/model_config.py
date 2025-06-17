MODEL_CONFIGS = {"stable": {"path": "models/stable/weights/best.pt", "precision": 0.988, "recall": 0.545, "map50": 0.682, "map50_95": 0.6, "description": "Initial stable model with high precision, WARNING: This version will have object detetion flickering"}, "v2_stable": {"path": "models/v2_stable/weights/best.pt", "precision": 0.895, "recall": 0.667, "map50": 0.831, "map50_95": 0.579, "description": "Fine-tuned model with better recall"}}

DEFAULT_MODEL = "v2_stable"  # Or "stable"
