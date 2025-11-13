"""
Machine learning models for 4D digit prediction.

Each model predicts digits for the 4 positions independently.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import lightgbm as lgb


class FourDDigitPredictor:
    """Base class for 4D digit prediction models."""

    def __init__(self):
        """Initialize digit predictor."""
        self.models = {}  # One model per digit position (0-3)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def train(self, X: np.ndarray, y_dict: Dict[int, np.ndarray]):
        """
        Train models for all digit positions.

        Args:
            X: Feature matrix
            y_dict: Dictionary mapping position (0-3) to target labels
        """
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Get probability distributions for each digit position.

        Args:
            X: Feature matrix

        Returns:
            Dictionary mapping position to probability arrays (shape: [n_samples, 10])
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Predict most likely digit for each position.

        Args:
            X: Feature matrix

        Returns:
            Dictionary mapping position to predicted digits
        """
        probs = self.predict_proba(X)
        return {pos: np.argmax(probs[pos], axis=1) for pos in probs}

    def generate_4d_numbers(self, X: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Generate top-K 4D numbers with confidence scores.

        Args:
            X: Feature matrix (single sample)
            top_k: Number of predictions to return

        Returns:
            List of (4d_number, confidence) tuples
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Get probability distributions for each position
        probs = self.predict_proba(X)

        # For single prediction, combine probabilities
        results = []

        # Strategy 1: Most confident combination
        most_likely = []
        total_conf = 1.0
        for pos in range(4):
            digit_probs = probs[pos][0]  # First sample
            best_digit = np.argmax(digit_probs)
            best_prob = digit_probs[best_digit]
            most_likely.append(best_digit)
            total_conf *= best_prob

        number = ''.join(str(d) for d in most_likely)
        results.append((number, total_conf))

        # Strategy 2: Sample from probability distributions
        for _ in range(top_k - 1):
            digits = []
            conf = 1.0
            for pos in range(4):
                digit_probs = probs[pos][0]
                # Sample from distribution
                digit = np.random.choice(10, p=digit_probs)
                digits.append(digit)
                conf *= digit_probs[digit]

            number = ''.join(str(d) for d in digits)
            if number not in [r[0] for r in results]:  # Avoid duplicates
                results.append((number, conf))

        # Sort by confidence
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class FourDRandomForest(FourDDigitPredictor):
    """Random Forest classifier for 4D prediction."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def train(self, X: np.ndarray, y_dict: Dict[int, np.ndarray]):
        """Train Random Forest for each digit position."""
        self.logger.info("Training Random Forest models...")

        for pos in range(4):
            if pos not in y_dict:
                raise ValueError(f"Missing labels for position {pos}")

            self.logger.info(f"Training position {pos} (thousands/hundreds/tens/ones)...")

            # Train classifier for this position
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )

            clf.fit(X, y_dict[pos])
            self.models[pos] = clf

            self.logger.info(f"Position {pos} training complete")

        self.logger.info("All Random Forest models trained")

    def predict_proba(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """Get probability distributions for each digit position."""
        if not self.models:
            raise ValueError("Models not trained yet")

        probs = {}
        for pos in range(4):
            probs[pos] = self.models[pos].predict_proba(X)

        return probs


class FourDLightGBM(FourDDigitPredictor):
    """LightGBM classifier for 4D prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        num_leaves: int = 31
    ):
        """
        Initialize LightGBM model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            num_leaves: Maximum number of leaves
        """
        super().__init__()
        self.params = {
            'objective': 'multiclass',
            'num_class': 10,
            'metric': 'multi_logloss',
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'verbose': -1
        }

    def train(self, X: np.ndarray, y_dict: Dict[int, np.ndarray]):
        """Train LightGBM for each digit position."""
        self.logger.info("Training LightGBM models...")

        for pos in range(4):
            if pos not in y_dict:
                raise ValueError(f"Missing labels for position {pos}")

            self.logger.info(f"Training position {pos}...")

            # Train classifier for this position
            clf = lgb.LGBMClassifier(**self.params)
            clf.fit(X, y_dict[pos])
            self.models[pos] = clf

            self.logger.info(f"Position {pos} training complete")

        self.logger.info("All LightGBM models trained")

    def predict_proba(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """Get probability distributions for each digit position."""
        if not self.models:
            raise ValueError("Models not trained yet")

        probs = {}
        for pos in range(4):
            probs[pos] = self.models[pos].predict_proba(X)

        return probs


class FourDEnsemble(FourDDigitPredictor):
    """Ensemble of multiple models for 4D prediction."""

    def __init__(self, models: List[FourDDigitPredictor], weights: List[float] = None):
        """
        Initialize ensemble model.

        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
        """
        super().__init__()
        self.base_models = models

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def train(self, X: np.ndarray, y_dict: Dict[int, np.ndarray]):
        """Train all base models."""
        self.logger.info(f"Training ensemble with {len(self.base_models)} models...")

        for i, model in enumerate(self.base_models):
            self.logger.info(f"Training base model {i + 1}/{len(self.base_models)}...")
            model.train(X, y_dict)

        self.logger.info("Ensemble training complete")

    def predict_proba(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """Get weighted average of probability distributions."""
        if not self.base_models:
            raise ValueError("No base models")

        # Initialize with zeros
        ensemble_probs = {pos: np.zeros((X.shape[0], 10)) for pos in range(4)}

        # Weighted average
        for model, weight in zip(self.base_models, self.weights):
            model_probs = model.predict_proba(X)
            for pos in range(4):
                ensemble_probs[pos] += weight * model_probs[pos]

        return ensemble_probs


def calculate_digit_accuracy(y_true: Dict[int, np.ndarray], y_pred: Dict[int, np.ndarray]) -> Dict:
    """
    Calculate accuracy metrics for digit predictions.

    Args:
        y_true: Dictionary of true labels per position
        y_pred: Dictionary of predicted labels per position

    Returns:
        Dictionary with accuracy metrics
    """
    results = {}

    # Per-position accuracy
    for pos in range(4):
        accuracy = np.mean(y_true[pos] == y_pred[pos])
        results[f'position_{pos}_accuracy'] = accuracy

    # Full 4D match accuracy (all 4 digits correct)
    full_match = True
    for pos in range(4):
        full_match = full_match & (y_true[pos] == y_pred[pos])
    results['full_match_accuracy'] = np.mean(full_match)

    # Average digit accuracy
    results['avg_digit_accuracy'] = np.mean([results[f'position_{pos}_accuracy'] for pos in range(4)])

    return results


def evaluate_prize_predictions(
    predictions: List[str],
    actual_prizes: Dict[str, List[str]]
) -> Dict:
    """
    Evaluate predictions against actual prize numbers.

    Args:
        predictions: List of predicted 4D numbers
        actual_prizes: Dictionary with prize categories and numbers
            {'first': [...], 'second': [...], 'third': [...],
             'starter': [...], 'consolation': [...]}

    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        'total_predictions': len(predictions),
        'first_prize_hits': 0,
        'second_prize_hits': 0,
        'third_prize_hits': 0,
        'starter_hits': 0,
        'consolation_hits': 0,
        'total_hits': 0,
        'hit_rate': 0.0
    }

    for pred in predictions:
        hit = False

        if pred in actual_prizes.get('first', []):
            results['first_prize_hits'] += 1
            hit = True
        elif pred in actual_prizes.get('second', []):
            results['second_prize_hits'] += 1
            hit = True
        elif pred in actual_prizes.get('third', []):
            results['third_prize_hits'] += 1
            hit = True
        elif pred in actual_prizes.get('starter', []):
            results['starter_hits'] += 1
            hit = True
        elif pred in actual_prizes.get('consolation', []):
            results['consolation_hits'] += 1
            hit = True

        if hit:
            results['total_hits'] += 1

    results['hit_rate'] = results['total_hits'] / len(predictions) if predictions else 0.0

    return results
