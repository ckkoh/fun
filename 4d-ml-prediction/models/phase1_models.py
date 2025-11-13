"""
Phase 1 Models for 4D Prediction

Phase 1 Methodology (inspired by TOTO P1_LightGBM):
- Direct 4D number prediction (10,000-way multi-output classification)
- Each 4D number (0000-9999) gets a binary classifier
- Simple LightGBM with default parameters
- Multi-hot encoding: Each draw's winning numbers marked as 1
- Top-K selection based on probability scores

This differs from the digit-by-digit approach (Phase 2) which predicts
each digit position separately.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict
import lightgbm as lgb


class FourDLightGBM_Phase1:
    """
    Phase 1 LightGBM for 4D prediction.
    Uses 10,000 binary classifiers (one per 4D number 0000-9999).

    This approach treats each 4D number as an atomic entity,
    rather than breaking it down into digit-level predictions.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42
    ):
        """Initialize Phase 1 LightGBM."""
        self.logger = logging.getLogger(__name__)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        # Store 10,000 individual models (one per 4D number)
        self.models = {}
        self.feature_names = None

    def _format_number(self, num: int) -> str:
        """Convert integer to 4-digit string format."""
        return f"{num:04d}"

    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str] = None):
        """
        Train the model.

        Args:
            X_train: Training features [n_samples, n_features]
            y_train: Training targets [n_samples, 10000] - multi-hot encoded
            feature_names: Feature names
        """
        self.logger.info(f"Training Phase 1 LightGBM on {X_train.shape[0]} samples...")
        self.logger.info(f"Features: {X_train.shape[1]}, Target dimensions: {y_train.shape}")
        self.feature_names = feature_names

        # Train a binary classifier for each 4D number
        # Note: This is memory-intensive, so we only train models for numbers
        # that appear in the training data to save resources

        # Find which 4D numbers appear in training data
        active_numbers = set()
        for i in range(y_train.shape[1]):
            if y_train[:, i].sum() > 0:
                active_numbers.add(i)

        self.logger.info(f"Found {len(active_numbers)} active 4D numbers in training data")

        # Train a classifier for each active number
        for num_idx in active_numbers:
            if num_idx % 1000 == 0:
                self.logger.info(f"  Training classifier {num_idx}/10000...")

            y_binary = y_train[:, num_idx]

            # Skip if this number never appears
            if y_binary.sum() == 0:
                continue

            # Create LightGBM classifier
            model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=1  # Use single thread per model to avoid overhead
            )

            # Train on this binary target
            model.fit(X_train, y_binary)

            self.models[num_idx] = model

        self.logger.info(f"Phase 1 training complete - {len(self.models)} models trained")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for all 10,000 4D numbers.

        Args:
            X: Features [n_samples, n_features]

        Returns:
            Probabilities [n_samples, 10000]
        """
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, 10000))

        # Get predictions from each trained model
        for num_idx, model in self.models.items():
            try:
                prob = model.predict_proba(X)[:, 1]  # Probability of class 1
                probs[:, num_idx] = prob
            except:
                # If prediction fails, use zero probability
                probs[:, num_idx] = 0.0

        return probs

    def predict_top_k(self, X: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Predict top-k 4D numbers with confidence scores.

        Args:
            X: Features [n_samples, n_features]
            k: Number of predictions

        Returns:
            List of (number, confidence) tuples for each sample
        """
        probs = self.predict_proba(X)

        results = []
        for i in range(X.shape[0]):
            # Get top-k indices
            top_k_indices = np.argsort(probs[i])[-k:][::-1]

            # Convert to 4D numbers with confidence scores
            predictions = [
                (self._format_number(idx), probs[i, idx])
                for idx in top_k_indices
            ]
            results.append(predictions)

        return results

    def get_number_probabilities(self, X: np.ndarray) -> Dict[str, float]:
        """
        Get probability scores for all 4D numbers.

        Args:
            X: Feature vector for prediction [1, n_features]

        Returns:
            Dictionary mapping 4D number (0000-9999) to probability score
        """
        probs = self.predict_proba(X.reshape(1, -1))[0]
        return {self._format_number(i): probs[i] for i in range(10000)}


def create_multilabel_target(df, number_cols: List[str]) -> np.ndarray:
    """
    Create multi-hot encoding for 4D numbers.

    Args:
        df: DataFrame with draw results
        number_cols: Column names containing winning 4D numbers

    Returns:
        Multi-hot encoded array [n_samples, 10000]
    """
    n_samples = len(df)
    y = np.zeros((n_samples, 10000), dtype=np.int8)

    for i, (idx, row) in enumerate(df.iterrows()):
        for col in number_cols:
            if col in row and pd.notna(row[col]):
                number_str = str(row[col]).zfill(4)
                number_int = int(number_str)
                y[i, number_int] = 1

    return y


import pandas as pd

def prepare_phase1_training_data(df: pd.DataFrame, feature_engineer):
    """
    Prepare training data for Phase 1 methodology.

    Args:
        df: DataFrame with historical draws
        feature_engineer: FourDFeatureEngineer instance

    Returns:
        X, y, feature_cols - Ready for Phase 1 training
    """
    # Engineer features
    df_featured = feature_engineer.engineer_features(df.copy())

    # Create multi-hot encoding for all 23 winning numbers per draw
    winning_cols = ['first_prize', 'second_prize', 'third_prize']
    winning_cols += [f'starter_{i}' for i in range(1, 11)]
    winning_cols += [f'consolation_{i}' for i in range(1, 11)]

    y = create_multilabel_target(df_featured, winning_cols)

    # Get feature columns (exclude metadata)
    exclude_cols = ['draw_id', 'draw_number', 'draw_date', 'day_of_week',
                    'first_prize', 'second_prize', 'third_prize']
    exclude_cols += [f'starter_{i}' for i in range(1, 11)]
    exclude_cols += [f'consolation_{i}' for i in range(1, 11)]

    # Select only numeric columns
    feature_cols = []
    for col in df_featured.columns:
        if col not in exclude_cols and df_featured[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            feature_cols.append(col)

    X = df_featured[feature_cols].fillna(0).values

    return X, y, feature_cols
