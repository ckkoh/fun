"""Simplified ML models for TOTO prediction."""

import numpy as np
import logging
from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import lightgbm as lgb


class TotoRandomForestSimple:
    """
    Simplified Random Forest for TOTO prediction.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ):
        """Initialize Random Forest."""
        self.logger = logging.getLogger(__name__)

        # Base random forest
        base_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )

        # Multi-output wrapper
        self.model = MultiOutputClassifier(base_rf, n_jobs=-1)
        self.feature_names = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str] = None):
        """
        Train the model.

        Args:
            X_train: Training features [n_samples, n_features]
            y_train: Training targets [n_samples, 49]
            feature_names: Feature names
        """
        self.logger.info(f"Training Random Forest on {X_train.shape[0]} samples...")
        self.feature_names = feature_names

        self.model.fit(X_train, y_train)

        self.logger.info("Random Forest training complete")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for all 49 numbers.

        Args:
            X: Features [n_samples, n_features]

        Returns:
            Probabilities [n_samples, 49]
        """
        # Get probabilities from each output
        probs = []
        for estimator in self.model.estimators_:
            prob = estimator.predict_proba(X)[:, 1]  # Probability of class 1
            probs.append(prob)

        return np.column_stack(probs)

    def predict_top_k(self, X: np.ndarray, k: int = 6) -> np.ndarray:
        """
        Predict top-k numbers.

        Args:
            X: Features [n_samples, n_features]
            k: Number of predictions

        Returns:
            Top-k numbers [n_samples, k]
        """
        probs = self.predict_proba(X)
        top_k_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
        return top_k_indices + 1  # Convert to 1-49


class TotoLightGBMSimple:
    """
    Simplified LightGBM for TOTO prediction.
    Uses 49 binary classifiers (one per number).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42
    ):
        """Initialize LightGBM."""
        self.logger = logging.getLogger(__name__)

        self.params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': 31,
            'min_child_samples': 10,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state
        }

        # 49 independent models (one per number)
        self.models = {}

    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str] = None):
        """
        Train 49 binary classifiers.

        Args:
            X_train: Training features [n_samples, n_features]
            y_train: Training targets [n_samples, 49]
            feature_names: Feature names
        """
        self.logger.info(f"Training 49 LightGBM models on {X_train.shape[0]} samples...")

        for number in range(1, 50):
            if number % 10 == 0:
                self.logger.info(f"Training model {number}/49...")

            # Target for this number
            y_train_num = y_train[:, number - 1]

            # Skip if no positive samples
            if y_train_num.sum() == 0:
                self.logger.warning(f"No positive samples for number {number}, skipping")
                continue

            # Create dataset
            train_data = lgb.Dataset(
                X_train,
                label=y_train_num,
                feature_name=feature_names
            )

            # Train model
            model = lgb.train(
                self.params,
                train_data,
                valid_sets=[train_data],
                callbacks=[lgb.log_evaluation(period=0)]
            )

            self.models[number] = model

        self.logger.info(f"LightGBM training complete: {len(self.models)} models trained")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for all 49 numbers.

        Args:
            X: Features [n_samples, n_features]

        Returns:
            Probabilities [n_samples, 49]
        """
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, 49))

        for number in range(1, 50):
            if number in self.models:
                model = self.models[number]
                probs[:, number - 1] = model.predict(X)

        return probs

    def predict_top_k(self, X: np.ndarray, k: int = 6) -> np.ndarray:
        """
        Predict top-k numbers.

        Args:
            X: Features [n_samples, n_features]
            k: Number of predictions

        Returns:
            Top-k numbers [n_samples, k]
        """
        probs = self.predict_proba(X)
        top_k_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
        return top_k_indices + 1  # Convert to 1-49


class SimpleEnsemble:
    """Simple ensemble of models."""

    def __init__(self, models: List):
        """
        Initialize ensemble.

        Args:
            models: List of trained models
        """
        self.models = models
        self.logger = logging.getLogger(__name__)

    def predict_top_k(self, X: np.ndarray, k: int = 6) -> np.ndarray:
        """
        Ensemble prediction using average probabilities.

        Args:
            X: Features [n_samples, n_features]
            k: Number of predictions

        Returns:
            Top-k numbers [n_samples, k]
        """
        # Get predictions from all models
        all_probs = []
        for model in self.models:
            probs = model.predict_proba(X)
            all_probs.append(probs)

        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)

        # Select top-k
        top_k_indices = np.argsort(avg_probs, axis=1)[:, -k:][:, ::-1]
        return top_k_indices + 1


def calculate_matches(predicted: np.ndarray, actual: np.ndarray) -> Tuple[int, List[int]]:
    """
    Calculate matches between predicted and actual numbers.

    Args:
        predicted: Predicted numbers [k]
        actual: Actual winning numbers [6]

    Returns:
        Tuple of (match_count, matching_numbers)
    """
    pred_set = set(predicted)
    actual_set = set(actual)
    matches = pred_set & actual_set

    return len(matches), sorted(list(matches))


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Create synthetic data for testing
    np.random.seed(42)
    X_train = np.random.randn(100, 20)
    y_train = np.random.randint(0, 2, (100, 49))

    # Train Random Forest
    print("Training Random Forest...")
    rf_model = TotoRandomForestSimple(n_estimators=50)
    rf_model.train(X_train, y_train)

    # Train LightGBM
    print("\nTraining LightGBM...")
    lgb_model = TotoLightGBMSimple(n_estimators=50)
    lgb_model.train(X_train, y_train)

    # Make predictions
    X_test = np.random.randn(1, 20)

    rf_pred = rf_model.predict_top_k(X_test, k=6)
    lgb_pred = lgb_model.predict_top_k(X_test, k=6)

    print(f"\nRandom Forest predictions: {rf_pred[0]}")
    print(f"LightGBM predictions: {lgb_pred[0]}")

    # Ensemble
    print("\nCreating ensemble...")
    ensemble = SimpleEnsemble([rf_model, lgb_model])
    ensemble_pred = ensemble.predict_top_k(X_test, k=6)
    print(f"Ensemble predictions: {ensemble_pred[0]}")
