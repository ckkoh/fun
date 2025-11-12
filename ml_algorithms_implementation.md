# TOTO ML Algorithms - Detailed Implementation Guide

## Overview
This document provides detailed implementation specifications for all 5 machine learning algorithms, including architecture, training procedures, and integration patterns.

---

## Algorithm 1: Deep Neural Network (DNN) with Embedding

### Architecture Specification

```python
# models/dnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TotoDNN(nn.Module):
    """
    Deep Neural Network with number embeddings for TOTO prediction
    """

    def __init__(
        self,
        num_numbers: int = 49,
        embedding_dim: int = 32,
        hidden_dims: list = [512, 256, 128],
        dropout: float = 0.3,
        history_length: int = 20
    ):
        super(TotoDNN, self).__init__()

        self.num_numbers = num_numbers
        self.embedding_dim = embedding_dim
        self.history_length = history_length

        # Number embeddings (1-49)
        self.number_embedding = nn.Embedding(
            num_embeddings=num_numbers + 1,  # +1 for padding
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # Input: flattened embedded history + engineered features
        input_dim = history_length * 6 * embedding_dim  # 6 numbers per draw

        # Hidden layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)

        # Output layer: 49 independent probabilities
        self.output = nn.Linear(prev_dim, num_numbers)

    def forward(self, history_numbers, features=None):
        """
        Args:
            history_numbers: [batch, history_length, 6] - past drawn numbers
            features: [batch, n_features] - engineered features (optional)

        Returns:
            [batch, 49] - probability scores for each number
        """
        batch_size = history_numbers.size(0)

        # Embed numbers: [batch, history_length, 6, embedding_dim]
        embedded = self.number_embedding(history_numbers)

        # Flatten: [batch, history_length * 6 * embedding_dim]
        embedded_flat = embedded.view(batch_size, -1)

        # Optional: concatenate with engineered features
        if features is not None:
            x = torch.cat([embedded_flat, features], dim=1)
        else:
            x = embedded_flat

        # Pass through hidden layers
        x = self.hidden(x)

        # Output logits
        logits = self.output(x)

        return logits

    def predict_top_k(self, history_numbers, k=6):
        """Predict top-k most likely numbers"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(history_numbers)
            probs = torch.sigmoid(logits)
            top_k = torch.topk(probs, k, dim=1)

        return top_k.indices + 1  # Convert to 1-49 range


class DNNTrainer:
    """Trainer for DNN model"""

    def __init__(
        self,
        model: TotoDNN,
        learning_rate: float = 0.001,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()  # Multi-label classification

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_batches = 0

        for batch in train_loader:
            history = batch['history'].to(self.device)
            target = batch['target'].to(self.device)  # [batch, 49] binary

            # Forward pass
            logits = self.model(history)
            loss = self.criterion(logits, target.float())

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        return total_loss / total_batches

    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        total_matches = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(self.device)
                target = batch['target'].to(self.device)

                logits = self.model(history)
                loss = self.criterion(logits, target.float())

                # Calculate matches (top-6 predictions)
                probs = torch.sigmoid(logits)
                pred_numbers = torch.topk(probs, 6, dim=1).indices
                true_numbers = torch.where(target == 1)[1].view(-1, 6)

                matches = self._count_matches(pred_numbers, true_numbers)
                total_matches += matches
                total_loss += loss.item()
                total_samples += target.size(0)

        return {
            'loss': total_loss / len(val_loader),
            'avg_matches': total_matches / total_samples
        }

    def _count_matches(self, pred, true):
        """Count matching numbers between predictions and truth"""
        matches = 0
        for p, t in zip(pred, true):
            matches += len(set(p.cpu().numpy()) & set(t.cpu().numpy()))
        return matches
```

---

## Algorithm 2: LSTM with Attention

### Architecture Specification

```python
# models/lstm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """Self-attention mechanism for LSTM outputs"""

    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs):
        """
        Args:
            lstm_outputs: [batch, seq_len, hidden_dim]

        Returns:
            context: [batch, hidden_dim]
            weights: [batch, seq_len]
        """
        # Calculate attention scores
        scores = self.attention(lstm_outputs)  # [batch, seq_len, 1]
        weights = F.softmax(scores, dim=1)  # [batch, seq_len, 1]

        # Weighted sum
        context = torch.sum(weights * lstm_outputs, dim=1)  # [batch, hidden_dim]

        return context, weights.squeeze(-1)


class TotoLSTM(nn.Module):
    """
    LSTM with attention for sequential TOTO prediction
    """

    def __init__(
        self,
        input_dim: int,
        lstm_hidden_dims: list = [256, 128],
        dense_dims: list = [128, 64],
        dropout: float = 0.3,
        num_numbers: int = 49
    ):
        super(TotoLSTM, self).__init__()

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        prev_dim = input_dim

        for hidden_dim in lstm_hidden_dims:
            self.lstm_layers.append(nn.LSTM(
                input_size=prev_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                dropout=dropout if len(self.lstm_layers) < len(lstm_hidden_dims) - 1 else 0
            ))
            prev_dim = hidden_dim

        # Attention
        self.attention = AttentionLayer(lstm_hidden_dims[-1])

        # Dense layers
        layers = []
        prev_dim = lstm_hidden_dims[-1]

        for dense_dim in dense_dims:
            layers.append(nn.Linear(prev_dim, dense_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dense_dim

        self.dense = nn.Sequential(*layers)

        # Output
        self.output = nn.Linear(prev_dim, num_numbers)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            [batch, 49] - probability scores
        """
        # Pass through LSTM layers
        for lstm in self.lstm_layers:
            x, (h_n, c_n) = lstm(x)

        # Apply attention
        context, attention_weights = self.attention(x)

        # Dense layers
        x = self.dense(context)

        # Output
        logits = self.output(x)

        return logits

    def forward_with_attention(self, x):
        """Forward pass returning attention weights"""
        for lstm in self.lstm_layers:
            x, _ = lstm(x)

        context, attention_weights = self.attention(x)
        x = self.dense(context)
        logits = self.output(x)

        return logits, attention_weights
```

---

## Algorithm 3: LightGBM Gradient Boosting

### Implementation Specification

```python
# models/lightgbm_model.py

import lightgbm as lgb
import numpy as np
from typing import List, Dict

class TotoLightGBM:
    """
    LightGBM-based model for TOTO prediction
    Uses 49 binary classifiers (one per number)
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20
    ):
        self.params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

        # 49 independent models (one per number)
        self.models = {}
        self.feature_importance = {}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,  # [n_samples, 49] binary matrix
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: List[str] = None
    ):
        """
        Train 49 binary classifiers

        Args:
            X_train: [n_samples, n_features]
            y_train: [n_samples, 49] - binary target matrix
        """
        print("Training 49 LightGBM models (one per number)...")

        for number in range(1, 50):
            print(f"Training model for number {number}/49", end='\r')

            # Target for this number
            y_train_num = y_train[:, number - 1]

            # Create dataset
            train_data = lgb.Dataset(
                X_train,
                label=y_train_num,
                feature_name=feature_names
            )

            eval_sets = [train_data]
            if X_val is not None and y_val is not None:
                y_val_num = y_val[:, number - 1]
                val_data = lgb.Dataset(
                    X_val,
                    label=y_val_num,
                    reference=train_data
                )
                eval_sets.append(val_data)

            # Train model
            model = lgb.train(
                self.params,
                train_data,
                valid_sets=eval_sets,
                callbacks=[lgb.early_stopping(stopping_rounds=50)]
            )

            self.models[number] = model
            self.feature_importance[number] = model.feature_importance(
                importance_type='gain'
            )

        print(f"\nTraining complete: {len(self.models)} models")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for all 49 numbers

        Returns:
            [n_samples, 49] - probability matrix
        """
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, 49))

        for number in range(1, 50):
            model = self.models[number]
            probs[:, number - 1] = model.predict(X)

        return probs

    def predict_top_k(self, X: np.ndarray, k: int = 6) -> np.ndarray:
        """
        Predict top-k numbers for each sample

        Returns:
            [n_samples, k] - predicted numbers (1-49)
        """
        probs = self.predict_proba(X)
        top_k_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
        return top_k_indices + 1  # Convert to 1-49

    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """Get aggregated feature importance across all models"""
        # Average importance across all 49 models
        avg_importance = np.mean(
            [imp for imp in self.feature_importance.values()],
            axis=0
        )

        # Get top features
        top_indices = np.argsort(avg_importance)[-top_n:][::-1]

        return {
            'indices': top_indices,
            'scores': avg_importance[top_indices]
        }
```

---

## Algorithm 4: Random Forest

### Implementation Specification

```python
# models/random_forest_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from typing import Dict

class TotoRandomForest:
    """
    Random Forest for TOTO prediction
    Uses MultiOutputClassifier for 49 simultaneous predictions
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 15,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        max_features: str = 'sqrt',
        n_jobs: int = -1
    ):
        # Base random forest
        base_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=True,
            oob_score=True,
            n_jobs=n_jobs,
            random_state=42,
            verbose=0
        )

        # Multi-output wrapper
        self.model = MultiOutputClassifier(base_rf, n_jobs=n_jobs)
        self.feature_names = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,  # [n_samples, 49]
        feature_names: list = None
    ):
        """Train the multi-output random forest"""
        self.feature_names = feature_names
        print("Training Random Forest (multi-output)...")

        self.model.fit(X_train, y_train)

        # Get OOB score (average across outputs)
        oob_scores = [est.oob_score_ for est in self.model.estimators_]
        avg_oob = np.mean(oob_scores)

        print(f"Training complete. Avg OOB score: {avg_oob:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for all 49 numbers

        Returns:
            [n_samples, 49]
        """
        # Get probabilities from each output
        probs = []
        for estimator in self.model.estimators_:
            prob = estimator.predict_proba(X)[:, 1]  # Probability of class 1
            probs.append(prob)

        return np.column_stack(probs)

    def predict_top_k(self, X: np.ndarray, k: int = 6) -> np.ndarray:
        """Predict top-k numbers"""
        probs = self.predict_proba(X)
        top_k_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
        return top_k_indices + 1

    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """Get feature importance (averaged across all outputs)"""
        # Average importance across all 49 classifiers
        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)

        avg_importance = np.mean(importances, axis=0)
        top_indices = np.argsort(avg_importance)[-top_n:][::-1]

        return {
            'features': [self.feature_names[i] if self.feature_names else f"feature_{i}"
                        for i in top_indices],
            'importance': avg_importance[top_indices]
        }
```

---

## Algorithm 5: Transformer

### Architecture Specification

```python
# models/transformer_model.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for draw sequences"""

    def __init__(self, d_model: int, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class TotoTransformer(nn.Module):
    """
    Transformer-based model for TOTO prediction
    """

    def __init__(
        self,
        num_numbers: int = 49,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super(TotoTransformer, self).__init__()

        self.d_model = d_model
        self.num_numbers = num_numbers

        # Number embedding (1-49)
        self.number_embedding = nn.Embedding(num_numbers + 1, d_model, padding_idx=0)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.fc = nn.Linear(d_model, num_numbers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.number_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        """
        Args:
            src: [batch, seq_len, 6] - sequence of draws (6 numbers each)
            src_mask: Optional mask

        Returns:
            [batch, 49] - logits for each number
        """
        batch_size, seq_len, num_per_draw = src.shape

        # Flatten to [batch, seq_len * 6]
        src_flat = src.view(batch_size, seq_len * num_per_draw)

        # Embed: [batch, seq_len * 6, d_model]
        embedded = self.number_embedding(src_flat) * math.sqrt(self.d_model)

        # Add positional encoding
        embedded = self.pos_encoder(embedded)

        # Pass through transformer
        output = self.transformer_encoder(embedded, src_mask)

        # Global average pooling over sequence
        pooled = output.mean(dim=1)  # [batch, d_model]

        # Project to 49 numbers
        logits = self.fc(pooled)  # [batch, 49]

        return logits


class TransformerTrainer:
    """Trainer for Transformer model"""

    def __init__(
        self,
        model: TotoTransformer,
        learning_rate: float = 0.0001,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device

        # Optimizer with warmup
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )

        # Focal loss (handles class imbalance)
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100
        )

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            src = batch['sequence'].to(self.device)
            target = batch['target'].to(self.device)

            # Forward
            logits = self.model(src)
            loss = self.criterion(logits, target)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / len(train_loader)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch, 49] logits
            targets: [batch, 49] binary
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_weight * bce_loss

        return focal_loss.mean()
```

---

## Model Ensemble Strategy

### Ensemble Implementation

```python
# models/ensemble.py

import numpy as np
from typing import List, Dict

class TotoEnsemble:
    """
    Ensemble of all 5 models
    """

    def __init__(self, models: Dict):
        """
        Args:
            models: Dictionary with keys ['dnn', 'lstm', 'lgb', 'rf', 'transformer']
        """
        self.models = models
        self.weights = {name: 1.0 for name in models.keys()}  # Equal weights initially

    def set_weights(self, weights: Dict[str, float]):
        """Set custom weights for each model"""
        self.weights = weights

    def predict_proba(self, X: Dict) -> np.ndarray:
        """
        Ensemble prediction (weighted average)

        Args:
            X: Dictionary with appropriate inputs for each model

        Returns:
            [n_samples, 49] - averaged probabilities
        """
        predictions = []
        total_weight = sum(self.weights.values())

        for name, model in self.models.items():
            # Get predictions from each model
            if name in ['dnn', 'lstm', 'transformer']:
                # PyTorch models
                probs = self._predict_torch_model(model, X[name])
            else:
                # Sklearn/LightGBM models
                probs = model.predict_proba(X[name])

            # Weight and add
            weighted_probs = probs * (self.weights[name] / total_weight)
            predictions.append(weighted_probs)

        # Average
        ensemble_probs = np.mean(predictions, axis=0)
        return ensemble_probs

    def predict_top_k(self, X: Dict, k: int = 6) -> np.ndarray:
        """Ensemble prediction of top-k numbers"""
        probs = self.predict_proba(X)
        top_k_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
        return top_k_indices + 1

    def predict_voting(self, X: Dict, k: int = 6) -> np.ndarray:
        """
        Voting-based ensemble: each model votes for top-k, select most voted

        Returns:
            [n_samples, 6] - final predictions
        """
        n_samples = list(X.values())[0].shape[0]
        vote_counts = np.zeros((n_samples, 49))

        for name, model in self.models.items():
            # Get top-k from each model
            if name in ['dnn', 'lstm', 'transformer']:
                top_k = self._predict_torch_top_k(model, X[name], k=10)
            else:
                top_k = model.predict_top_k(X[name], k=10)

            # Add votes
            for sample_idx in range(n_samples):
                for num in top_k[sample_idx]:
                    vote_counts[sample_idx, num - 1] += self.weights[name]

        # Select top-6 by votes
        final_predictions = np.argsort(vote_counts, axis=1)[:, -k:][:, ::-1] + 1
        return final_predictions

    def _predict_torch_model(self, model, X):
        """Helper for PyTorch models"""
        model.eval()
        with torch.no_grad():
            logits = model(X)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def _predict_torch_top_k(self, model, X, k):
        """Helper for PyTorch top-k prediction"""
        model.eval()
        with torch.no_grad():
            logits = model(X)
            probs = torch.sigmoid(logits)
            top_k = torch.topk(probs, k, dim=1).indices.cpu().numpy() + 1
        return top_k


class StackingEnsemble:
    """
    Stacking ensemble with meta-learner
    """

    def __init__(self, base_models: Dict, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def train_meta_model(self, X_train: Dict, y_train: np.ndarray):
        """Train meta-learner on base model predictions"""

        # Get predictions from all base models
        base_predictions = []

        for name, model in self.base_models.items():
            probs = self._get_model_predictions(model, X_train[name], name)
            base_predictions.append(probs)

        # Stack predictions: [n_samples, 49 * n_models]
        X_meta = np.hstack(base_predictions)

        # Train meta-model
        self.meta_model.train(X_meta, y_train)

    def predict_proba(self, X: Dict) -> np.ndarray:
        """Predict using stacked ensemble"""

        # Get base predictions
        base_predictions = []
        for name, model in self.base_models.items():
            probs = self._get_model_predictions(model, X[name], name)
            base_predictions.append(probs)

        # Stack and predict with meta-model
        X_meta = np.hstack(base_predictions)
        final_probs = self.meta_model.predict_proba(X_meta)

        return final_probs

    def _get_model_predictions(self, model, X, model_name):
        """Get predictions from a model"""
        if model_name in ['dnn', 'lstm', 'transformer']:
            return self._predict_torch_model(model, X)
        else:
            return model.predict_proba(X)
```

---

## Training Pipeline

### Complete Training Orchestration

```python
# training/train_all_models.py

import logging
from pathlib import Path
from typing import Dict
import json

class TotoModelTrainer:
    """
    Orchestrates training of all 5 models
    """

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.results = {}

    def train_all_models(self, train_data, val_data):
        """Train all 5 algorithms"""

        self.logger.info("=" * 60)
        self.logger.info("TRAINING ALL TOTO PREDICTION MODELS")
        self.logger.info("=" * 60)

        # 1. Train DNN
        self.logger.info("\n[1/5] Training Deep Neural Network...")
        self.models['dnn'] = self._train_dnn(train_data, val_data)

        # 2. Train LSTM
        self.logger.info("\n[2/5] Training LSTM with Attention...")
        self.models['lstm'] = self._train_lstm(train_data, val_data)

        # 3. Train LightGBM
        self.logger.info("\n[3/5] Training LightGBM...")
        self.models['lgb'] = self._train_lightgbm(train_data, val_data)

        # 4. Train Random Forest
        self.logger.info("\n[4/5] Training Random Forest...")
        self.models['rf'] = self._train_random_forest(train_data, val_data)

        # 5. Train Transformer
        self.logger.info("\n[5/5] Training Transformer...")
        self.models['transformer'] = self._train_transformer(train_data, val_data)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("ALL MODELS TRAINED SUCCESSFULLY")
        self.logger.info("=" * 60)

        return self.models

    def _train_dnn(self, train_data, val_data):
        """Train DNN model"""
        from models.dnn_model import TotoDNN, DNNTrainer

        model = TotoDNN(**self.config['dnn'])
        trainer = DNNTrainer(model)

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(self.config['epochs']):
            train_loss = trainer.train_epoch(train_data)
            val_metrics = trainer.evaluate(val_data)

            self.logger.info(
                f"Epoch {epoch+1}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_matches={val_metrics['avg_matches']:.2f}"
            )

            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model
                self._save_model(model, 'dnn_best.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered")
                    break

        return model

    def _train_lstm(self, train_data, val_data):
        """Train LSTM model"""
        # Similar structure to DNN training
        pass

    def _train_lightgbm(self, train_data, val_data):
        """Train LightGBM model"""
        from models.lightgbm_model import TotoLightGBM

        model = TotoLightGBM(**self.config['lightgbm'])
        model.train(
            X_train=train_data['X'],
            y_train=train_data['y'],
            X_val=val_data['X'],
            y_val=val_data['y']
        )

        return model

    def _train_random_forest(self, train_data, val_data):
        """Train Random Forest model"""
        from models.random_forest_model import TotoRandomForest

        model = TotoRandomForest(**self.config['random_forest'])
        model.train(train_data['X'], train_data['y'])

        return model

    def _train_transformer(self, train_data, val_data):
        """Train Transformer model"""
        # Similar to DNN training
        pass

    def _save_model(self, model, filename):
        """Save model checkpoint"""
        save_path = Path(self.config['model_dir']) / filename
        if hasattr(model, 'state_dict'):  # PyTorch
            torch.save(model.state_dict(), save_path)
        else:  # Sklearn
            import joblib
            joblib.dump(model, save_path)

    def save_training_report(self, output_path: str):
        """Save training summary"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
```

---

## Configuration Template

```yaml
# config/training_config.yaml

dnn:
  num_numbers: 49
  embedding_dim: 32
  hidden_dims: [512, 256, 128]
  dropout: 0.3
  history_length: 20

lstm:
  input_dim: 128
  lstm_hidden_dims: [256, 128]
  dense_dims: [128, 64]
  dropout: 0.3

lightgbm:
  n_estimators: 1000
  max_depth: 8
  learning_rate: 0.05
  num_leaves: 31
  min_child_samples: 20

random_forest:
  n_estimators: 300
  max_depth: 15
  min_samples_split: 10
  min_samples_leaf: 4
  max_features: sqrt

transformer:
  d_model: 256
  nhead: 8
  num_layers: 4
  dim_feedforward: 512
  dropout: 0.1
  max_seq_len: 100

training:
  epochs: 100
  batch_size: 32
  device: cuda
  model_dir: models/checkpoints

ensemble:
  method: weighted_average  # or 'voting' or 'stacking'
  weights:
    dnn: 1.0
    lstm: 1.0
    lgb: 1.0
    rf: 1.0
    transformer: 1.0
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-12
**Status:** Implementation Specification Complete
