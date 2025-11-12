"""Deep Learning models for TOTO prediction - Phase 2."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, List


class TotoDNN(nn.Module):
    """Deep Neural Network with embeddings for TOTO prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        num_numbers: int = 49
    ):
        super(TotoDNN, self).__init__()

        self.logger = logging.getLogger(__name__)

        # Build hidden layers
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

    def forward(self, x):
        """
        Args:
            x: [batch, input_dim]

        Returns:
            [batch, 49] - logits for each number
        """
        x = self.hidden(x)
        logits = self.output(x)
        return logits


class TotoLSTM(nn.Module):
    """LSTM for sequential TOTO prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_numbers: int = 49
    ):
        super(TotoLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention (simplified)
        self.attention = nn.Linear(hidden_dim, 1)

        # Output
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_numbers)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            [batch, 49] - logits
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]

        # Simple attention
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden_dim]

        # Output
        logits = self.fc(context)
        return logits


class TotoTransformer(nn.Module):
    """Simplified Transformer for TOTO prediction."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_numbers: int = 49
    ):
        super(TotoTransformer, self).__init__()

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding (simple learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_numbers)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            [batch, 49] - logits
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer
        x = self.transformer(x)  # [batch, seq_len, d_model]

        # Global average pooling
        x = x.mean(dim=1)  # [batch, d_model]

        # Output
        logits = self.fc(x)
        return logits


class DeepLearningTrainer:
    """Trainer for deep learning models."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, X_train, y_train, batch_size=32):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        # Create batches
        indices = torch.randperm(len(X_train))

        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i + batch_size]

            X_batch = torch.FloatTensor(X_train[batch_indices]).to(self.device)
            y_batch = torch.FloatTensor(y_train[batch_indices]).to(self.device)

            # Forward pass
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def evaluate(self, X_val, y_val, batch_size=32):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        n_batches = 0

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                X_batch = torch.FloatTensor(X_val[i:i + batch_size]).to(self.device)
                y_batch = torch.FloatTensor(y_val[i:i + batch_size]).to(self.device)

                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        predictions = np.vstack(all_preds)

        # Calculate average matches
        pred_numbers = np.argsort(predictions, axis=1)[:, -6:]
        true_numbers = np.where(y_val == 1)[1].reshape(-1, 6)

        matches = []
        for pred, true in zip(pred_numbers, true_numbers):
            match_count = len(set(pred) & set(true))
            matches.append(match_count)

        avg_matches = np.mean(matches)

        return {
            'loss': avg_loss,
            'avg_matches': avg_matches
        }

    def predict_proba(self, X):
        """Predict probabilities."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits)
            return probs.cpu().numpy()

    def predict_top_k(self, X, k=6):
        """Predict top-k numbers."""
        probs = self.predict_proba(X)
        top_k_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
        return top_k_indices + 1


def train_dl_model(
    model_class,
    model_params,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    early_stopping_patience=10,
    device='cpu'
):
    """
    Train a deep learning model with early stopping.

    Returns:
        Trained model and training history
    """
    logger = logging.getLogger(__name__)

    # Initialize model
    model = model_class(**model_params)
    trainer = DeepLearningTrainer(model, learning_rate=learning_rate, device=device)

    logger.info(f"Training {model_class.__name__}...")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float('inf')
    best_val_matches = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_matches': []}

    for epoch in range(epochs):
        # Train
        train_loss = trainer.train_epoch(X_train, y_train, batch_size)

        # Validate
        val_metrics = trainer.evaluate(X_val, y_val, batch_size)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_matches'].append(val_metrics['avg_matches'])

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch+1}/{epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_matches={val_metrics['avg_matches']:.2f}"
            )

        # Early stopping based on validation matches
        if val_metrics['avg_matches'] > best_val_matches:
            best_val_matches = val_metrics['avg_matches']
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    model.load_state_dict(best_model_state)
    logger.info(f"  Best validation: matches={best_val_matches:.2f}, loss={best_val_loss:.4f}")

    return trainer, history


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create synthetic data
    X_train = np.random.randn(100, 50).astype(np.float32)
    y_train = np.random.randint(0, 2, (100, 49)).astype(np.float32)
    X_val = np.random.randn(20, 50).astype(np.float32)
    y_val = np.random.randint(0, 2, (20, 49)).astype(np.float32)

    # Test DNN
    print("Testing DNN...")
    dnn_params = {'input_dim': 50, 'hidden_dims': [128, 64], 'dropout': 0.3}
    dnn_trainer, _ = train_dl_model(
        TotoDNN, dnn_params, X_train, y_train, X_val, y_val,
        epochs=20, batch_size=16
    )

    # Test predictions
    predictions = dnn_trainer.predict_top_k(X_val[:1])
    print(f"DNN Predictions: {predictions[0]}")
