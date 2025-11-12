"""Feature engineering for TOTO ML prediction."""

import pandas as pd
import numpy as np
from typing import List
import logging


class TotoFeatureEngineer:
    """
    Transforms raw draw data into ML-ready features.
    """

    def __init__(self, lookback_windows: List[int] = None):
        """
        Initialize feature engineer.

        Args:
            lookback_windows: Windows for frequency features
        """
        self.lookback_windows = lookback_windows or [5, 10, 20]
        self.logger = logging.getLogger(__name__)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline.

        Args:
            df: DataFrame with draw data

        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature engineering...")

        df = df.copy()
        df = df.sort_values('draw_date').reset_index(drop=True)

        # Temporal features
        df = self._add_temporal_features(df)

        # Statistical features
        df = self._add_statistical_features(df)

        # Frequency features
        df = self._add_frequency_features(df)

        # Lag features
        df = self._add_lag_features(df)

        self.logger.info(f"Feature engineering complete. Shape: {df.shape}")

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df['draw_date'] = pd.to_datetime(df['draw_date'])

        # Basic temporal
        df['year'] = df['draw_date'].dt.year
        df['month'] = df['draw_date'].dt.month
        df['quarter'] = df['draw_date'].dt.quarter
        df['day_of_year'] = df['draw_date'].dt.dayofyear

        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

        # Draw sequence
        df['draw_sequence'] = range(len(df))

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features about each draw."""
        # Get winning numbers as array
        number_cols = ['number_1', 'number_2', 'number_3',
                      'number_4', 'number_5', 'number_6']
        numbers = df[number_cols].values

        # Statistics of drawn numbers
        df['numbers_mean'] = numbers.mean(axis=1)
        df['numbers_std'] = numbers.std(axis=1)
        df['numbers_min'] = numbers.min(axis=1)
        df['numbers_max'] = numbers.max(axis=1)
        df['numbers_range'] = df['numbers_max'] - df['numbers_min']
        df['numbers_sum'] = numbers.sum(axis=1)

        # Odd/even ratio
        df['odd_count'] = (numbers % 2).sum(axis=1)
        df['even_count'] = 6 - df['odd_count']
        df['odd_even_ratio'] = df['odd_count'] / 6

        # Range distribution (low/mid/high)
        df['low_count'] = (numbers <= 16).sum(axis=1)  # 1-16
        df['mid_count'] = ((numbers > 16) & (numbers <= 33)).sum(axis=1)  # 17-33
        df['high_count'] = (numbers > 33).sum(axis=1)  # 34-49

        # Consecutive numbers
        df['consecutive_count'] = self._count_consecutive(numbers)

        return df

    def _add_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add number frequency features."""
        # Create binary matrix: [n_draws x 49]
        number_matrix = self._create_number_matrix(df)

        # For each lookback window
        for window in self.lookback_windows:
            # Rolling frequency for each number
            rolling_freq = pd.DataFrame(number_matrix).rolling(
                window=window,
                min_periods=1
            ).sum()

            # Aggregate features (instead of per-number)
            df[f'freq_sum_w{window}'] = rolling_freq.sum(axis=1)
            df[f'freq_mean_w{window}'] = rolling_freq.mean(axis=1)
            df[f'freq_std_w{window}'] = rolling_freq.std(axis=1)

        # All-time frequency
        cumsum_freq = pd.DataFrame(number_matrix).cumsum()
        df['freq_alltime_sum'] = cumsum_freq.sum(axis=1)
        df['freq_alltime_mean'] = cumsum_freq.mean(axis=1)

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features from previous draws."""
        stat_cols = [
            'numbers_mean', 'numbers_std', 'numbers_sum',
            'odd_count', 'consecutive_count'
        ]

        for lag in [1, 2, 3]:
            for col in stat_cols:
                if col in df.columns:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)

        return df

    def _create_number_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create binary matrix indicating which numbers appeared in each draw.

        Returns:
            Array of shape [n_draws, 49]
        """
        n_draws = len(df)
        matrix = np.zeros((n_draws, 49), dtype=int)

        for idx, row in df.iterrows():
            winning = [row[f'number_{i}'] for i in range(1, 7)]
            for num in winning:
                matrix[idx, int(num) - 1] = 1

        return matrix

    def _count_consecutive(self, numbers: np.ndarray) -> np.ndarray:
        """Count consecutive numbers in each draw."""
        counts = []
        for row in numbers:
            sorted_nums = np.sort(row)
            count = 0
            for i in range(len(sorted_nums) - 1):
                if sorted_nums[i + 1] - sorted_nums[i] == 1:
                    count += 1
            counts.append(count)
        return np.array(counts)

    def prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create binary target vectors [n_samples, 49].
        1 if number appears in draw, 0 otherwise.

        Args:
            df: DataFrame with draw data

        Returns:
            Array of shape [n_draws, 49]
        """
        n_draws = len(df)
        targets = np.zeros((n_draws, 49), dtype=int)

        for idx, row in df.iterrows():
            winning = [row[f'number_{i}'] for i in range(1, 7)]
            for num in winning:
                targets[idx, int(num) - 1] = 1

        return targets


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from database.db_manager import DatabaseManager

    # Load data
    db = DatabaseManager('data/toto.db')
    draws = db.get_all_draws()
    df = pd.DataFrame(draws)

    print(f"Loaded {len(df)} draws")

    # Engineer features
    engineer = TotoFeatureEngineer()
    df_features = engineer.engineer_features(df)

    print(f"\nFeatures shape: {df_features.shape}")
    print(f"Feature columns: {list(df_features.columns)}")

    # Create targets
    targets = engineer.prepare_target(df)
    print(f"\nTargets shape: {targets.shape}")
