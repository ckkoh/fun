"""
Feature engineering for 4D number prediction.

This module extracts temporal, digit-level, and number-level features
from historical 4D draw data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging
from datetime import datetime


class FourDFeatureEngineer:
    """Feature engineering for 4D lottery prediction."""

    def __init__(self, lookback_windows: List[int] = [5, 10, 20]):
        """
        Initialize feature engineer.

        Args:
            lookback_windows: Windows for rolling statistics
        """
        self.lookback_windows = lookback_windows
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from raw draw data.

        Args:
            df: DataFrame with 4D draw data

        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature engineering...")

        df = df.copy()

        # Convert draw_date to datetime
        df['draw_date'] = pd.to_datetime(df['draw_date'])

        # Sort by date
        df = df.sort_values('draw_date').reset_index(drop=True)

        # Add temporal features
        df = self._add_temporal_features(df)

        # Add digit-level features for each position
        df = self._add_digit_features(df)

        # Add number pattern features
        df = self._add_pattern_features(df)

        # Add prize frequency features
        df = self._add_prize_features(df)

        self.logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df['year'] = df['draw_date'].dt.year
        df['month'] = df['draw_date'].dt.month
        df['day'] = df['draw_date'].dt.day
        df['day_of_week_num'] = df['draw_date'].dt.dayofweek
        df['week_of_year'] = df['draw_date'].dt.isocalendar().week
        df['quarter'] = df['draw_date'].dt.quarter

        # Cyclical encoding for month and day
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

        # Days since epoch (useful for trend)
        epoch = pd.Timestamp('2020-01-01')
        df['days_since_epoch'] = (df['draw_date'] - epoch).dt.days

        return df

    def _extract_digits(self, number: str, position: int) -> int:
        """Extract digit at specific position from 4D number."""
        if pd.isna(number) or len(str(number)) != 4:
            return -1
        return int(str(number)[position])

    def _add_digit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add digit-level features for each position (0=thousands, 1=hundreds, 2=tens, 3=ones)."""

        # Extract digits from first prize
        for pos in range(4):
            col_name = f'first_prize_digit_{pos}'
            df[col_name] = df['first_prize'].apply(lambda x: self._extract_digits(x, pos))

        # For each position, calculate frequency features
        for pos in range(4):
            digit_col = f'first_prize_digit_{pos}'

            # Digit frequency in last N draws
            for window in self.lookback_windows:
                # Count occurrences of each digit in rolling window
                for digit in range(10):
                    df[f'digit_{pos}_freq_{digit}_last_{window}'] = (
                        df[digit_col].rolling(window=window, min_periods=1)
                        .apply(lambda x: (x == digit).sum())
                    )

            # Digit statistics in rolling windows
            for window in self.lookback_windows:
                df[f'digit_{pos}_mean_last_{window}'] = (
                    df[digit_col].rolling(window=window, min_periods=1).mean()
                )
                df[f'digit_{pos}_std_last_{window}'] = (
                    df[digit_col].rolling(window=window, min_periods=1).std()
                )

            # Days since each digit last appeared
            for digit in range(10):
                mask = df[digit_col] == digit
                last_occurrence = df.index[mask].tolist()

                days_since = []
                for idx in df.index:
                    prev_occurrences = [i for i in last_occurrence if i < idx]
                    if prev_occurrences:
                        days_since.append(idx - max(prev_occurrences))
                    else:
                        days_since.append(idx + 1)  # Haven't seen it yet

                df[f'digit_{pos}_days_since_{digit}'] = days_since

        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add number pattern features."""

        def analyze_number_pattern(number: str) -> Dict:
            """Analyze patterns in a 4D number."""
            if pd.isna(number) or len(str(number)) != 4:
                return {
                    'sum': 0,
                    'is_same_digit': 0,
                    'is_sequential': 0,
                    'is_palindrome': 0,
                    'num_even': 0,
                    'num_odd': 0,
                    'has_zero': 0,
                    'has_eight': 0,
                    'has_nine': 0
                }

            digits = [int(d) for d in str(number)]

            return {
                'sum': sum(digits),
                'is_same_digit': 1 if len(set(digits)) == 1 else 0,
                'is_sequential': 1 if digits == list(range(digits[0], digits[0] + 4)) or digits == list(range(digits[0], digits[0] - 4, -1)) else 0,
                'is_palindrome': 1 if str(number) == str(number)[::-1] else 0,
                'num_even': sum(1 for d in digits if d % 2 == 0),
                'num_odd': sum(1 for d in digits if d % 2 == 1),
                'has_zero': 1 if 0 in digits else 0,
                'has_eight': 1 if 8 in digits else 0,
                'has_nine': 1 if 9 in digits else 0
            }

        # Analyze first prize patterns
        patterns = df['first_prize'].apply(analyze_number_pattern)
        pattern_df = pd.DataFrame(patterns.tolist())

        # Add pattern features
        for col in pattern_df.columns:
            df[f'first_prize_{col}'] = pattern_df[col]

        # Rolling statistics of patterns
        for window in [5, 10, 20]:
            df[f'avg_sum_last_{window}'] = df['first_prize_sum'].rolling(window=window, min_periods=1).mean()
            df[f'avg_even_last_{window}'] = df['first_prize_num_even'].rolling(window=window, min_periods=1).mean()
            df[f'palindrome_count_last_{window}'] = df['first_prize_is_palindrome'].rolling(window=window, min_periods=1).sum()

        return df

    def _add_prize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on prize categories."""

        # Count total unique numbers per draw (should be 23)
        def count_unique_numbers(row):
            numbers = [row['first_prize'], row['second_prize'], row['third_prize']]
            for i in range(1, 11):
                for prefix in ['starter', 'consolation']:
                    key = f'{prefix}_{i}'
                    if key in row and pd.notna(row[key]):
                        numbers.append(row[key])
            return len(set(numbers))

        df['total_unique_numbers'] = df.apply(count_unique_numbers, axis=1)

        # Prize category digit distribution
        def get_all_prize_digits(row, position):
            """Get all digits at a position across all prizes."""
            digits = []
            # Top 3
            for prize in ['first_prize', 'second_prize', 'third_prize']:
                if prize in row and pd.notna(row[prize]):
                    digits.append(self._extract_digits(row[prize], position))
            # Starters and consolations
            for i in range(1, 11):
                for prefix in ['starter', 'consolation']:
                    key = f'{prefix}_{i}'
                    if key in row and pd.notna(row[key]):
                        digits.append(self._extract_digits(row[key], position))
            return [d for d in digits if d != -1]

        # For each position, calculate distribution across all 23 numbers
        for pos in range(4):
            df[f'pos_{pos}_all_digits'] = df.apply(lambda row: get_all_prize_digits(row, pos), axis=1)

            # Average digit value across all prizes
            df[f'pos_{pos}_avg_all_prizes'] = df[f'pos_{pos}_all_digits'].apply(
                lambda x: np.mean(x) if len(x) > 0 else 5.0
            )

            # Most common digit across all prizes
            df[f'pos_{pos}_mode_all_prizes'] = df[f'pos_{pos}_all_digits'].apply(
                lambda x: max(set(x), key=x.count) if len(x) > 0 else 5
            )

        # Clean up temporary columns
        for pos in range(4):
            df = df.drop(columns=[f'pos_{pos}_all_digits'])

        return df

    def prepare_training_data(self, df: pd.DataFrame, target_position: int) -> tuple:
        """
        Prepare training data for a specific digit position.

        Args:
            df: DataFrame with features
            target_position: Which digit position to predict (0-3)

        Returns:
            Tuple of (X, y, feature_columns)
        """
        # Get target column
        target_col = f'first_prize_digit_{target_position}'

        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")

        # Define feature columns to use
        exclude_patterns = [
            'draw_id', 'draw_number', 'draw_date', 'day_of_week',
            'first_prize', 'second_prize', 'third_prize',
            'starter_', 'consolation_',
            'draw_type', 'created_at',
            'first_prize_digit_'  # Exclude all digit columns (we'll only use features)
        ]

        feature_cols = []
        for col in df.columns:
            # Skip if column matches any exclude pattern
            if any(pattern in col for pattern in exclude_patterns):
                continue
            # Only include numeric columns
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                feature_cols.append(col)

        # Get features and target
        X = df[feature_cols].fillna(0).values
        y = df[target_col].values

        self.logger.info(f"Prepared training data for position {target_position}: X shape {X.shape}, y shape {y.shape}")
        return X, y, feature_cols
