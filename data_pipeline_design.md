# TOTO Data Collection & Preprocessing Pipeline Design

## Overview
This document details the complete data pipeline for collecting, validating, and preprocessing Singapore Pools TOTO historical data for machine learning model training.

---

## 1. Data Collection Architecture

### 1.1 Web Scraping Strategy

**Target URL Pattern:**
```
Base: https://www.singaporepools.com.sg/en/product/Pages/toto_results.aspx
Historical: Access via draw number or date range queries
```

**Scraping Approach:**
```python
# Pseudo-architecture
class TotoDataScraper:
    """
    Responsible for fetching raw data from Singapore Pools
    """

    def __init__(self, base_url, rate_limit=2.0):
        self.base_url = base_url
        self.rate_limit = rate_limit  # seconds between requests
        self.session = requests.Session()

    def fetch_draw_results(self, draw_number=None, date=None):
        """Fetch single draw results"""
        pass

    def fetch_date_range(self, start_date, end_date):
        """Fetch multiple draws in date range"""
        pass

    def parse_draw_page(self, html_content):
        """Extract structured data from HTML"""
        pass
```

**Rate Limiting:**
- Minimum 2 seconds between requests
- Exponential backoff on errors
- Respect robots.txt
- Use session pooling

### 1.2 Data Schema

**Database: SQLite/PostgreSQL**

```sql
CREATE TABLE toto_draws (
    draw_id INTEGER PRIMARY KEY,
    draw_number INTEGER UNIQUE NOT NULL,
    draw_date DATE NOT NULL,
    day_of_week VARCHAR(10) NOT NULL,

    -- Winning numbers (sorted ascending)
    number_1 INTEGER NOT NULL CHECK (number_1 BETWEEN 1 AND 49),
    number_2 INTEGER NOT NULL CHECK (number_2 BETWEEN 1 AND 49),
    number_3 INTEGER NOT NULL CHECK (number_3 BETWEEN 1 AND 49),
    number_4 INTEGER NOT NULL CHECK (number_4 BETWEEN 1 AND 49),
    number_5 INTEGER NOT NULL CHECK (number_5 BETWEEN 1 AND 49),
    number_6 INTEGER NOT NULL CHECK (number_6 BETWEEN 1 AND 49),
    additional_number INTEGER NOT NULL CHECK (additional_number BETWEEN 1 AND 49),

    -- Prize information
    prize_pool DECIMAL(12, 2),

    -- Winners per group
    group_1_winners INTEGER DEFAULT 0,
    group_2_winners INTEGER DEFAULT 0,
    group_3_winners INTEGER DEFAULT 0,
    group_4_winners INTEGER DEFAULT 0,

    -- Draw metadata
    draw_type VARCHAR(20) DEFAULT 'normal', -- normal/cascade/hongbao/special
    is_rollover BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT unique_numbers CHECK (
        number_1 < number_2 AND
        number_2 < number_3 AND
        number_3 < number_4 AND
        number_4 < number_5 AND
        number_5 < number_6
    ),
    CONSTRAINT additional_not_in_main CHECK (
        additional_number NOT IN (number_1, number_2, number_3,
                                  number_4, number_5, number_6)
    ),
    CONSTRAINT valid_day CHECK (
        day_of_week IN ('Monday', 'Thursday')
    )
);

-- Indexes for efficient querying
CREATE INDEX idx_draw_date ON toto_draws(draw_date DESC);
CREATE INDEX idx_draw_number ON toto_draws(draw_number DESC);
CREATE INDEX idx_draw_type ON toto_draws(draw_type);

-- Table for individual number history (denormalized for fast queries)
CREATE TABLE number_history (
    id SERIAL PRIMARY KEY,
    draw_id INTEGER REFERENCES toto_draws(draw_id),
    number INTEGER NOT NULL CHECK (number BETWEEN 1 AND 49),
    is_additional BOOLEAN DEFAULT FALSE,
    position INTEGER CHECK (position BETWEEN 1 AND 6), -- NULL if additional

    UNIQUE(draw_id, number)
);

CREATE INDEX idx_number_history_number ON number_history(number, draw_id DESC);
```

### 1.3 Data Collection Pipeline

```python
# pipeline/data_collector.py

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

class TotoDataCollector:
    """
    Orchestrates data collection, validation, and storage
    """

    def __init__(self, scraper, database, validator):
        self.scraper = scraper
        self.db = database
        self.validator = validator
        self.logger = logging.getLogger(__name__)

    def collect_historical_data(self, start_date: str, end_date: str):
        """
        Collect all draws between date range

        Args:
            start_date: ISO format 'YYYY-MM-DD'
            end_date: ISO format 'YYYY-MM-DD'
        """
        draws_to_fetch = self._generate_draw_dates(start_date, end_date)
        self.logger.info(f"Fetching {len(draws_to_fetch)} potential draws")

        successful = 0
        failed = 0

        for draw_date in draws_to_fetch:
            try:
                # Fetch raw data
                raw_data = self.scraper.fetch_draw_by_date(draw_date)

                if raw_data is None:
                    self.logger.warning(f"No draw found for {draw_date}")
                    continue

                # Validate data
                if not self.validator.validate(raw_data):
                    self.logger.error(f"Validation failed for {draw_date}")
                    failed += 1
                    continue

                # Transform to database format
                db_record = self._transform_to_db_format(raw_data)

                # Store in database
                self.db.insert_draw(db_record)
                successful += 1

                self.logger.info(f"Successfully stored draw for {draw_date}")

                # Rate limiting
                time.sleep(self.scraper.rate_limit)

            except Exception as e:
                self.logger.error(f"Error processing {draw_date}: {str(e)}")
                failed += 1

        self.logger.info(f"Collection complete: {successful} successful, {failed} failed")
        return successful, failed

    def _generate_draw_dates(self, start_date: str, end_date: str) -> List[str]:
        """Generate all Monday and Thursday dates in range"""
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        dates = []
        current = start

        while current <= end:
            # 0=Monday, 3=Thursday
            if current.weekday() in [0, 3]:
                dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)

        return dates

    def _transform_to_db_format(self, raw_data: Dict) -> Dict:
        """Transform scraped data to database schema"""
        numbers = sorted(raw_data['winning_numbers'])

        return {
            'draw_number': raw_data['draw_number'],
            'draw_date': raw_data['draw_date'],
            'day_of_week': raw_data['day_of_week'],
            'number_1': numbers[0],
            'number_2': numbers[1],
            'number_3': numbers[2],
            'number_4': numbers[3],
            'number_5': numbers[4],
            'number_6': numbers[5],
            'additional_number': raw_data['additional_number'],
            'prize_pool': raw_data.get('prize_pool'),
            'group_1_winners': raw_data.get('group_1_winners', 0),
            'group_2_winners': raw_data.get('group_2_winners', 0),
            'group_3_winners': raw_data.get('group_3_winners', 0),
            'group_4_winners': raw_data.get('group_4_winners', 0),
            'draw_type': raw_data.get('draw_type', 'normal'),
            'is_rollover': raw_data.get('is_rollover', False),
        }

    def update_latest_draws(self, lookback_days: int = 7):
        """Fetch and update most recent draws"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        return self.collect_historical_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
```

---

## 2. Data Validation

### 2.1 Validation Rules

```python
# pipeline/data_validator.py

from typing import Dict, List, Tuple
import logging

class TotoDataValidator:
    """
    Validates scraped TOTO data for correctness
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules = [
            self._validate_numbers_range,
            self._validate_numbers_unique,
            self._validate_additional_number,
            self._validate_date_format,
            self._validate_day_of_week,
            self._validate_draw_number,
            self._validate_prize_data,
        ]

    def validate(self, data: Dict) -> bool:
        """
        Run all validation rules

        Returns:
            True if all validations pass, False otherwise
        """
        for rule in self.validation_rules:
            is_valid, error_msg = rule(data)
            if not is_valid:
                self.logger.error(f"Validation failed: {error_msg}")
                return False

        return True

    def _validate_numbers_range(self, data: Dict) -> Tuple[bool, str]:
        """All numbers must be between 1 and 49"""
        numbers = data['winning_numbers'] + [data['additional_number']]

        for num in numbers:
            if not (1 <= num <= 49):
                return False, f"Number {num} out of range [1, 49]"

        return True, ""

    def _validate_numbers_unique(self, data: Dict) -> Tuple[bool, str]:
        """6 winning numbers must be unique"""
        numbers = data['winning_numbers']

        if len(numbers) != 6:
            return False, f"Expected 6 numbers, got {len(numbers)}"

        if len(set(numbers)) != 6:
            return False, "Winning numbers are not unique"

        return True, ""

    def _validate_additional_number(self, data: Dict) -> Tuple[bool, str]:
        """Additional number must not be in winning numbers"""
        additional = data['additional_number']
        winning = data['winning_numbers']

        if additional in winning:
            return False, f"Additional number {additional} appears in winning numbers"

        return True, ""

    def _validate_date_format(self, data: Dict) -> Tuple[bool, str]:
        """Date must be valid ISO format"""
        try:
            date_str = data['draw_date']
            datetime.fromisoformat(date_str)
            return True, ""
        except ValueError as e:
            return False, f"Invalid date format: {str(e)}"

    def _validate_day_of_week(self, data: Dict) -> Tuple[bool, str]:
        """Draw must be on Monday or Thursday"""
        day = data['day_of_week']

        if day not in ['Monday', 'Thursday']:
            return False, f"Invalid day of week: {day}"

        # Cross-check with date
        date_obj = datetime.fromisoformat(data['draw_date'])
        actual_day = date_obj.strftime('%A')

        if day != actual_day:
            return False, f"Day mismatch: stated {day}, actual {actual_day}"

        return True, ""

    def _validate_draw_number(self, data: Dict) -> Tuple[bool, str]:
        """Draw number must be positive integer"""
        draw_num = data.get('draw_number')

        if draw_num is None:
            return False, "Missing draw_number"

        if not isinstance(draw_num, int) or draw_num <= 0:
            return False, f"Invalid draw_number: {draw_num}"

        return True, ""

    def _validate_prize_data(self, data: Dict) -> Tuple[bool, str]:
        """Prize pool and winners should be non-negative"""
        prize_pool = data.get('prize_pool')

        if prize_pool is not None and prize_pool < 0:
            return False, f"Invalid prize_pool: {prize_pool}"

        # Check winner counts
        for group in range(1, 5):
            key = f'group_{group}_winners'
            winners = data.get(key, 0)
            if winners < 0:
                return False, f"Invalid {key}: {winners}"

        return True, ""
```

---

## 3. Data Preprocessing

### 3.1 Feature Engineering Pipeline

```python
# pipeline/feature_engineer.py

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime

class TotoFeatureEngineer:
    """
    Transforms raw draw data into ML-ready features
    """

    def __init__(self, lookback_windows: List[int] = [5, 10, 20, 50, 100]):
        self.lookback_windows = lookback_windows

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline

        Args:
            df: DataFrame with columns [draw_date, number_1-6, additional_number]

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        df = df.sort_values('draw_date').reset_index(drop=True)

        # Temporal features
        df = self._add_temporal_features(df)

        # Number frequency features
        df = self._add_frequency_features(df)

        # Statistical features
        df = self._add_statistical_features(df)

        # Pattern features
        df = self._add_pattern_features(df)

        # Lag features
        df = self._add_lag_features(df)

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
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

        # Days since epoch (for trends)
        epoch = df['draw_date'].min()
        df['days_since_start'] = (df['draw_date'] - epoch).dt.days

        return df

    def _add_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add number frequency features for each of 49 numbers"""

        # Create binary matrix: [n_draws x 49]
        number_matrix = self._create_number_matrix(df)

        # For each lookback window
        for window in self.lookback_windows:
            # Rolling frequency for each number
            rolling_freq = pd.DataFrame(number_matrix).rolling(
                window=window,
                min_periods=1
            ).sum()

            # Add as features
            for num in range(1, 50):
                df[f'freq_n{num}_w{window}'] = rolling_freq[num - 1]

        # All-time frequency
        cumsum_freq = pd.DataFrame(number_matrix).cumsum()
        for num in range(1, 50):
            df[f'freq_n{num}_alltime'] = cumsum_freq[num - 1]

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features about each draw"""

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

        # Skewness and kurtosis (using scipy)
        from scipy.stats import skew, kurtosis
        df['numbers_skew'] = [skew(row) for row in numbers]
        df['numbers_kurtosis'] = [kurtosis(row) for row in numbers]

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

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern-based features"""

        # Gap analysis: draws since each number last appeared
        for num in range(1, 50):
            gaps = []
            last_seen = -1

            for idx, row in df.iterrows():
                winning = [row[f'number_{i}'] for i in range(1, 7)]
                winning.append(row['additional_number'])

                if num in winning:
                    gaps.append(idx - last_seen)
                    last_seen = idx
                else:
                    gaps.append(idx - last_seen if last_seen >= 0 else 0)

            df[f'gap_n{num}'] = gaps

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features from previous draws"""

        stat_cols = [
            'numbers_mean', 'numbers_std', 'numbers_sum',
            'odd_count', 'consecutive_count'
        ]

        for lag in [1, 2, 3, 5, 10]:
            for col in stat_cols:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

        return df

    def _create_number_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create binary matrix indicating which numbers appeared in each draw

        Returns:
            Array of shape [n_draws, 49]
        """
        n_draws = len(df)
        matrix = np.zeros((n_draws, 49), dtype=int)

        for idx, row in df.iterrows():
            winning = [row[f'number_{i}'] for i in range(1, 7)]
            for num in winning:
                matrix[idx, num - 1] = 1

        return matrix

    def _count_consecutive(self, numbers: np.ndarray) -> np.ndarray:
        """Count consecutive numbers in each draw"""
        counts = []
        for row in numbers:
            sorted_nums = np.sort(row)
            count = 0
            for i in range(len(sorted_nums) - 1):
                if sorted_nums[i + 1] - sorted_nums[i] == 1:
                    count += 1
            counts.append(count)
        return np.array(counts)
```

### 3.2 Data Loader for ML Models

```python
# pipeline/data_loader.py

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, List

class TotoDataLoader:
    """
    Prepares data for ML model training
    """

    def __init__(self, feature_engineer: TotoFeatureEngineer):
        self.feature_engineer = feature_engineer

    def load_and_prepare(
        self,
        db_connection,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load from database and split into train/val/test

        Returns:
            train_df, val_df, test_df
        """
        # Load raw data
        query = """
            SELECT
                draw_date,
                draw_number,
                number_1, number_2, number_3,
                number_4, number_5, number_6,
                additional_number,
                draw_type
            FROM toto_draws
            ORDER BY draw_date ASC
        """
        df = pd.read_sql(query, db_connection)

        # Engineer features
        df = self.feature_engineer.engineer_features(df)

        # Remove rows with NaN (from lag features)
        df = df.dropna()

        # Time-based split
        n_total = len(df)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val

        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]
        test_df = df.iloc[n_train + n_val:]

        print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        return train_df, val_df, test_df

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for RNN/LSTM models

        Args:
            df: DataFrame with features
            sequence_length: Number of past draws to use

        Returns:
            X: [n_samples, sequence_length, n_features]
            y: [n_samples, 49] (binary target for each number)
        """
        # Select feature columns
        feature_cols = [col for col in df.columns
                       if col not in ['draw_date', 'draw_number',
                                     'number_1', 'number_2', 'number_3',
                                     'number_4', 'number_5', 'number_6',
                                     'additional_number']]

        X_features = df[feature_cols].values

        # Create target: binary vector of 49 numbers
        y_target = self._create_target_vectors(df)

        # Create sequences
        X_sequences = []
        y_sequences = []

        for i in range(sequence_length, len(df)):
            X_sequences.append(X_features[i - sequence_length:i])
            y_sequences.append(y_target[i])

        return np.array(X_sequences), np.array(y_sequences)

    def _create_target_vectors(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create binary target vectors [n_samples, 49]
        1 if number appears in draw, 0 otherwise
        """
        n_draws = len(df)
        targets = np.zeros((n_draws, 49), dtype=int)

        for idx, row in df.iterrows():
            winning = [row[f'number_{i}'] for i in range(1, 7)]
            for num in winning:
                targets[idx, num - 1] = 1

        return targets

    def get_time_series_cv_splits(
        self,
        df: pd.DataFrame,
        n_splits: int = 5
    ):
        """
        Create time-series cross-validation splits
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return tscv.split(df)
```

---

## 4. Data Quality Monitoring

### 4.1 Data Quality Metrics

```python
# pipeline/data_quality.py

import pandas as pd
import numpy as np
from typing import Dict, List

class DataQualityMonitor:
    """
    Monitors data quality and detects anomalies
    """

    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data quality report
        """
        report = {
            'total_draws': len(df),
            'date_range': {
                'start': df['draw_date'].min(),
                'end': df['draw_date'].max(),
            },
            'missing_data': self._check_missing_data(df),
            'number_distribution': self._analyze_number_distribution(df),
            'temporal_gaps': self._check_temporal_gaps(df),
            'outliers': self._detect_outliers(df),
            'data_completeness': self._check_completeness(df),
        }

        return report

    def _check_missing_data(self, df: pd.DataFrame) -> Dict:
        """Check for missing values"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        return {
            col: {'count': int(missing[col]), 'percentage': float(missing_pct[col])}
            for col in df.columns if missing[col] > 0
        }

    def _analyze_number_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze distribution of drawn numbers"""
        all_numbers = []
        for col in ['number_1', 'number_2', 'number_3',
                   'number_4', 'number_5', 'number_6']:
            all_numbers.extend(df[col].tolist())

        unique, counts = np.unique(all_numbers, return_counts=True)

        return {
            'min_frequency': int(counts.min()),
            'max_frequency': int(counts.max()),
            'mean_frequency': float(counts.mean()),
            'std_frequency': float(counts.std()),
            'most_common': int(unique[counts.argmax()]),
            'least_common': int(unique[counts.argmin()]),
        }

    def _check_temporal_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """Check for unexpected gaps in draw dates"""
        df = df.sort_values('draw_date')
        dates = pd.to_datetime(df['draw_date'])

        gaps = []
        for i in range(1, len(dates)):
            days_diff = (dates.iloc[i] - dates.iloc[i-1]).days

            # Expected: 3 days (Mon to Thu) or 4 days (Thu to Mon)
            if days_diff > 7:  # Suspicious gap
                gaps.append({
                    'from': str(dates.iloc[i-1]),
                    'to': str(dates.iloc[i]),
                    'days': days_diff
                })

        return gaps

    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect statistical outliers"""
        outliers = {}

        # Check prize pool outliers
        if 'prize_pool' in df.columns:
            prize_pool = df['prize_pool'].dropna()
            q1, q3 = prize_pool.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers['prize_pool'] = {
                'count': int(((prize_pool < q1 - 1.5*iqr) |
                             (prize_pool > q3 + 1.5*iqr)).sum())
            }

        return outliers

    def _check_completeness(self, df: pd.DataFrame) -> Dict:
        """Check data completeness"""
        required_cols = ['draw_date', 'number_1', 'number_2', 'number_3',
                        'number_4', 'number_5', 'number_6', 'additional_number']

        completeness = {}
        for col in required_cols:
            if col in df.columns:
                completeness[col] = float((df[col].notna().sum() / len(df)) * 100)

        return completeness
```

---

## 5. Pipeline Orchestration

### 5.1 Main Pipeline Class

```python
# pipeline/main_pipeline.py

import logging
from typing import Optional
import schedule
import time

class TotoPipeline:
    """
    Orchestrates the entire data pipeline
    """

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.scraper = TotoDataScraper(config['base_url'])
        self.validator = TotoDataValidator()
        self.db = Database(config['db_connection'])
        self.collector = TotoDataCollector(self.scraper, self.db, self.validator)
        self.feature_engineer = TotoFeatureEngineer()
        self.data_loader = TotoDataLoader(self.feature_engineer)
        self.quality_monitor = DataQualityMonitor()

    def run_initial_setup(self):
        """One-time historical data collection"""
        self.logger.info("Starting initial data collection...")

        # Collect historical data (1968 to now)
        success, failed = self.collector.collect_historical_data(
            start_date='1968-06-09',  # First TOTO draw
            end_date=datetime.now().strftime('%Y-%m-%d')
        )

        self.logger.info(f"Initial setup complete: {success} draws collected")

        # Generate quality report
        df = self.data_loader.load_raw_data()
        report = self.quality_monitor.generate_quality_report(df)

        self.logger.info(f"Data quality report: {report}")

    def run_daily_update(self):
        """Daily update to fetch latest draws"""
        self.logger.info("Running daily update...")

        success, failed = self.collector.update_latest_draws(lookback_days=7)

        if success > 0:
            self.logger.info(f"Updated {success} new draws")
        else:
            self.logger.info("No new draws found")

    def schedule_updates(self):
        """Schedule automatic updates"""
        # Run every Monday and Thursday at 7:30 PM (after draw at 6:30 PM)
        schedule.every().monday.at("19:30").do(self.run_daily_update)
        schedule.every().thursday.at("19:30").do(self.run_daily_update)

        self.logger.info("Scheduled automatic updates")

        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def prepare_ml_data(self):
        """Prepare data for ML training"""
        self.logger.info("Preparing ML-ready data...")

        train_df, val_df, test_df = self.data_loader.load_and_prepare(
            db_connection=self.db.connection
        )

        # Save to disk
        train_df.to_parquet('data/train.parquet')
        val_df.to_parquet('data/val.parquet')
        test_df.to_parquet('data/test.parquet')

        self.logger.info("ML data saved to disk")

        return train_df, val_df, test_df
```

---

## 6. Configuration

```yaml
# config.yaml

data_collection:
  base_url: "https://www.singaporepools.com.sg/en/product/Pages/toto_results.aspx"
  rate_limit: 2.0  # seconds between requests
  retry_attempts: 3
  retry_backoff: 2.0  # exponential backoff multiplier

database:
  type: "postgresql"  # or "sqlite"
  connection_string: "postgresql://user:pass@localhost:5432/toto_ml"
  pool_size: 5

feature_engineering:
  lookback_windows: [5, 10, 20, 50, 100]
  sequence_length: 50  # for RNN/LSTM
  include_external_features: true

data_splits:
  train_size: 0.70
  val_size: 0.15
  test_size: 0.15

logging:
  level: "INFO"
  file: "logs/pipeline.log"
  max_bytes: 10485760  # 10MB
  backup_count: 5

monitoring:
  quality_check_frequency: "daily"
  alert_on_missing_draws: true
  alert_threshold_days: 5
```

---

## 7. Usage Examples

### 7.1 Initial Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run initial data collection
python pipeline/main_pipeline.py --mode initial_setup

# Verify data quality
python pipeline/main_pipeline.py --mode quality_check
```

### 7.2 Daily Updates

```bash
# Manual update
python pipeline/main_pipeline.py --mode update

# Start scheduled updates (runs continuously)
python pipeline/main_pipeline.py --mode schedule
```

### 7.3 Prepare ML Data

```bash
# Generate train/val/test splits
python pipeline/main_pipeline.py --mode prepare_ml_data

# Output files:
# - data/train.parquet
# - data/val.parquet
# - data/test.parquet
```

---

## 8. Next Steps

After data pipeline is implemented:
1. Implement the 5 ML algorithms
2. Training pipeline setup
3. Model evaluation framework
4. Prediction service deployment

---

**Document Version:** 1.0
**Last Updated:** 2025-11-12
**Status:** Design Complete - Ready for Implementation
