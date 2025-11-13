"""
Phase 3 Models for 4D Prediction

Phase 3A: Quick Wins - Statistical + Pattern-Based Approaches
- Statistical Frequency Method: Pure statistical analysis without ML
- Pattern Mining Method: Rule-based with domain knowledge

These methods address Phase 1 & 2 limitations:
- No overfitting (no training required)
- Fast predictions (<1 second)
- Highly interpretable
- Naturally handle class imbalance
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import logging


class StatisticalFrequencyPredictor:
    """
    Phase 3A Method 1: Statistical Frequency-Based Prediction

    Uses pure statistical analysis without ML:
    - Historical frequency of each 4D number
    - Recency scores (days since last appearance)
    - Position-specific digit frequencies
    - Pattern occurrence rates

    No training required - computes statistics on-the-fly.
    """

    def __init__(self,
                 freq_weight: float = 0.3,
                 recency_weight: float = 0.3,
                 position_weight: float = 0.2,
                 pattern_weight: float = 0.2):
        """Initialize with scoring weights."""
        self.logger = logging.getLogger(__name__)
        self.freq_weight = freq_weight
        self.recency_weight = recency_weight
        self.position_weight = position_weight
        self.pattern_weight = pattern_weight

        # Statistics storage
        self.number_frequencies = Counter()
        self.number_last_seen = {}
        self.digit_position_freq = [Counter() for _ in range(4)]
        self.pattern_stats = {}

    def analyze_historical_data(self, df: pd.DataFrame):
        """Compute statistics from historical draws."""
        self.logger.info("Analyzing historical data for statistical features...")

        # Reset statistics
        self.number_frequencies.clear()
        self.number_last_seen.clear()
        self.digit_position_freq = [Counter() for _ in range(4)]

        # Extract all winning numbers from all draws
        winning_cols = ['first_prize', 'second_prize', 'third_prize']
        winning_cols += [f'starter_{i}' for i in range(1, 11)]
        winning_cols += [f'consolation_{i}' for i in range(1, 11)]

        for idx, row in df.iterrows():
            draw_date = row['draw_date']

            for col in winning_cols:
                if col in row and pd.notna(row[col]):
                    number = str(row[col]).zfill(4)

                    # Update frequency
                    self.number_frequencies[number] += 1

                    # Update last seen date
                    self.number_last_seen[number] = draw_date

                    # Update digit position frequencies
                    for pos, digit in enumerate(number):
                        self.digit_position_freq[pos][digit] += 1

        self.logger.info(f"Analyzed {len(df)} draws")
        self.logger.info(f"Found {len(self.number_frequencies)} unique 4D numbers")

    def _compute_frequency_score(self, number: str, total_draws: int) -> float:
        """Frequency score: normalized appearance count."""
        count = self.number_frequencies.get(number, 0)
        if total_draws == 0:
            return 0.0
        return count / total_draws

    def _compute_recency_score(self, number: str, current_date: str) -> float:
        """Recency score: inverse of days since last appearance."""
        if number not in self.number_last_seen:
            # Never appeared - assign low score
            return 0.0

        last_seen = self.number_last_seen[number]
        days_diff = (pd.to_datetime(current_date) - pd.to_datetime(last_seen)).days

        if days_diff == 0:
            days_diff = 1  # Avoid division by zero

        # Inverse recency: more recent = higher score
        return 1.0 / (days_diff + 1)

    def _compute_position_score(self, number: str) -> float:
        """Position score: average digit frequency across positions."""
        scores = []
        for pos, digit in enumerate(number):
            total = sum(self.digit_position_freq[pos].values())
            if total > 0:
                freq = self.digit_position_freq[pos].get(digit, 0)
                scores.append(freq / total)
            else:
                scores.append(0.0)

        return np.mean(scores) if scores else 0.0

    def _compute_pattern_score(self, number: str) -> float:
        """Pattern score: bonus for special patterns."""
        score = 0.0

        # Palindrome
        if number == number[::-1]:
            score += 0.3

        # Repeating digits
        unique_digits = len(set(number))
        if unique_digits == 1:  # All same (e.g., 1111)
            score += 0.4
        elif unique_digits == 2:  # Two distinct digits (e.g., 1122)
            score += 0.2

        # Sequential
        digits = [int(d) for d in number]
        is_seq_asc = all(digits[i+1] - digits[i] == 1 for i in range(3))
        is_seq_desc = all(digits[i] - digits[i+1] == 1 for i in range(3))
        if is_seq_asc or is_seq_desc:
            score += 0.3

        # Sum in middle range (10-26 is common)
        digit_sum = sum(digits)
        if 10 <= digit_sum <= 26:
            score += 0.1

        return score

    def predict_top_k(self, df: pd.DataFrame, current_date: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Generate top-K predictions based on statistical scores.

        Args:
            df: Historical draws DataFrame
            current_date: Current date for recency calculation
            k: Number of predictions to return

        Returns:
            List of (number, score) tuples sorted by score
        """
        # Analyze historical data
        self.analyze_historical_data(df)

        total_draws = len(df)

        # Compute composite score for all 10,000 numbers
        scores = {}

        for num in range(10000):
            number = f"{num:04d}"

            freq_score = self._compute_frequency_score(number, total_draws)
            recency_score = self._compute_recency_score(number, current_date)
            position_score = self._compute_position_score(number)
            pattern_score = self._compute_pattern_score(number)

            # Weighted combination
            composite_score = (
                self.freq_weight * freq_score +
                self.recency_weight * recency_score +
                self.position_weight * position_score +
                self.pattern_weight * pattern_score
            )

            scores[number] = composite_score

        # Sort by score and return top-K
        sorted_numbers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_numbers[:k]


class PatternMiningPredictor:
    """
    Phase 3A Method 2: Pattern Mining with Rule-Based Generation

    Mines frequent patterns from historical data:
    - Palindromes, sequences, repeating digits
    - Sum ranges, digit pairs
    - Conditional patterns based on recent draws

    Generates numbers matching successful patterns.
    """

    def __init__(self):
        """Initialize pattern mining predictor."""
        self.logger = logging.getLogger(__name__)

        # Pattern statistics
        self.pattern_success_rates = {}
        self.digit_pair_frequencies = Counter()
        self.sum_distribution = Counter()

    def analyze_patterns(self, df: pd.DataFrame):
        """Analyze pattern frequencies in winning numbers."""
        self.logger.info("Mining patterns from historical data...")

        winning_cols = ['first_prize', 'second_prize', 'third_prize']
        winning_cols += [f'starter_{i}' for i in range(1, 11)]
        winning_cols += [f'consolation_{i}' for i in range(1, 11)]

        pattern_counts = Counter()
        total_numbers = 0

        for idx, row in df.iterrows():
            for col in winning_cols:
                if col in row and pd.notna(row[col]):
                    number = str(row[col]).zfill(4)
                    total_numbers += 1

                    # Identify patterns
                    patterns = self._identify_patterns(number)
                    for pattern in patterns:
                        pattern_counts[pattern] += 1

                    # Digit pair frequencies
                    for i in range(3):
                        pair = number[i:i+2]
                        self.digit_pair_frequencies[pair] += 1

                    # Sum distribution
                    digit_sum = sum(int(d) for d in number)
                    self.sum_distribution[digit_sum] += 1

        # Calculate success rates
        for pattern, count in pattern_counts.items():
            self.pattern_success_rates[pattern] = count / total_numbers

        self.logger.info(f"Found {len(self.pattern_success_rates)} patterns")
        self.logger.info(f"Top patterns: {sorted(self.pattern_success_rates.items(), key=lambda x: x[1], reverse=True)[:5]}")

    def _identify_patterns(self, number: str) -> List[str]:
        """Identify patterns in a 4D number."""
        patterns = []

        # Palindrome
        if number == number[::-1]:
            patterns.append('palindrome')

        # Repeating digits
        unique = len(set(number))
        if unique == 1:
            patterns.append('all_same')
        elif unique == 2:
            patterns.append('two_distinct')
        elif unique == 3:
            patterns.append('three_distinct')
        else:
            patterns.append('all_different')

        # Sequential
        digits = [int(d) for d in number]
        if all(digits[i+1] - digits[i] == 1 for i in range(3)):
            patterns.append('seq_ascending')
        elif all(digits[i] - digits[i+1] == 1 for i in range(3)):
            patterns.append('seq_descending')

        # Sum ranges
        digit_sum = sum(digits)
        if digit_sum < 10:
            patterns.append('sum_low')
        elif 10 <= digit_sum <= 20:
            patterns.append('sum_mid_low')
        elif 21 <= digit_sum <= 30:
            patterns.append('sum_mid_high')
        else:
            patterns.append('sum_high')

        # Even/odd
        even_count = sum(1 for d in digits if d % 2 == 0)
        if even_count == 4:
            patterns.append('all_even')
        elif even_count == 0:
            patterns.append('all_odd')
        else:
            patterns.append(f'mixed_even_{even_count}')

        return patterns

    def _analyze_recent_trends(self, df: pd.DataFrame, last_n: int = 10) -> Dict:
        """Analyze patterns in recent N draws."""
        recent = df.tail(last_n)

        trends = {
            'avg_sum': 0,
            'palindrome_rate': 0,
            'common_patterns': []
        }

        sums = []
        palindrome_count = 0
        pattern_counts = Counter()

        for _, row in recent.iterrows():
            if 'first_prize' in row and pd.notna(row['first_prize']):
                number = str(row['first_prize']).zfill(4)

                digit_sum = sum(int(d) for d in number)
                sums.append(digit_sum)

                if number == number[::-1]:
                    palindrome_count += 1

                patterns = self._identify_patterns(number)
                for p in patterns:
                    pattern_counts[p] += 1

        trends['avg_sum'] = np.mean(sums) if sums else 18
        trends['palindrome_rate'] = palindrome_count / last_n
        trends['common_patterns'] = [p for p, _ in pattern_counts.most_common(3)]

        return trends

    def generate_candidate_numbers(self, active_patterns: List[str], count: int = 100) -> List[str]:
        """Generate candidate numbers matching active patterns."""
        candidates = set()

        for num in range(10000):
            number = f"{num:04d}"
            patterns = self._identify_patterns(number)

            # Check if number matches any active patterns
            match_count = sum(1 for p in active_patterns if p in patterns)

            if match_count >= 2:  # Matches at least 2 patterns
                candidates.add(number)

            if len(candidates) >= count:
                break

        return list(candidates)

    def predict_top_k(self, df: pd.DataFrame, k: int = 10) -> List[Tuple[str, float]]:
        """
        Generate top-K predictions based on pattern mining.

        Args:
            df: Historical draws DataFrame
            k: Number of predictions to return

        Returns:
            List of (number, confidence) tuples
        """
        # Analyze patterns
        self.analyze_patterns(df)

        # Analyze recent trends
        trends = self._analyze_recent_trends(df, last_n=10)

        # Determine active patterns based on trends
        active_patterns = trends['common_patterns']

        # Add pattern based on average sum
        avg_sum = trends['avg_sum']
        if avg_sum < 15:
            active_patterns.append('sum_low')
        elif avg_sum < 22:
            active_patterns.append('sum_mid_low')
        else:
            active_patterns.append('sum_mid_high')

        self.logger.info(f"Active patterns: {active_patterns}")

        # Generate candidates matching active patterns
        candidates = self.generate_candidate_numbers(active_patterns, count=1000)

        self.logger.info(f"Generated {len(candidates)} candidate numbers")

        # Score candidates
        scores = {}
        for number in candidates:
            score = 0.0
            patterns = self._identify_patterns(number)

            # Score based on pattern success rates
            for pattern in patterns:
                if pattern in self.pattern_success_rates:
                    score += self.pattern_success_rates[pattern]

            # Bonus for digit pairs
            for i in range(3):
                pair = number[i:i+2]
                if pair in self.digit_pair_frequencies:
                    total_pairs = sum(self.digit_pair_frequencies.values())
                    score += self.digit_pair_frequencies[pair] / total_pairs

            scores[number] = score

        # Sort and return top-K
        sorted_numbers = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Normalize scores to 0-1 range
        if sorted_numbers:
            max_score = sorted_numbers[0][1]
            if max_score > 0:
                sorted_numbers = [(num, score/max_score) for num, score in sorted_numbers]

        return sorted_numbers[:k]
