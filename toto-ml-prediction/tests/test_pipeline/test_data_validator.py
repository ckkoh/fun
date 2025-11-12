"""Tests for data validator."""

import pytest
from pipeline.data_validator import TotoDataValidator


class TestTotoDataValidator:
    """Test suite for TotoDataValidator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.validator = TotoDataValidator()

    def test_valid_draw(self):
        """Test validation of valid draw data."""
        valid_draw = {
            'draw_number': 1,
            'draw_date': '2025-01-06',
            'day_of_week': 'Monday',
            'winning_numbers': [5, 12, 18, 24, 35, 42],
            'additional_number': 9,
            'prize_pool': 2500000.00,
            'group_1_winners': 1,
            'group_2_winners': 2,
            'group_3_winners': 10,
            'group_4_winners': 100,
            'draw_type': 'normal'
        }

        assert self.validator.validate(valid_draw) is True

    def test_invalid_numbers_out_of_range(self):
        """Test validation fails for numbers out of range."""
        invalid_draw = {
            'draw_number': 1,
            'draw_date': '2025-01-06',
            'day_of_week': 'Monday',
            'winning_numbers': [5, 12, 18, 24, 35, 50],  # 50 is out of range
            'additional_number': 9,
            'prize_pool': 2500000.00,
            'group_1_winners': 0,
        }

        assert self.validator.validate(invalid_draw) is False

    def test_invalid_unsorted_numbers(self):
        """Test validation fails for unsorted numbers."""
        invalid_draw = {
            'draw_number': 1,
            'draw_date': '2025-01-06',
            'day_of_week': 'Monday',
            'winning_numbers': [42, 12, 18, 24, 35, 5],  # Not sorted
            'additional_number': 9,
            'prize_pool': 2500000.00,
            'group_1_winners': 0,
        }

        assert self.validator.validate(invalid_draw) is False

    def test_invalid_duplicate_number(self):
        """Test validation fails when additional number is in winning numbers."""
        invalid_draw = {
            'draw_number': 1,
            'draw_date': '2025-01-06',
            'day_of_week': 'Monday',
            'winning_numbers': [5, 12, 18, 24, 35, 42],
            'additional_number': 12,  # Duplicate
            'prize_pool': 2500000.00,
            'group_1_winners': 0,
        }

        assert self.validator.validate(invalid_draw) is False

    def test_invalid_wrong_day(self):
        """Test validation fails for draws not on Monday or Thursday."""
        invalid_draw = {
            'draw_number': 1,
            'draw_date': '2025-01-07',  # Tuesday
            'day_of_week': 'Tuesday',
            'winning_numbers': [5, 12, 18, 24, 35, 42],
            'additional_number': 9,
            'prize_pool': 2500000.00,
            'group_1_winners': 0,
        }

        assert self.validator.validate(invalid_draw) is False

    def test_invalid_date_mismatch(self):
        """Test validation fails when stated day doesn't match actual date."""
        invalid_draw = {
            'draw_number': 1,
            'draw_date': '2025-01-06',  # Monday
            'day_of_week': 'Thursday',  # Wrong!
            'winning_numbers': [5, 12, 18, 24, 35, 42],
            'additional_number': 9,
            'prize_pool': 2500000.00,
            'group_1_winners': 0,
        }

        assert self.validator.validate(invalid_draw) is False

    def test_invalid_negative_prize(self):
        """Test validation fails for negative prize pool."""
        invalid_draw = {
            'draw_number': 1,
            'draw_date': '2025-01-06',
            'day_of_week': 'Monday',
            'winning_numbers': [5, 12, 18, 24, 35, 42],
            'additional_number': 9,
            'prize_pool': -1000.00,  # Negative
            'group_1_winners': 0,
        }

        assert self.validator.validate(invalid_draw) is False

    def test_batch_validation(self):
        """Test batch validation of multiple draws."""
        draws = [
            {
                'draw_number': 1,
                'draw_date': '2025-01-06',
                'day_of_week': 'Monday',
                'winning_numbers': [5, 12, 18, 24, 35, 42],
                'additional_number': 9,
                'prize_pool': 2500000.00,
                'group_1_winners': 1,
            },
            {
                'draw_number': 2,
                'draw_date': '2025-01-09',
                'day_of_week': 'Thursday',
                'winning_numbers': [50, 12, 18, 24, 35, 5],  # Invalid (50, unsorted)
                'additional_number': 9,
                'prize_pool': 2500000.00,
                'group_1_winners': 0,
            },
            {
                'draw_number': 3,
                'draw_date': '2025-01-13',
                'day_of_week': 'Monday',
                'winning_numbers': [1, 2, 3, 4, 5, 6],
                'additional_number': 7,
                'prize_pool': 3000000.00,
                'group_1_winners': 0,
            },
        ]

        valid, invalid = self.validator.validate_batch(draws)

        assert len(valid) == 2
        assert len(invalid) == 1
