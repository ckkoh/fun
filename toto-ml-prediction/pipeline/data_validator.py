"""Data validation for TOTO draw results."""

import logging
from typing import Dict, Tuple
from datetime import datetime


class TotoDataValidator:
    """Validates scraped TOTO data for correctness."""

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
            self._validate_numbers_sorted,
        ]

    def validate(self, data: Dict) -> bool:
        """
        Run all validation rules.

        Args:
            data: Draw data dictionary to validate

        Returns:
            True if all validations pass, False otherwise
        """
        for rule in self.validation_rules:
            is_valid, error_msg = rule(data)
            if not is_valid:
                self.logger.error(f"Validation failed: {error_msg}")
                return False

        self.logger.debug("All validation rules passed")
        return True

    def _validate_numbers_range(self, data: Dict) -> Tuple[bool, str]:
        """All numbers must be between 1 and 49."""
        numbers = data.get('winning_numbers', []) + [data.get('additional_number')]

        for num in numbers:
            if num is None:
                return False, "Missing number in data"
            if not isinstance(num, int):
                return False, f"Number must be integer, got {type(num)}"
            if not (1 <= num <= 49):
                return False, f"Number {num} out of range [1, 49]"

        return True, ""

    def _validate_numbers_unique(self, data: Dict) -> Tuple[bool, str]:
        """6 winning numbers must be unique."""
        numbers = data.get('winning_numbers', [])

        if len(numbers) != 6:
            return False, f"Expected 6 numbers, got {len(numbers)}"

        if len(set(numbers)) != 6:
            return False, "Winning numbers are not unique"

        return True, ""

    def _validate_additional_number(self, data: Dict) -> Tuple[bool, str]:
        """Additional number must not be in winning numbers."""
        additional = data.get('additional_number')
        winning = data.get('winning_numbers', [])

        if additional is None:
            return False, "Missing additional number"

        if additional in winning:
            return False, f"Additional number {additional} appears in winning numbers"

        return True, ""

    def _validate_numbers_sorted(self, data: Dict) -> Tuple[bool, str]:
        """Winning numbers should be in ascending order."""
        numbers = data.get('winning_numbers', [])

        if len(numbers) < 2:
            return True, ""

        for i in range(len(numbers) - 1):
            if numbers[i] >= numbers[i + 1]:
                return False, f"Numbers not sorted: {numbers}"

        return True, ""

    def _validate_date_format(self, data: Dict) -> Tuple[bool, str]:
        """Date must be valid ISO format."""
        try:
            date_str = data.get('draw_date')
            if not date_str:
                return False, "Missing draw_date"

            datetime.strptime(date_str, '%Y-%m-%d')
            return True, ""
        except ValueError as e:
            return False, f"Invalid date format: {str(e)}"

    def _validate_day_of_week(self, data: Dict) -> Tuple[bool, str]:
        """Draw must be on Monday or Thursday."""
        day = data.get('day_of_week')

        if not day:
            return False, "Missing day_of_week"

        if day not in ['Monday', 'Thursday']:
            return False, f"Invalid day of week: {day}"

        # Cross-check with date
        try:
            date_str = data.get('draw_date')
            if date_str:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                actual_day = date_obj.strftime('%A')

                if day != actual_day:
                    return False, f"Day mismatch: stated {day}, actual {actual_day}"
        except:
            pass  # Date validation will catch this

        return True, ""

    def _validate_draw_number(self, data: Dict) -> Tuple[bool, str]:
        """Draw number must be positive integer."""
        draw_num = data.get('draw_number')

        if draw_num is None:
            return False, "Missing draw_number"

        if not isinstance(draw_num, int):
            return False, f"draw_number must be integer, got {type(draw_num)}"

        if draw_num <= 0:
            return False, f"draw_number must be positive, got {draw_num}"

        return True, ""

    def _validate_prize_data(self, data: Dict) -> Tuple[bool, str]:
        """Prize pool and winners should be non-negative if present."""
        prize_pool = data.get('prize_pool')

        if prize_pool is not None:
            if not isinstance(prize_pool, (int, float)):
                return False, f"Invalid prize_pool type: {type(prize_pool)}"
            if prize_pool < 0:
                return False, f"Invalid prize_pool: {prize_pool}"

        # Check winner counts
        for group in range(1, 5):
            key = f'group_{group}_winners'
            winners = data.get(key, 0)

            if not isinstance(winners, int):
                return False, f"{key} must be integer"
            if winners < 0:
                return False, f"Invalid {key}: {winners}"

        return True, ""

    def validate_batch(self, data_list: list) -> Tuple[list, list]:
        """
        Validate a batch of draws.

        Args:
            data_list: List of draw data dictionaries

        Returns:
            Tuple of (valid_draws, invalid_draws)
        """
        valid = []
        invalid = []

        for data in data_list:
            if self.validate(data):
                valid.append(data)
            else:
                invalid.append(data)

        self.logger.info(
            f"Batch validation: {len(valid)} valid, {len(invalid)} invalid"
        )

        return valid, invalid


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    validator = TotoDataValidator()

    # Valid draw
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

    print("Testing valid draw...")
    is_valid = validator.validate(valid_draw)
    print(f"Valid: {is_valid}\n")

    # Invalid draw (numbers not sorted)
    invalid_draw_1 = {
        'draw_number': 2,
        'draw_date': '2025-01-09',
        'day_of_week': 'Thursday',
        'winning_numbers': [42, 12, 18, 24, 35, 5],  # Not sorted
        'additional_number': 9,
        'prize_pool': 2500000.00,
        'group_1_winners': 0,
    }

    print("Testing invalid draw (unsorted)...")
    is_valid = validator.validate(invalid_draw_1)
    print(f"Valid: {is_valid}\n")

    # Invalid draw (additional in winning)
    invalid_draw_2 = {
        'draw_number': 3,
        'draw_date': '2025-01-13',
        'day_of_week': 'Monday',
        'winning_numbers': [5, 12, 18, 24, 35, 42],
        'additional_number': 12,  # Already in winning numbers
        'prize_pool': 2500000.00,
        'group_1_winners': 0,
    }

    print("Testing invalid draw (duplicate number)...")
    is_valid = validator.validate(invalid_draw_2)
    print(f"Valid: {is_valid}\n")

    # Batch validation
    print("Testing batch validation...")
    valid_list, invalid_list = validator.validate_batch([
        valid_draw, invalid_draw_1, invalid_draw_2
    ])
    print(f"Valid: {len(valid_list)}, Invalid: {len(invalid_list)}")
