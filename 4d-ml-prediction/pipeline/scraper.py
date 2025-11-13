"""
Data scraper for Singapore Pools 4D lottery results.

This module handles fetching 4D draw results with mock mode for development.
"""

import random
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup


class FourDDataScraper:
    """Scraper for 4D lottery results with mock mode support."""

    def __init__(
        self,
        mock_mode: bool = True,
        base_url: str = "https://www.singaporepools.com.sg/en/product/pages/4d_results.aspx",
        rate_limit: float = 2.0
    ):
        """
        Initialize 4D data scraper.

        Args:
            mock_mode: If True, generate mock data instead of real scraping
            base_url: Base URL for Singapore Pools 4D results
            rate_limit: Seconds to wait between requests
        """
        self.mock_mode = mock_mode
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.logger = logging.getLogger(__name__)

        if not mock_mode:
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

        logging.basicConfig(level=logging.INFO)

    def _rate_limit_wait(self):
        """Implement rate limiting between requests."""
        if not self.mock_mode:
            time.sleep(self.rate_limit)

    def _generate_mock_4d_number(self, seed: int) -> str:
        """
        Generate a consistent mock 4D number.

        Args:
            seed: Seed for random generation

        Returns:
            4-digit string (0000-9999)
        """
        random.seed(seed)
        return f"{random.randint(0, 9999):04d}"

    def _generate_mock_draw(self, draw_number: int, draw_date: str) -> Dict:
        """
        Generate mock draw data for testing.

        Args:
            draw_number: Draw number
            draw_date: Draw date in 'YYYY-MM-DD' format

        Returns:
            Dictionary with draw data
        """
        # Seed random with draw number for consistency
        random.seed(draw_number)

        # Generate 23 unique 4D numbers
        all_numbers = set()

        # Generate first prize (seed based on draw_number)
        first_prize = f"{random.randint(0, 9999):04d}"
        all_numbers.add(first_prize)

        # Generate second prize
        second_prize = f"{random.randint(0, 9999):04d}"
        while second_prize in all_numbers:
            second_prize = f"{random.randint(0, 9999):04d}"
        all_numbers.add(second_prize)

        # Generate third prize
        third_prize = f"{random.randint(0, 9999):04d}"
        while third_prize in all_numbers:
            third_prize = f"{random.randint(0, 9999):04d}"
        all_numbers.add(third_prize)

        # Generate 10 starter prizes
        starter_prizes = []
        for i in range(10):
            number = f"{random.randint(0, 9999):04d}"
            while number in all_numbers:
                number = f"{random.randint(0, 9999):04d}"
            starter_prizes.append(number)
            all_numbers.add(number)

        # Generate 10 consolation prizes
        consolation_prizes = []
        for i in range(10):
            number = f"{random.randint(0, 9999):04d}"
            while number in all_numbers:
                number = f"{random.randint(0, 9999):04d}"
            consolation_prizes.append(number)
            all_numbers.add(number)

        # Determine day of week
        date_obj = datetime.strptime(draw_date, '%Y-%m-%d')
        day_of_week = date_obj.strftime('%A')

        draw_data = {
            'draw_number': draw_number,
            'draw_date': draw_date,
            'day_of_week': day_of_week,
            'first_prize': first_prize,
            'second_prize': second_prize,
            'third_prize': third_prize,
            'draw_type': 'normal'
        }

        # Add starter prizes
        for i, number in enumerate(starter_prizes, 1):
            draw_data[f'starter_{i}'] = number

        # Add consolation prizes
        for i, number in enumerate(consolation_prizes, 1):
            draw_data[f'consolation_{i}'] = number

        return draw_data

    def fetch_draw_by_date(self, draw_date: str) -> Optional[Dict]:
        """
        Fetch draw data by date.

        Args:
            draw_date: Date in 'YYYY-MM-DD' format

        Returns:
            Dictionary with draw data or None if not found
        """
        if self.mock_mode:
            # Generate consistent draw number from date
            base_date = datetime(2020, 1, 1)
            target_date = datetime.strptime(draw_date, '%Y-%m-%d')
            days_diff = (target_date - base_date).days
            draw_number = 5000 + (days_diff // 2)  # Roughly 3 draws per week

            self.logger.info(f"Fetching draw for {draw_date} (mock mode)")
            return self._generate_mock_draw(draw_number, draw_date)

        # Real scraping implementation (placeholder)
        self.logger.info(f"Fetching draw for {draw_date} from Singapore Pools")
        self._rate_limit_wait()

        try:
            # This is a placeholder - actual implementation would need to:
            # 1. Construct proper URL with date parameter
            # 2. Parse the HTML response
            # 3. Extract all 23 winning numbers
            response = self.session.get(
                f"{self.base_url}?date={draw_date}",
                timeout=10
            )
            response.raise_for_status()

            # Parse response (implementation needed)
            draw_data = self._parse_draw_page(response.text, draw_date)
            return draw_data

        except requests.RequestException as e:
            self.logger.error(f"Error fetching draw for {draw_date}: {e}")
            return None

    def fetch_date_range(
        self,
        start_date: str,
        end_date: str,
        progress_callback=None
    ) -> List[Dict]:
        """
        Fetch all draws within a date range.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            progress_callback: Optional callback function for progress updates

        Returns:
            List of draw dictionaries
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        draws = []
        current = start

        # 4D draws are Wednesday, Saturday, Sunday
        while current <= end:
            day_name = current.strftime('%A')

            # Only fetch on draw days
            if day_name in ['Wednesday', 'Saturday', 'Sunday']:
                draw_date = current.strftime('%Y-%m-%d')
                draw_data = self.fetch_draw_by_date(draw_date)

                if draw_data:
                    draws.append(draw_data)
                    self.logger.info(f"Fetched draw for {draw_date}")

                    if progress_callback:
                        progress_callback(draw_date, len(draws))

            current += timedelta(days=1)

        self.logger.info(f"Fetched {len(draws)} draws from {start_date} to {end_date}")
        return draws

    def _parse_draw_page(self, html_content: str, draw_date: str) -> Optional[Dict]:
        """
        Parse HTML page to extract draw data.

        This is a placeholder - actual implementation would need to
        parse the specific HTML structure of Singapore Pools website.

        Args:
            html_content: HTML content from the page
            draw_date: Date of the draw

        Returns:
            Dictionary with draw data or None if parsing fails
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Placeholder implementation
            # Actual implementation would need to:
            # 1. Find the table/section with 4D results
            # 2. Extract first, second, third prizes
            # 3. Extract 10 starter prizes
            # 4. Extract 10 consolation prizes

            # For now, return None to indicate parsing not implemented
            self.logger.warning("Real HTML parsing not implemented yet")
            return None

        except Exception as e:
            self.logger.error(f"Error parsing draw page: {e}")
            return None

    def validate_draw_data(self, draw_data: Dict) -> bool:
        """
        Validate draw data before insertion.

        Args:
            draw_data: Draw data dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = ['draw_number', 'draw_date', 'day_of_week',
                          'first_prize', 'second_prize', 'third_prize']

        # Check required fields
        for field in required_fields:
            if field not in draw_data or not draw_data[field]:
                self.logger.error(f"Missing required field: {field}")
                return False

        # Validate 4D number format (exactly 4 digits)
        number_fields = ['first_prize', 'second_prize', 'third_prize']
        for i in range(1, 11):
            number_fields.extend([f'starter_{i}', f'consolation_{i}'])

        for field in number_fields:
            if field in draw_data and draw_data[field]:
                number = draw_data[field]
                if not isinstance(number, str) or len(number) != 4 or not number.isdigit():
                    self.logger.error(f"Invalid 4D number format for {field}: {number}")
                    return False

        # Check for duplicate numbers
        all_numbers = [
            draw_data['first_prize'],
            draw_data['second_prize'],
            draw_data['third_prize']
        ]

        for i in range(1, 11):
            for prefix in ['starter', 'consolation']:
                key = f'{prefix}_{i}'
                if key in draw_data and draw_data[key]:
                    all_numbers.append(draw_data[key])

        if len(all_numbers) != len(set(all_numbers)):
            self.logger.error("Duplicate numbers found in draw")
            return False

        # Validate date format
        try:
            datetime.strptime(draw_data['draw_date'], '%Y-%m-%d')
        except ValueError:
            self.logger.error(f"Invalid date format: {draw_data['draw_date']}")
            return False

        return True
