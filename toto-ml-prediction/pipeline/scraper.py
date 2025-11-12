"""Web scraper for Singapore Pools TOTO results."""

import time
import logging
import random
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup


class TotoDataScraper:
    """
    Scraper for fetching TOTO draw results from Singapore Pools.

    Note: This implementation includes a mock mode for testing and development.
    Real scraping should comply with Singapore Pools' terms of service.
    """

    def __init__(
        self,
        base_url: str = "https://www.singaporepools.com.sg/en/product/Pages/toto_results.aspx",
        rate_limit: float = 2.0,
        max_retries: int = 3,
        mock_mode: bool = True
    ):
        """
        Initialize scraper.

        Args:
            base_url: Base URL for TOTO results
            rate_limit: Minimum seconds between requests
            max_retries: Maximum retry attempts
            mock_mode: If True, use mock data instead of real scraping
        """
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.mock_mode = mock_mode
        self.logger = logging.getLogger(__name__)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        self.last_request_time = 0

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

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

        # Generate 6 unique random numbers
        winning_numbers = sorted(random.sample(range(1, 50), 6))

        # Generate additional number (not in winning numbers)
        remaining = [n for n in range(1, 50) if n not in winning_numbers]
        additional_number = random.choice(remaining)

        # Generate mock prize data
        prize_pool = random.uniform(1000000, 15000000)
        group_1_winners = random.choice([0, 0, 0, 1, 2])  # Usually 0 or few winners

        # Determine day of week
        date_obj = datetime.strptime(draw_date, '%Y-%m-%d')
        day_of_week = date_obj.strftime('%A')

        return {
            'draw_number': draw_number,
            'draw_date': draw_date,
            'day_of_week': day_of_week,
            'winning_numbers': winning_numbers,
            'additional_number': additional_number,
            'prize_pool': round(prize_pool, 2),
            'group_1_winners': group_1_winners,
            'group_2_winners': random.randint(0, 5),
            'group_3_winners': random.randint(5, 50),
            'group_4_winners': random.randint(50, 500),
            'draw_type': 'normal'
        }

    def fetch_draw_by_number(self, draw_number: int) -> Optional[Dict]:
        """
        Fetch draw data by draw number.

        Args:
            draw_number: The draw number to fetch

        Returns:
            Dictionary with draw data or None if not found
        """
        if self.mock_mode:
            self.logger.info(f"Fetching draw {draw_number} (mock mode)")
            # Generate a mock date (assuming draws start from 1968)
            base_date = datetime(1968, 6, 9)  # First TOTO draw
            days_offset = (draw_number - 1) * 4  # Roughly 2 draws per week
            draw_date = (base_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')

            return self._generate_mock_draw(draw_number, draw_date)

        # Real scraping implementation (placeholder)
        self.logger.info(f"Fetching draw {draw_number} from Singapore Pools")
        self._rate_limit_wait()

        try:
            # This is a placeholder - actual implementation would need to:
            # 1. Construct proper URL with draw number
            # 2. Parse the HTML response
            # 3. Extract draw data
            response = self.session.get(
                f"{self.base_url}?drawNumber={draw_number}",
                timeout=10
            )
            response.raise_for_status()

            # Parse response (implementation needed)
            draw_data = self._parse_draw_page(response.text)
            return draw_data

        except requests.RequestException as e:
            self.logger.error(f"Error fetching draw {draw_number}: {e}")
            return None

    def fetch_draw_by_date(self, draw_date: str) -> Optional[Dict]:
        """
        Fetch draw data by date.

        Args:
            draw_date: Date in 'YYYY-MM-DD' format

        Returns:
            Dictionary with draw data or None if not found
        """
        date_obj = datetime.strptime(draw_date, '%Y-%m-%d')
        day_name = date_obj.strftime('%A')

        # Check if it's a draw day
        if day_name not in ['Monday', 'Thursday']:
            self.logger.warning(f"{draw_date} is not a draw day (Monday/Thursday)")
            return None

        if self.mock_mode:
            self.logger.info(f"Fetching draw for {draw_date} (mock mode)")

            # Calculate approximate draw number
            base_date = datetime(1968, 6, 9)
            days_diff = (date_obj - base_date).days
            draw_number = max(1, days_diff // 4)

            return self._generate_mock_draw(draw_number, draw_date)

        # Real scraping implementation
        self._rate_limit_wait()

        try:
            response = self.session.get(
                f"{self.base_url}?date={draw_date}",
                timeout=10
            )
            response.raise_for_status()

            draw_data = self._parse_draw_page(response.text)
            return draw_data

        except requests.RequestException as e:
            self.logger.error(f"Error fetching draw for {draw_date}: {e}")
            return None

    def fetch_latest_draw(self) -> Optional[Dict]:
        """
        Fetch the most recent draw.

        Returns:
            Dictionary with latest draw data
        """
        if self.mock_mode:
            # Return a recent mock draw
            today = datetime.now()
            # Find last Monday or Thursday
            days_back = 0
            while True:
                check_date = today - timedelta(days=days_back)
                if check_date.strftime('%A') in ['Monday', 'Thursday']:
                    draw_date = check_date.strftime('%Y-%m-%d')
                    break
                days_back += 1
                if days_back > 7:  # Safety check
                    break

            return self.fetch_draw_by_date(draw_date)

        # Real scraping
        self._rate_limit_wait()

        try:
            response = self.session.get(self.base_url, timeout=10)
            response.raise_for_status()

            draw_data = self._parse_draw_page(response.text)
            return draw_data

        except requests.RequestException as e:
            self.logger.error(f"Error fetching latest draw: {e}")
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

        while current <= end:
            day_name = current.strftime('%A')

            # Only fetch on draw days (Monday/Thursday)
            if day_name in ['Monday', 'Thursday']:
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

    def _parse_draw_page(self, html_content: str) -> Optional[Dict]:
        """
        Parse HTML page to extract draw data.

        This is a placeholder implementation. Actual implementation would
        need to be customized based on Singapore Pools website structure.

        Args:
            html_content: HTML content to parse

        Returns:
            Dictionary with parsed draw data
        """
        try:
            soup = BeautifulSoup(html_content, 'lxml')

            # Placeholder parsing logic
            # Actual implementation would need to:
            # 1. Find draw number
            # 2. Extract winning numbers
            # 3. Extract additional number
            # 4. Extract prize information
            # 5. Extract winner counts

            # This is just a skeleton
            draw_data = {
                'draw_number': None,
                'draw_date': None,
                'day_of_week': None,
                'winning_numbers': [],
                'additional_number': None,
                'prize_pool': None,
                'group_1_winners': 0,
                'draw_type': 'normal'
            }

            return draw_data

        except Exception as e:
            self.logger.error(f"Error parsing HTML: {e}")
            return None


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create scraper in mock mode
    scraper = TotoDataScraper(mock_mode=True)

    # Fetch latest draw
    print("Fetching latest draw...")
    latest = scraper.fetch_latest_draw()
    print(f"Latest draw: {latest}")

    # Fetch specific draw
    print("\nFetching draw #100...")
    draw_100 = scraper.fetch_draw_by_number(100)
    print(f"Draw 100: {draw_100}")

    # Fetch by date
    print("\nFetching draw for 2025-01-06 (Monday)...")
    date_draw = scraper.fetch_draw_by_date('2025-01-06')
    print(f"Draw for 2025-01-06: {date_draw}")

    # Fetch date range
    print("\nFetching draws from 2025-01-01 to 2025-01-31...")
    draws = scraper.fetch_date_range('2025-01-01', '2025-01-31')
    print(f"Fetched {len(draws)} draws")
