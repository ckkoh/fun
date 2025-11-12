"""Data collection orchestration for TOTO ML system."""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Tuple
from tqdm import tqdm

from pipeline.scraper import TotoDataScraper
from pipeline.data_validator import TotoDataValidator
from database.db_manager import DatabaseManager


class TotoDataCollector:
    """
    Orchestrates data collection, validation, and storage.
    """

    def __init__(
        self,
        scraper: TotoDataScraper,
        database: DatabaseManager,
        validator: TotoDataValidator
    ):
        """
        Initialize data collector.

        Args:
            scraper: Data scraper instance
            database: Database manager instance
            validator: Data validator instance
        """
        self.scraper = scraper
        self.db = database
        self.validator = validator
        self.logger = logging.getLogger(__name__)

    def collect_historical_data(
        self,
        start_date: str,
        end_date: str,
        skip_existing: bool = True
    ) -> Tuple[int, int]:
        """
        Collect all draws between date range.

        Args:
            start_date: ISO format 'YYYY-MM-DD'
            end_date: ISO format 'YYYY-MM-DD'
            skip_existing: If True, skip draws already in database

        Returns:
            Tuple of (successful_count, failed_count)
        """
        self.logger.info(f"Starting historical data collection from {start_date} to {end_date}")

        # Generate list of potential draw dates
        draws_to_fetch = self._generate_draw_dates(start_date, end_date)
        self.logger.info(f"Found {len(draws_to_fetch)} potential draw dates")

        successful = 0
        failed = 0
        skipped = 0

        # Progress bar
        pbar = tqdm(draws_to_fetch, desc="Collecting draws")

        for draw_date in pbar:
            try:
                # Check if already exists
                if skip_existing:
                    existing = self.db.get_draw_by_date(draw_date)
                    if existing:
                        self.logger.debug(f"Draw for {draw_date} already exists, skipping")
                        skipped += 1
                        pbar.set_postfix({
                            'success': successful,
                            'failed': failed,
                            'skipped': skipped
                        })
                        continue

                # Fetch raw data
                raw_data = self.scraper.fetch_draw_by_date(draw_date)

                if raw_data is None:
                    self.logger.warning(f"No draw found for {draw_date}")
                    failed += 1
                    continue

                # Validate data
                if not self.validator.validate(raw_data):
                    self.logger.error(f"Validation failed for {draw_date}")
                    failed += 1
                    continue

                # Transform to database format
                db_record = self._transform_to_db_format(raw_data)

                # Store in database
                draw_id = self.db.insert_draw(db_record)

                if draw_id > 0:
                    successful += 1
                    self.logger.info(f"Successfully stored draw for {draw_date}")
                else:
                    failed += 1

                # Update progress bar
                pbar.set_postfix({
                    'success': successful,
                    'failed': failed,
                    'skipped': skipped
                })

            except Exception as e:
                self.logger.error(f"Error processing {draw_date}: {str(e)}")
                failed += 1

        self.logger.info(
            f"Collection complete: {successful} successful, "
            f"{failed} failed, {skipped} skipped"
        )

        return successful, failed

    def update_latest_draws(self, lookback_days: int = 7) -> Tuple[int, int]:
        """
        Fetch and update most recent draws.

        Args:
            lookback_days: Number of days to look back

        Returns:
            Tuple of (successful_count, failed_count)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        return self.collect_historical_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            skip_existing=True
        )

    def collect_by_draw_numbers(
        self,
        start_draw: int,
        end_draw: int
    ) -> Tuple[int, int]:
        """
        Collect draws by draw number range.

        Args:
            start_draw: Starting draw number
            end_draw: Ending draw number

        Returns:
            Tuple of (successful_count, failed_count)
        """
        self.logger.info(f"Collecting draws {start_draw} to {end_draw}")

        successful = 0
        failed = 0

        pbar = tqdm(range(start_draw, end_draw + 1), desc="Collecting by draw number")

        for draw_number in pbar:
            try:
                # Check if exists
                existing = self.db.get_draw_by_number(draw_number)
                if existing:
                    self.logger.debug(f"Draw {draw_number} already exists")
                    continue

                # Fetch
                raw_data = self.scraper.fetch_draw_by_number(draw_number)

                if raw_data is None:
                    failed += 1
                    continue

                # Validate
                if not self.validator.validate(raw_data):
                    failed += 1
                    continue

                # Store
                db_record = self._transform_to_db_format(raw_data)
                draw_id = self.db.insert_draw(db_record)

                if draw_id > 0:
                    successful += 1
                else:
                    failed += 1

                pbar.set_postfix({'success': successful, 'failed': failed})

            except Exception as e:
                self.logger.error(f"Error processing draw {draw_number}: {e}")
                failed += 1

        return successful, failed

    def _generate_draw_dates(self, start_date: str, end_date: str) -> List[str]:
        """Generate all Monday and Thursday dates in range."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        dates = []
        current = start

        while current <= end:
            # 0=Monday, 3=Thursday
            if current.weekday() in [0, 3]:
                dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)

        return dates

    def _transform_to_db_format(self, raw_data: dict) -> dict:
        """Transform scraped data to database schema format."""
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

    def verify_data_integrity(self) -> dict:
        """
        Verify integrity of collected data.

        Returns:
            Dictionary with integrity check results
        """
        self.logger.info("Running data integrity checks...")

        results = {
            'total_draws': 0,
            'date_gaps': [],
            'duplicate_numbers': [],
            'missing_data': []
        }

        # Get all draws
        all_draws = self.db.get_all_draws()
        results['total_draws'] = len(all_draws)

        if len(all_draws) < 2:
            return results

        # Check for date gaps
        for i in range(len(all_draws) - 1):
            date1 = datetime.strptime(all_draws[i]['draw_date'], '%Y-%m-%d')
            date2 = datetime.strptime(all_draws[i + 1]['draw_date'], '%Y-%m-%d')
            days_diff = (date2 - date1).days

            # Expected: 3-4 days between draws
            if days_diff > 7:
                results['date_gaps'].append({
                    'from': all_draws[i]['draw_date'],
                    'to': all_draws[i + 1]['draw_date'],
                    'days': days_diff
                })

        # Check for missing prize data
        for draw in all_draws:
            if draw.get('prize_pool') is None:
                results['missing_data'].append({
                    'draw_number': draw['draw_number'],
                    'field': 'prize_pool'
                })

        self.logger.info(f"Integrity check complete: {results}")
        return results


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize components
    scraper = TotoDataScraper(mock_mode=True)
    database = DatabaseManager("data/toto.db")
    validator = TotoDataValidator()

    # Create collector
    collector = TotoDataCollector(scraper, database, validator)

    # Collect historical data (small sample for testing)
    print("Collecting January 2025 draws...")
    success, failed = collector.collect_historical_data(
        '2025-01-01',
        '2025-01-31'
    )
    print(f"Results: {success} successful, {failed} failed")

    # Get database stats
    stats = database.get_database_stats()
    print(f"\nDatabase stats: {stats}")

    # Verify integrity
    print("\nVerifying data integrity...")
    integrity = collector.verify_data_integrity()
    print(f"Integrity check: {integrity}")
