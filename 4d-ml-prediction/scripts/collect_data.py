"""
Collect 4D draw data from 2024-2025.

This script collects historical 4D draw data for training.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from datetime import datetime, timedelta
from pipeline.scraper import FourDDataScraper
from database.db_manager import FourDDatabaseManager


def main():
    print("=" * 70)
    print("COLLECTING 4D DRAW DATA")
    print("=" * 70)
    print()

    db = FourDDatabaseManager()
    scraper = FourDDataScraper(mock_mode=True)

    # Collect 2024-2025 data (Wed, Sat, Sun draws)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 6, 30)

    # Generate draw schedule
    draw_dates = []
    current_date = start_date

    while current_date <= end_date:
        # 2=Wednesday, 5=Saturday, 6=Sunday
        if current_date.weekday() in [2, 5, 6]:
            draw_dates.append(current_date)
        current_date += timedelta(days=1)

    print(f"Target period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Expected draws: {len(draw_dates)} (Wed, Sat, Sun)")
    print()

    # Collect draws
    collected = 0
    skipped = 0
    errors = 0

    for i, draw_date in enumerate(draw_dates):
        draw_date_str = draw_date.strftime('%Y-%m-%d')

        try:
            # Fetch draw data
            draw_data = scraper.fetch_draw_by_date(draw_date_str)

            if draw_data:
                # Validate
                if not scraper.validate_draw_data(draw_data):
                    errors += 1
                    print(f"✗ Validation failed for {draw_date_str}")
                    continue

                # Insert into database
                result = db.insert_draw(draw_data)
                if result > 0:
                    collected += 1
                    if collected % 50 == 0:
                        print(f"✓ Collected {collected} draws...")
                elif result == -1:
                    skipped += 1
                else:
                    errors += 1
            else:
                errors += 1
                print(f"✗ Could not fetch data for {draw_date_str}")

        except Exception as e:
            errors += 1
            print(f"✗ Exception for {draw_date_str}: {e}")

    print()
    print("=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    print(f"Collected: {collected}")
    print(f"Skipped:   {skipped} (already exist)")
    print(f"Errors:    {errors}")
    print()

    # Verify data
    stats = db.get_database_stats()
    print("=" * 70)
    print("DATABASE STATISTICS")
    print("=" * 70)
    print(f"Total draws:     {stats['total_draws']}")
    print(f"Date range:      {stats['earliest_draw']} to {stats['latest_draw']}")
    print(f"Unique numbers:  {stats['unique_numbers']}")
    print()
    print("✅ Data collection complete!")


if __name__ == "__main__":
    main()
