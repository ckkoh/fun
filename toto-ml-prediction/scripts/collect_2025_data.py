"""
Collect TOTO data for January through June 2025.

This script collects mock TOTO draw data for the first half of 2025
to enable incremental learning evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from datetime import datetime, timedelta
from pipeline.scraper import TotoDataScraper
from database.db_manager import DatabaseManager


def main():
    print("=" * 70)
    print("COLLECTING 2025 DATA (JANUARY - JUNE)")
    print("=" * 70)
    print()

    db = DatabaseManager()
    scraper = TotoDataScraper(mock_mode=True)

    # Get latest draw to determine starting point
    latest_draw = db.get_latest_draw()
    if latest_draw:
        latest_number = latest_draw['draw_number']
        latest_date = datetime.strptime(latest_draw['draw_date'], '%Y-%m-%d')
        print(f"Latest draw: #{latest_number} on {latest_draw['draw_date']}")
    else:
        print("No existing draws found. Starting from scratch.")
        latest_number = 5000
        latest_date = datetime(2025, 1, 1)

    # Calculate draws needed from Feb to Jun 2025
    # TOTO draws Monday and Thursday (approximately 8-9 draws per month)
    # Feb-Jun = 5 months × 8 draws = ~40 draws

    start_date = datetime(2025, 2, 1)
    end_date = datetime(2025, 6, 30)

    # Generate draw schedule (Mondays and Thursdays)
    draw_dates = []
    current_date = start_date

    while current_date <= end_date:
        # 0=Monday, 3=Thursday
        if current_date.weekday() in [0, 3]:
            draw_dates.append(current_date)
        current_date += timedelta(days=1)

    print(f"Target period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Expected draws: {len(draw_dates)}")
    print()

    # Collect draws
    collected = 0
    skipped = 0
    errors = 0

    for i, draw_date in enumerate(draw_dates):
        draw_number = latest_number + 1 + i
        draw_date_str = draw_date.strftime('%Y-%m-%d')

        try:
            # Check if draw already exists
            existing = db.get_draw_by_date(draw_date_str)
            if existing:
                skipped += 1
                continue

            # Fetch draw data using scraper
            draw_data = scraper.fetch_draw_by_date(draw_date_str)

            if draw_data:
                # Transform winning_numbers list to individual number fields
                winning_nums = draw_data.get('winning_numbers', [])
                if len(winning_nums) == 6:
                    draw_data['number_1'] = winning_nums[0]
                    draw_data['number_2'] = winning_nums[1]
                    draw_data['number_3'] = winning_nums[2]
                    draw_data['number_4'] = winning_nums[3]
                    draw_data['number_5'] = winning_nums[4]
                    draw_data['number_6'] = winning_nums[5]

                    # Use draw number if not provided
                    if 'draw_number' not in draw_data:
                        draw_data['draw_number'] = draw_number

                    # Insert into database
                    result = db.insert_draw(draw_data)
                    if result > 0:
                        collected += 1
                        if collected % 10 == 0:
                            print(f"✓ Collected {collected} draws...")
                    else:
                        skipped += 1
                else:
                    errors += 1
                    print(f"✗ Invalid winning numbers for {draw_date_str}")
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
    all_2025_draws = [d for d in db.get_all_draws() if d['draw_date'] >= '2025-01-01' and d['draw_date'] <= '2025-06-30']
    print(f"Total 2025 draws in database (Jan-Jun): {len(all_2025_draws)}")
    print(f"Date range: {all_2025_draws[0]['draw_date']} to {all_2025_draws[-1]['draw_date']}")
    print()
    print("✅ Data collection complete!")


if __name__ == "__main__":
    main()
