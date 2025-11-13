"""Database manager for 4D ML prediction system."""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class FourDDatabaseManager:
    """Manages database operations for 4D draws and predictions."""

    def __init__(self, db_path: str = "data/4d.db"):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._initialize_database()

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        schema_path = Path(__file__).parent / "schema.sql"

        with open(schema_path, "r") as f:
            schema_sql = f.read()

        conn = self.get_connection()
        try:
            conn.executescript(schema_sql)
            conn.commit()
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
        finally:
            conn.close()

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    def insert_draw(self, draw_data: Dict) -> int:
        """
        Insert a new draw into the database.

        Args:
            draw_data: Dictionary containing draw information

        Returns:
            ID of inserted draw, or -1 if already exists
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Check if draw already exists
            cursor.execute(
                "SELECT draw_id FROM 4d_draws WHERE draw_number = ?",
                (draw_data['draw_number'],)
            )
            if cursor.fetchone():
                self.logger.info(f"Draw {draw_data['draw_number']} already exists")
                return -1

            # Insert main draw data
            cursor.execute("""
                INSERT INTO 4d_draws (
                    draw_number, draw_date, day_of_week,
                    first_prize, second_prize, third_prize,
                    starter_1, starter_2, starter_3, starter_4, starter_5,
                    starter_6, starter_7, starter_8, starter_9, starter_10,
                    consolation_1, consolation_2, consolation_3, consolation_4, consolation_5,
                    consolation_6, consolation_7, consolation_8, consolation_9, consolation_10,
                    draw_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                draw_data['draw_number'],
                draw_data['draw_date'],
                draw_data['day_of_week'],
                draw_data['first_prize'],
                draw_data['second_prize'],
                draw_data['third_prize'],
                draw_data.get('starter_1'),
                draw_data.get('starter_2'),
                draw_data.get('starter_3'),
                draw_data.get('starter_4'),
                draw_data.get('starter_5'),
                draw_data.get('starter_6'),
                draw_data.get('starter_7'),
                draw_data.get('starter_8'),
                draw_data.get('starter_9'),
                draw_data.get('starter_10'),
                draw_data.get('consolation_1'),
                draw_data.get('consolation_2'),
                draw_data.get('consolation_3'),
                draw_data.get('consolation_4'),
                draw_data.get('consolation_5'),
                draw_data.get('consolation_6'),
                draw_data.get('consolation_7'),
                draw_data.get('consolation_8'),
                draw_data.get('consolation_9'),
                draw_data.get('consolation_10'),
                draw_data.get('draw_type', 'normal')
            ))

            draw_id = cursor.lastrowid

            # Insert into number_history table
            numbers_to_insert = [
                (draw_id, draw_data['first_prize'], 'first', None),
                (draw_id, draw_data['second_prize'], 'second', None),
                (draw_id, draw_data['third_prize'], 'third', None),
            ]

            # Add starter prizes
            for i in range(1, 11):
                key = f'starter_{i}'
                if key in draw_data and draw_data[key]:
                    numbers_to_insert.append((draw_id, draw_data[key], 'starter', i))

            # Add consolation prizes
            for i in range(1, 11):
                key = f'consolation_{i}'
                if key in draw_data and draw_data[key]:
                    numbers_to_insert.append((draw_id, draw_data[key], 'consolation', i))

            cursor.executemany("""
                INSERT INTO 4d_number_history (draw_id, number, prize_category, position)
                VALUES (?, ?, ?, ?)
            """, numbers_to_insert)

            # Update digit frequency
            self._update_digit_frequency(cursor, draw_id, draw_data)

            # Update number frequency
            self._update_number_frequency(cursor, draw_id, draw_data)

            conn.commit()
            self.logger.info(f"Inserted draw {draw_data['draw_number']} with {len(numbers_to_insert)} numbers")
            return draw_id

        except sqlite3.IntegrityError as e:
            conn.rollback()
            self.logger.error(f"Draw {draw_data['draw_number']} already exists: {e}")
            return -1
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error inserting draw: {e}")
            raise
        finally:
            conn.close()

    def _update_digit_frequency(self, cursor, draw_id: int, draw_data: Dict):
        """Update digit frequency table."""
        # Get all winning numbers
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

        # Update frequency for each digit position
        for number in all_numbers:
            for pos, digit in enumerate(number):
                cursor.execute("""
                    INSERT INTO digit_frequency (digit, position, frequency, last_appearance_draw_id, last_appearance_date)
                    VALUES (?, ?, 1, ?, ?)
                    ON CONFLICT(digit, position) DO UPDATE SET
                        frequency = frequency + 1,
                        last_appearance_draw_id = ?,
                        last_appearance_date = ?
                """, (int(digit), pos, draw_id, draw_data['draw_date'], draw_id, draw_data['draw_date']))

    def _update_number_frequency(self, cursor, draw_id: int, draw_data: Dict):
        """Update number frequency table."""
        # Update for each prize category
        categories = [
            ('first_prize', 'first_prize_count'),
            ('second_prize', 'second_prize_count'),
            ('third_prize', 'third_prize_count'),
        ]

        for field, count_field in categories:
            number = draw_data[field]
            cursor.execute(f"""
                INSERT INTO number_frequency (number, total_appearances, {count_field}, last_appearance_date, last_appearance_draw_id)
                VALUES (?, 1, 1, ?, ?)
                ON CONFLICT(number) DO UPDATE SET
                    total_appearances = total_appearances + 1,
                    {count_field} = {count_field} + 1,
                    last_appearance_date = ?,
                    last_appearance_draw_id = ?
            """, (number, draw_data['draw_date'], draw_id, draw_data['draw_date'], draw_id))

        # Update starter numbers
        for i in range(1, 11):
            key = f'starter_{i}'
            if key in draw_data and draw_data[key]:
                number = draw_data[key]
                cursor.execute("""
                    INSERT INTO number_frequency (number, total_appearances, starter_count, last_appearance_date, last_appearance_draw_id)
                    VALUES (?, 1, 1, ?, ?)
                    ON CONFLICT(number) DO UPDATE SET
                        total_appearances = total_appearances + 1,
                        starter_count = starter_count + 1,
                        last_appearance_date = ?,
                        last_appearance_draw_id = ?
                """, (number, draw_data['draw_date'], draw_id, draw_data['draw_date'], draw_id))

        # Update consolation numbers
        for i in range(1, 11):
            key = f'consolation_{i}'
            if key in draw_data and draw_data[key]:
                number = draw_data[key]
                cursor.execute("""
                    INSERT INTO number_frequency (number, total_appearances, consolation_count, last_appearance_date, last_appearance_draw_id)
                    VALUES (?, 1, 1, ?, ?)
                    ON CONFLICT(number) DO UPDATE SET
                        total_appearances = total_appearances + 1,
                        consolation_count = consolation_count + 1,
                        last_appearance_date = ?,
                        last_appearance_draw_id = ?
                """, (number, draw_data['draw_date'], draw_id, draw_data['draw_date'], draw_id))

    def get_all_draws(self, limit: Optional[int] = None) -> List[Dict]:
        """Get all draws from database."""
        conn = self.get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM 4d_draws ORDER BY draw_date ASC"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_draws_in_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Get draws within a date range."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM 4d_draws
            WHERE draw_date BETWEEN ? AND ?
            ORDER BY draw_date ASC
        """, (start_date, end_date))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_latest_draw(self) -> Optional[Dict]:
        """Get the most recent draw."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM 4d_draws
            ORDER BY draw_date DESC, draw_number DESC
            LIMIT 1
        """)

        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def get_digit_frequency(self, position: int, last_n_draws: Optional[int] = None) -> Dict[int, int]:
        """Get digit frequency for a specific position."""
        conn = self.get_connection()
        cursor = conn.cursor()

        if last_n_draws:
            # Get digits from last N draws
            cursor.execute("""
                SELECT digit, COUNT(*) as count
                FROM digit_frequency df
                JOIN 4d_draws d ON df.last_appearance_draw_id = d.draw_id
                WHERE df.position = ?
                AND d.draw_id IN (
                    SELECT draw_id FROM 4d_draws
                    ORDER BY draw_date DESC
                    LIMIT ?
                )
                GROUP BY digit
            """, (position, last_n_draws))
        else:
            cursor.execute("""
                SELECT digit, frequency as count
                FROM digit_frequency
                WHERE position = ?
            """, (position,))

        rows = cursor.fetchall()
        conn.close()

        return {row['digit']: row['count'] for row in rows}

    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM 4d_draws")
        total_draws = cursor.fetchone()['count']

        cursor.execute("SELECT MIN(draw_date) as min_date, MAX(draw_date) as max_date FROM 4d_draws")
        date_range = cursor.fetchone()

        cursor.execute("SELECT COUNT(DISTINCT number) as count FROM 4d_number_history")
        unique_numbers = cursor.fetchone()['count']

        conn.close()

        return {
            'total_draws': total_draws,
            'earliest_draw': date_range['min_date'],
            'latest_draw': date_range['max_date'],
            'unique_numbers': unique_numbers
        }
