"""Database manager for TOTO ML prediction system."""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


class DatabaseManager:
    """Manages database operations for TOTO draws and predictions."""

    def __init__(self, db_path: str = "data/toto.db"):
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
            ID of inserted draw
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Insert main draw data
            cursor.execute("""
                INSERT INTO toto_draws (
                    draw_number, draw_date, day_of_week,
                    number_1, number_2, number_3, number_4, number_5, number_6,
                    additional_number, prize_pool,
                    group_1_winners, group_2_winners, group_3_winners, group_4_winners,
                    draw_type, is_rollover
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                draw_data['draw_number'],
                draw_data['draw_date'],
                draw_data['day_of_week'],
                draw_data['number_1'],
                draw_data['number_2'],
                draw_data['number_3'],
                draw_data['number_4'],
                draw_data['number_5'],
                draw_data['number_6'],
                draw_data['additional_number'],
                draw_data.get('prize_pool'),
                draw_data.get('group_1_winners', 0),
                draw_data.get('group_2_winners', 0),
                draw_data.get('group_3_winners', 0),
                draw_data.get('group_4_winners', 0),
                draw_data.get('draw_type', 'normal'),
                draw_data.get('is_rollover', False)
            ))

            draw_id = cursor.lastrowid

            # Insert into number_history table
            numbers = [
                draw_data['number_1'], draw_data['number_2'], draw_data['number_3'],
                draw_data['number_4'], draw_data['number_5'], draw_data['number_6']
            ]

            for position, number in enumerate(numbers, 1):
                cursor.execute("""
                    INSERT INTO number_history (draw_id, number, is_additional, position)
                    VALUES (?, ?, 0, ?)
                """, (draw_id, number, position))

            # Insert additional number
            cursor.execute("""
                INSERT INTO number_history (draw_id, number, is_additional, position)
                VALUES (?, ?, 1, NULL)
            """, (draw_id, draw_data['additional_number']))

            conn.commit()
            self.logger.info(f"Inserted draw {draw_data['draw_number']} successfully")

            return draw_id

        except sqlite3.IntegrityError as e:
            conn.rollback()
            self.logger.warning(f"Draw {draw_data['draw_number']} already exists: {e}")
            return -1
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error inserting draw: {e}")
            raise
        finally:
            conn.close()

    def get_draw_by_number(self, draw_number: int) -> Optional[Dict]:
        """Get draw data by draw number."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM toto_draws WHERE draw_number = ?
            """, (draw_number,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

        finally:
            conn.close()

    def get_draw_by_date(self, draw_date: str) -> Optional[Dict]:
        """Get draw data by date."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM toto_draws WHERE draw_date = ?
            """, (draw_date,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

        finally:
            conn.close()

    def get_latest_draw(self) -> Optional[Dict]:
        """Get the most recent draw."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM toto_draws
                ORDER BY draw_date DESC
                LIMIT 1
            """)

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

        finally:
            conn.close()

    def get_all_draws(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all draws ordered by date.

        Args:
            limit: Optional limit on number of draws to return

        Returns:
            List of draw dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            if limit:
                cursor.execute("""
                    SELECT * FROM toto_draws
                    ORDER BY draw_date ASC
                    LIMIT ?
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT * FROM toto_draws
                    ORDER BY draw_date ASC
                """)

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        finally:
            conn.close()

    def get_draws_in_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Get all draws within a date range."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM toto_draws
                WHERE draw_date BETWEEN ? AND ?
                ORDER BY draw_date ASC
            """, (start_date, end_date))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        finally:
            conn.close()

    def get_number_frequency(self, number: int, last_n_draws: Optional[int] = None) -> int:
        """
        Get frequency of a number appearing in draws.

        Args:
            number: Number to check (1-49)
            last_n_draws: Optional limit to last N draws

        Returns:
            Count of appearances
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            if last_n_draws:
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM number_history
                    WHERE number = ?
                    AND draw_id IN (
                        SELECT draw_id FROM toto_draws
                        ORDER BY draw_date DESC
                        LIMIT ?
                    )
                """, (number, last_n_draws))
            else:
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM number_history
                    WHERE number = ?
                """, (number,))

            result = cursor.fetchone()
            return result['count'] if result else 0

        finally:
            conn.close()

    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Total draws
            cursor.execute("SELECT COUNT(*) as count FROM toto_draws")
            total_draws = cursor.fetchone()['count']

            # Date range
            cursor.execute("""
                SELECT MIN(draw_date) as first_draw, MAX(draw_date) as last_draw
                FROM toto_draws
            """)
            date_range = cursor.fetchone()

            # Number frequency stats
            cursor.execute("""
                SELECT number, COUNT(*) as frequency
                FROM number_history
                WHERE is_additional = 0
                GROUP BY number
                ORDER BY frequency DESC
                LIMIT 5
            """)
            most_frequent = cursor.fetchall()

            return {
                'total_draws': total_draws,
                'first_draw': date_range['first_draw'],
                'last_draw': date_range['last_draw'],
                'most_frequent_numbers': [
                    {'number': row['number'], 'frequency': row['frequency']}
                    for row in most_frequent
                ]
            }

        finally:
            conn.close()

    def insert_prediction(
        self,
        draw_number: int,
        model_name: str,
        predicted_numbers: List[int]
    ) -> int:
        """
        Insert a model prediction.

        Args:
            draw_number: Draw number for prediction
            model_name: Name of the model
            predicted_numbers: List of 6 predicted numbers

        Returns:
            Prediction ID
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO predictions (draw_number, model_name, predicted_numbers)
                VALUES (?, ?, ?)
            """, (draw_number, model_name, json.dumps(predicted_numbers)))

            prediction_id = cursor.lastrowid
            conn.commit()

            self.logger.info(
                f"Inserted prediction for draw {draw_number} by {model_name}"
            )

            return prediction_id

        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error inserting prediction: {e}")
            raise
        finally:
            conn.close()

    def update_prediction_results(
        self,
        prediction_id: int,
        actual_numbers: List[int],
        matches: int,
        reward: float
    ):
        """Update prediction with actual results."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE predictions
                SET actual_numbers = ?, matches = ?, reward = ?
                WHERE prediction_id = ?
            """, (json.dumps(actual_numbers), matches, reward, prediction_id))

            conn.commit()
            self.logger.info(f"Updated prediction {prediction_id} with results")

        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error updating prediction results: {e}")
            raise
        finally:
            conn.close()

    def close(self):
        """Close database connection (for cleanup)."""
        self.logger.info("Database manager closed")


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create database manager
    db = DatabaseManager("data/toto.db")

    # Example: Insert a draw
    sample_draw = {
        'draw_number': 1,
        'draw_date': '2025-01-06',
        'day_of_week': 'Monday',
        'number_1': 5,
        'number_2': 12,
        'number_3': 18,
        'number_4': 24,
        'number_5': 35,
        'number_6': 42,
        'additional_number': 9,
        'prize_pool': 2500000.00,
        'group_1_winners': 1,
        'draw_type': 'normal'
    }

    db.insert_draw(sample_draw)

    # Get stats
    stats = db.get_database_stats()
    print(f"Database stats: {stats}")
