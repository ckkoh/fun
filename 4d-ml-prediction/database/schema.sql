-- Singapore Pools 4D Database Schema

-- Main draws table
CREATE TABLE IF NOT EXISTS four_d_draws (
    draw_id INTEGER PRIMARY KEY AUTOINCREMENT,
    draw_number INTEGER UNIQUE NOT NULL,
    draw_date DATE NOT NULL,
    day_of_week VARCHAR(10) NOT NULL,

    -- Top 3 prizes
    first_prize VARCHAR(4) NOT NULL CHECK (length(first_prize) = 4),
    second_prize VARCHAR(4) NOT NULL CHECK (length(second_prize) = 4),
    third_prize VARCHAR(4) NOT NULL CHECK (length(third_prize) = 4),

    -- Starter prizes (10 numbers)
    starter_1 VARCHAR(4) CHECK (starter_1 IS NULL OR length(starter_1) = 4),
    starter_2 VARCHAR(4) CHECK (starter_2 IS NULL OR length(starter_2) = 4),
    starter_3 VARCHAR(4) CHECK (starter_3 IS NULL OR length(starter_3) = 4),
    starter_4 VARCHAR(4) CHECK (starter_4 IS NULL OR length(starter_4) = 4),
    starter_5 VARCHAR(4) CHECK (starter_5 IS NULL OR length(starter_5) = 4),
    starter_6 VARCHAR(4) CHECK (starter_6 IS NULL OR length(starter_6) = 4),
    starter_7 VARCHAR(4) CHECK (starter_7 IS NULL OR length(starter_7) = 4),
    starter_8 VARCHAR(4) CHECK (starter_8 IS NULL OR length(starter_8) = 4),
    starter_9 VARCHAR(4) CHECK (starter_9 IS NULL OR length(starter_9) = 4),
    starter_10 VARCHAR(4) CHECK (starter_10 IS NULL OR length(starter_10) = 4),

    -- Consolation prizes (10 numbers)
    consolation_1 VARCHAR(4) CHECK (consolation_1 IS NULL OR length(consolation_1) = 4),
    consolation_2 VARCHAR(4) CHECK (consolation_2 IS NULL OR length(consolation_2) = 4),
    consolation_3 VARCHAR(4) CHECK (consolation_3 IS NULL OR length(consolation_3) = 4),
    consolation_4 VARCHAR(4) CHECK (consolation_4 IS NULL OR length(consolation_4) = 4),
    consolation_5 VARCHAR(4) CHECK (consolation_5 IS NULL OR length(consolation_5) = 4),
    consolation_6 VARCHAR(4) CHECK (consolation_6 IS NULL OR length(consolation_6) = 4),
    consolation_7 VARCHAR(4) CHECK (consolation_7 IS NULL OR length(consolation_7) = 4),
    consolation_8 VARCHAR(4) CHECK (consolation_8 IS NULL OR length(consolation_8) = 4),
    consolation_9 VARCHAR(4) CHECK (consolation_9 IS NULL OR length(consolation_9) = 4),
    consolation_10 VARCHAR(4) CHECK (consolation_10 IS NULL OR length(consolation_10) = 4),

    draw_type VARCHAR(20) DEFAULT 'normal',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_date CHECK (draw_date IS NOT NULL),
    CONSTRAINT valid_day CHECK (day_of_week IN ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
);

-- Number history table for frequency analysis
CREATE TABLE IF NOT EXISTS four_d_number_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    draw_id INTEGER NOT NULL,
    number VARCHAR(4) NOT NULL,
    prize_category VARCHAR(20) NOT NULL,  -- 'first', 'second', 'third', 'starter', 'consolation'
    position INTEGER,  -- Position in category (1-10 for starter/consolation, NULL for top 3)

    FOREIGN KEY (draw_id) REFERENCES four_d_draws(draw_id) ON DELETE CASCADE,
    CONSTRAINT valid_category CHECK (prize_category IN ('first', 'second', 'third', 'starter', 'consolation')),
    CONSTRAINT valid_position CHECK (position IS NULL OR (position >= 1 AND position <= 10))
);

-- Digit frequency tracking per position
CREATE TABLE IF NOT EXISTS digit_frequency (
    digit INTEGER NOT NULL CHECK (digit >= 0 AND digit <= 9),
    position INTEGER NOT NULL CHECK (position >= 0 AND position <= 3),  -- 0=thousands, 1=hundreds, 2=tens, 3=ones
    frequency INTEGER DEFAULT 0,
    last_appearance_draw_id INTEGER,
    last_appearance_date DATE,

    PRIMARY KEY (digit, position),
    FOREIGN KEY (last_appearance_draw_id) REFERENCES four_d_draws(draw_id) ON DELETE SET NULL
);

-- 4D number frequency (full 4-digit numbers)
CREATE TABLE IF NOT EXISTS number_frequency (
    number VARCHAR(4) PRIMARY KEY CHECK (length(number) = 4),
    total_appearances INTEGER DEFAULT 0,
    first_prize_count INTEGER DEFAULT 0,
    second_prize_count INTEGER DEFAULT 0,
    third_prize_count INTEGER DEFAULT 0,
    starter_count INTEGER DEFAULT 0,
    consolation_count INTEGER DEFAULT 0,
    last_appearance_date DATE,
    last_appearance_draw_id INTEGER,

    FOREIGN KEY (last_appearance_draw_id) REFERENCES four_d_draws(draw_id) ON DELETE SET NULL
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_four_d_draws_date ON four_d_draws(draw_date);
CREATE INDEX IF NOT EXISTS idx_four_d_draws_day ON four_d_draws(day_of_week);
CREATE INDEX IF NOT EXISTS idx_four_d_draws_number ON four_d_draws(draw_number);

CREATE INDEX IF NOT EXISTS idx_number_history_draw ON four_d_number_history(draw_id);
CREATE INDEX IF NOT EXISTS idx_number_history_number ON four_d_number_history(number);
CREATE INDEX IF NOT EXISTS idx_number_history_category ON four_d_number_history(prize_category);

CREATE INDEX IF NOT EXISTS idx_digit_freq_digit ON digit_frequency(digit);
CREATE INDEX IF NOT EXISTS idx_digit_freq_position ON digit_frequency(position);

CREATE INDEX IF NOT EXISTS idx_number_freq_number ON number_frequency(number);
CREATE INDEX IF NOT EXISTS idx_number_freq_appearances ON number_frequency(total_appearances);
