-- TOTO ML Prediction Database Schema

-- Main table for TOTO draws
CREATE TABLE IF NOT EXISTS toto_draws (
    draw_id INTEGER PRIMARY KEY AUTOINCREMENT,
    draw_number INTEGER UNIQUE NOT NULL,
    draw_date DATE NOT NULL,
    day_of_week VARCHAR(10) NOT NULL,

    -- Winning numbers (sorted ascending)
    number_1 INTEGER NOT NULL CHECK (number_1 BETWEEN 1 AND 49),
    number_2 INTEGER NOT NULL CHECK (number_2 BETWEEN 1 AND 49),
    number_3 INTEGER NOT NULL CHECK (number_3 BETWEEN 1 AND 49),
    number_4 INTEGER NOT NULL CHECK (number_4 BETWEEN 1 AND 49),
    number_5 INTEGER NOT NULL CHECK (number_5 BETWEEN 1 AND 49),
    number_6 INTEGER NOT NULL CHECK (number_6 BETWEEN 1 AND 49),
    additional_number INTEGER NOT NULL CHECK (additional_number BETWEEN 1 AND 49),

    -- Prize information
    prize_pool DECIMAL(12, 2),

    -- Winners per group
    group_1_winners INTEGER DEFAULT 0,
    group_2_winners INTEGER DEFAULT 0,
    group_3_winners INTEGER DEFAULT 0,
    group_4_winners INTEGER DEFAULT 0,

    -- Draw metadata
    draw_type VARCHAR(20) DEFAULT 'normal', -- normal/cascade/hongbao/special
    is_rollover BOOLEAN DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CHECK (number_1 < number_2),
    CHECK (number_2 < number_3),
    CHECK (number_3 < number_4),
    CHECK (number_4 < number_5),
    CHECK (number_5 < number_6),
    CHECK (day_of_week IN ('Monday', 'Thursday'))
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_draw_date ON toto_draws(draw_date DESC);
CREATE INDEX IF NOT EXISTS idx_draw_number ON toto_draws(draw_number DESC);
CREATE INDEX IF NOT EXISTS idx_draw_type ON toto_draws(draw_type);

-- Table for individual number history (denormalized for fast queries)
CREATE TABLE IF NOT EXISTS number_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    draw_id INTEGER NOT NULL,
    number INTEGER NOT NULL CHECK (number BETWEEN 1 AND 49),
    is_additional BOOLEAN DEFAULT 0,
    position INTEGER CHECK (position BETWEEN 1 AND 6), -- NULL if additional

    FOREIGN KEY (draw_id) REFERENCES toto_draws(draw_id) ON DELETE CASCADE,
    UNIQUE(draw_id, number)
);

CREATE INDEX IF NOT EXISTS idx_number_history_number ON number_history(number, draw_id DESC);
CREATE INDEX IF NOT EXISTS idx_number_history_draw ON number_history(draw_id);

-- Table for model predictions (for tracking)
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    draw_number INTEGER NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    predicted_numbers TEXT NOT NULL, -- JSON array of 6 numbers
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Actual results (filled after draw)
    actual_numbers TEXT,
    matches INTEGER,
    reward DECIMAL(12, 2),

    FOREIGN KEY (draw_number) REFERENCES toto_draws(draw_number)
);

CREATE INDEX IF NOT EXISTS idx_predictions_draw ON predictions(draw_number);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(predicted_at DESC);
