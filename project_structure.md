# TOTO ML Project Structure

## Directory Layout

```
toto-ml-prediction/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── .env.example
│
├── config/
│   ├── __init__.py
│   ├── training_config.yaml
│   ├── data_config.yaml
│   └── deployment_config.yaml
│
├── data/
│   ├── raw/                    # Raw scraped data
│   ├── processed/              # Processed features
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   └── test.parquet
│   └── external/               # External data (holidays, etc.)
│
├── database/
│   ├── __init__.py
│   ├── schema.sql
│   ├── migrations/
│   └── db_manager.py
│
├── pipeline/
│   ├── __init__.py
│   ├── scraper.py              # Web scraping
│   ├── data_validator.py       # Data validation
│   ├── data_collector.py       # Data collection orchestration
│   ├── feature_engineer.py     # Feature engineering
│   ├── data_loader.py          # ML data preparation
│   ├── data_quality.py         # Quality monitoring
│   └── main_pipeline.py        # Pipeline orchestration
│
├── models/
│   ├── __init__.py
│   ├── dnn_model.py            # Deep Neural Network
│   ├── lstm_model.py           # LSTM with Attention
│   ├── lightgbm_model.py       # LightGBM
│   ├── random_forest_model.py  # Random Forest
│   ├── transformer_model.py    # Transformer
│   ├── ensemble.py             # Ensemble strategies
│   └── base_model.py           # Base model interface
│
├── training/
│   ├── __init__.py
│   ├── train_all_models.py     # Main training script
│   ├── hyperparameter_tuning.py
│   ├── cross_validation.py
│   └── callbacks.py            # Training callbacks
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # Evaluation metrics
│   ├── reward_calculator.py    # Reward functions
│   ├── backtesting.py          # Backtesting framework
│   ├── statistical_tests.py    # Significance testing
│   └── visualizations.py       # Result visualization
│
├── inference/
│   ├── __init__.py
│   ├── predictor.py            # Prediction service
│   └── model_loader.py         # Model loading utilities
│
├── api/
│   ├── __init__.py
│   ├── app.py                  # FastAPI application
│   ├── routes.py               # API routes
│   └── schemas.py              # Pydantic schemas
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_results_analysis.ipynb
│
├── scripts/
│   ├── setup_database.sh
│   ├── collect_historical_data.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── deploy.sh
│
├── tests/
│   ├── __init__.py
│   ├── test_pipeline/
│   ├── test_models/
│   ├── test_evaluation/
│   └── test_api/
│
├── logs/
│   ├── pipeline.log
│   ├── training.log
│   └── api.log
│
├── models/
│   └── checkpoints/            # Saved model weights
│       ├── dnn_best.pt
│       ├── lstm_best.pt
│       ├── lgb_best.pkl
│       ├── rf_best.pkl
│       └── transformer_best.pt
│
├── docs/
│   ├── toto_ml_prediction_plan.md
│   ├── data_pipeline_design.md
│   ├── ml_algorithms_implementation.md
│   ├── api_documentation.md
│   └── deployment_guide.md
│
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── .dockerignore
```

---

## Requirements

### requirements.txt

```txt
# Core ML/DL
torch==2.1.0
torchvision==0.16.0
tensorflow==2.14.0
scikit-learn==1.3.2
xgboost==2.0.3
lightgbm==4.1.0

# Data Processing
pandas==2.1.3
numpy==1.26.2
polars==0.19.19
scipy==1.11.4

# Feature Engineering
feature-engine==1.6.2
category-encoders==2.6.3
tsfresh==0.20.2

# Web Scraping
requests==2.31.0
beautifulsoup4==4.12.2
selenium==4.15.2
lxml==4.9.3

# Database
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
sqlite3

# Hyperparameter Optimization
optuna==3.4.0
hyperopt==0.2.7

# Experiment Tracking
mlflow==2.8.1
wandb==0.16.0

# API & Deployment
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Monitoring
prometheus-client==0.19.0

# Visualization
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0

# Utilities
pyyaml==6.0.1
python-dotenv==1.0.0
click==8.1.7
tqdm==4.66.1
joblib==1.3.2

# Scheduling
schedule==1.2.0
apscheduler==3.10.4

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1

# Code Quality
black==23.11.0
flake8==6.1.0
mypy==1.7.1
isort==5.12.0

# Documentation
sphinx==7.2.6
sphinx-rtd-theme==2.0.0
```

### requirements-dev.txt

```txt
# Development Tools
ipython==8.18.1
jupyter==1.0.0
jupyterlab==4.0.9
notebook==7.0.6

# Profiling
py-spy==0.3.14
memory-profiler==0.61.0

# Debugging
ipdb==0.13.13
```

---

## Setup Instructions

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/toto-ml-prediction.git
cd toto-ml-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

### 2. Environment Variables

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/toto_ml
# or
DATABASE_URL=sqlite:///data/toto.db

# API
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=your-secret-key-here

# ML Training
DEVICE=cuda  # or 'cpu'
MODEL_DIR=models/checkpoints
LOG_LEVEL=INFO

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=toto-prediction

# Weights & Biases
WANDB_API_KEY=your-wandb-api-key
WANDB_PROJECT=toto-ml
```

### 3. Database Setup

```bash
# Initialize database
python scripts/setup_database.py

# Or using shell script
chmod +x scripts/setup_database.sh
./scripts/setup_database.sh
```

### 4. Data Collection

```bash
# Collect historical data
python scripts/collect_historical_data.py --start-date 1968-06-09 --end-date 2025-11-12

# Or run the pipeline
python pipeline/main_pipeline.py --mode initial_setup
```

### 5. Train Models

```bash
# Train all 5 models
python scripts/train_models.py --config config/training_config.yaml

# Or train specific model
python scripts/train_models.py --model dnn --config config/training_config.yaml
```

### 6. Evaluate Models

```bash
# Evaluate all models
python scripts/evaluate_models.py --test-data data/processed/test.parquet

# Backtesting
python scripts/evaluate_models.py --mode backtest --lookback 100
```

### 7. API Deployment

```bash
# Development server
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# Production (with Gunicorn)
gunicorn api.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install package
RUN pip install -e .

# Expose API port
EXPOSE 8000

# Run API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: toto_ml
      POSTGRES_USER: toto_user
      POSTGRES_PASSWORD: toto_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://toto_user:toto_password@postgres:5432/toto_ml
      DEVICE: cpu
    depends_on:
      - postgres
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.8.1
    ports:
      - "5000:5000"
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql://toto_user:toto_password@postgres:5432/mlflow
      MLFLOW_ARTIFACT_ROOT: /mlflow/artifacts
    depends_on:
      - postgres
    volumes:
      - mlflow_data:/mlflow
    command: mlflow server --host 0.0.0.0 --port 5000

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  postgres_data:
  mlflow_data:
  prometheus_data:
  grafana_data:
```

---

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=. --cov-report=html

# Specific test suite
pytest tests/test_models/

# Verbose mode
pytest tests/ -v
```

---

## Code Quality

```bash
# Format code
black .

# Sort imports
isort .

# Lint
flake8 .

# Type checking
mypy .

# All at once
black . && isort . && flake8 . && mypy .
```

---

## Usage Examples

### Python API

```python
from pipeline.main_pipeline import TotoPipeline
from training.train_all_models import TotoModelTrainer
from evaluation.backtesting import Backtester
from inference.predictor import TotoPredictor

# 1. Collect data
pipeline = TotoPipeline(config)
pipeline.run_initial_setup()

# 2. Train models
trainer = TotoModelTrainer(config)
models = trainer.train_all_models(train_data, val_data)

# 3. Evaluate
backtester = Backtester(models)
results = backtester.run_backtest(test_data)

# 4. Make predictions
predictor = TotoPredictor(models)
predictions = predictor.predict_next_draw()
print(f"Predicted numbers: {predictions}")
```

### REST API

```bash
# Get prediction for next draw
curl -X GET http://localhost:8000/api/v1/predict

# Get historical performance
curl -X GET http://localhost:8000/api/v1/performance

# Get model metrics
curl -X GET http://localhost:8000/api/v1/metrics
```

---

## Monitoring & Logging

### Logging Configuration

```python
# logging_config.py

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'detailed',
            'filename': 'logs/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file'],
    },
}
```

---

## Continuous Integration

### .github/workflows/ci.yml

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Lint with flake8
      run: flake8 .

    - name: Type check with mypy
      run: mypy .

    - name: Test with pytest
      run: |
        pytest tests/ --cov=. --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

---

## Performance Optimization

### GPU Acceleration

```python
# Ensure CUDA is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Parallel Data Loading

```python
# DataLoader with multiple workers
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=4,  # Parallel loading
    pin_memory=True,  # Faster GPU transfer
    prefetch_factor=2
)
```

---

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Slow Training**
   - Enable GPU acceleration
   - Increase num_workers in DataLoader
   - Use batch processing

3. **Poor Model Performance**
   - Check data quality
   - Verify feature engineering
   - Tune hyperparameters
   - Ensemble multiple models

4. **Database Connection Issues**
   - Check DATABASE_URL
   - Verify database is running
   - Check firewall settings

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Disclaimer

This project is for **educational and research purposes only**. Singapore Pools TOTO is a game of chance, and outcomes cannot be reliably predicted. Do not use this system for actual gambling. Always gamble responsibly.

---

**Project Status:** Planning & Design Complete
**Next Phase:** Implementation
**Estimated Timeline:** 10 weeks
