# TOTO ML Prediction System

A comprehensive machine learning system for exploring pattern recognition in Singapore Pools TOTO lottery draws.

⚠️ **IMPORTANT DISCLAIMER**: This is an **educational and research project only**. Lottery draws are random by design and cannot be reliably predicted. Do not use this system for actual gambling. Always gamble responsibly.

## Project Overview

This system implements 5 different machine learning algorithms to explore patterns in TOTO draw data:

1. **Deep Neural Network (DNN)** with number embeddings
2. **LSTM with Attention** for sequential pattern recognition
3. **LightGBM** gradient boosting with 49 binary classifiers
4. **Random Forest** ensemble learning
5. **Transformer** with self-attention mechanism

## Features

- ✅ Complete data collection pipeline with web scraping
- ✅ Comprehensive data validation
- ✅ SQLite/PostgreSQL database support
- ✅ 200+ engineered features (temporal, statistical, patterns)
- ✅ Reward structure aligned with prize tiers
- ✅ Time-series cross-validation
- ✅ Ensemble strategies (weighted average, voting, stacking)
- ✅ Experiment tracking (MLflow, W&B)
- ✅ REST API for predictions
- ✅ Docker deployment support

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/toto-ml-prediction.git
cd toto-ml-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

### Initialize Database

```bash
# The database will be automatically created on first run
python -c "from database.db_manager import DatabaseManager; DatabaseManager('data/toto.db')"
```

### Collect Data

```bash
# Test with mock data (recommended for development)
python pipeline/data_collector.py

# For real data collection (requires proper scraper implementation)
# Edit config/data_config.yaml and set mock_mode: false
```

### Train Models

```bash
# Train all 5 models
python scripts/train_models.py --config config/training_config.yaml

# Train specific model
python scripts/train_models.py --model dnn
```

### Make Predictions

```bash
# Predict next draw
python scripts/predict.py --ensemble

# API mode
uvicorn api.app:app --reload
```

## Project Structure

```
toto-ml-prediction/
├── config/              # Configuration files
├── data/                # Data storage
├── database/            # Database schema and manager
├── pipeline/            # Data collection and preprocessing
├── models/              # ML model implementations
├── training/            # Training scripts
├── evaluation/          # Evaluation metrics and backtesting
├── inference/           # Prediction service
├── api/                 # REST API
├── notebooks/           # Jupyter notebooks for analysis
├── scripts/             # Utility scripts
├── tests/               # Unit and integration tests
└── docs/                # Planning and documentation
```

## Development Status

### ✅ Phase 1: Data Infrastructure (Complete)
- [x] Database schema and manager
- [x] Web scraper with mock mode
- [x] Data validator
- [x] Data collector orchestration
- [x] Configuration files

### 🔄 Phase 2: Feature Engineering (In Progress)
- [ ] Feature engineering pipeline
- [ ] Data loader for ML models
- [ ] Feature selection

### 📋 Phase 3: Model Implementation (Planned)
- [ ] DNN with embeddings
- [ ] LSTM with attention
- [ ] LightGBM
- [ ] Random Forest
- [ ] Transformer
- [ ] Ensemble strategies

### 📋 Phase 4: Training & Evaluation (Planned)
- [ ] Training pipeline
- [ ] Hyperparameter tuning
- [ ] Cross-validation
- [ ] Backtesting framework
- [ ] Performance metrics

### 📋 Phase 5: Deployment (Planned)
- [ ] REST API
- [ ] Docker containerization
- [ ] Monitoring dashboard
- [ ] Documentation

## Documentation

Comprehensive planning documents are available in the `docs/` directory:

- [Main Planning Document](docs/toto_ml_prediction_plan.md)
- [Data Pipeline Design](docs/data_pipeline_design.md)
- [ML Algorithms Implementation](docs/ml_algorithms_implementation.md)
- [Project Structure](docs/project_structure.md)

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test suite
pytest tests/test_pipeline/
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access API
curl http://localhost:8000/api/v1/predict
```

## Performance Expectations

### Realistic Baseline
- **Random selection**: ~0.7347 matches per draw
- **Expected performance**: Approximately equal to random baseline
- **Best case scenario**: Marginal improvement (0.05-0.1 matches)

### Why This Project?
Despite limited predictive power, this project offers:
- Complete ML pipeline implementation
- Multiple algorithm comparison
- Rigorous evaluation methodology
- Educational value for ML/DL techniques

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE) file for details

## Acknowledgments

- Singapore Pools for TOTO game data
- Open source ML/DL community

## Disclaimer

**This project is for educational and research purposes only.**

- Lottery draws are designed to be random
- Past results do not predict future outcomes
- Do not use this system for actual gambling
- The authors take no responsibility for any gambling losses
- Always gamble responsibly

## Contact

For questions or issues, please open a GitHub issue.

---

**Project Status**: Development - Phase 1 Complete
**Last Updated**: 2025-11-12
**Python Version**: 3.9+
