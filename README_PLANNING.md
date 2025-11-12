# TOTO ML Number Prediction - Planning Documentation

## Project Overview

This repository contains comprehensive planning documentation for a machine learning system designed to explore pattern recognition in Singapore Pools TOTO lottery draws. The project implements 5 different machine learning algorithms with proper reward structures aligned to prize tiers.

**⚠️ Important:** This is an **educational and research project**. Lottery draws are random by design, and this system should not be used for actual gambling purposes.

---

## 📋 Planning Documents

### 1. **Main Planning Document**
   📄 [`toto_ml_prediction_plan.md`](toto_ml_prediction_plan.md)

   **Contents:**
   - Executive summary with disclaimers
   - TOTO game structure analysis
   - Prize structure breakdown (7 groups)
   - Five ML algorithms specification
   - Detailed feature engineering strategy
   - Reward function design
   - Training and evaluation framework
   - Risk mitigation strategies
   - 10-week implementation roadmap

### 2. **Data Pipeline Design**
   📄 [`data_pipeline_design.md`](data_pipeline_design.md)

   **Contents:**
   - Web scraping architecture
   - Database schema (PostgreSQL/SQLite)
   - Data validation rules
   - Feature engineering pipeline
   - Data quality monitoring
   - Automated update scheduling
   - Complete code implementations

### 3. **ML Algorithms Implementation**
   📄 [`ml_algorithms_implementation.md`](ml_algorithms_implementation.md)

   **Contents:**
   - Detailed specifications for all 5 algorithms:
     1. Deep Neural Network (DNN) with Embeddings
     2. LSTM with Attention Mechanism
     3. LightGBM Gradient Boosting
     4. Random Forest
     5. Transformer
   - Complete PyTorch/Sklearn implementations
   - Ensemble strategies (averaging, voting, stacking)
   - Training pipeline orchestration

### 4. **Project Structure**
   📄 [`project_structure.md`](project_structure.md)

   **Contents:**
   - Complete directory layout
   - Requirements and dependencies
   - Setup instructions
   - Docker deployment configuration
   - Testing and CI/CD pipelines
   - API documentation
   - Monitoring and logging setup

---

## 🎯 Key Features

### Machine Learning Algorithms

1. **Deep Neural Network (DNN)**
   - Number embeddings (49-dimensional)
   - 3 hidden layers [512, 256, 128]
   - Dropout regularization (0.3)
   - Binary cross-entropy loss

2. **LSTM with Attention**
   - 2 LSTM layers [256, 128]
   - Self-attention mechanism
   - Sequence length: 50 draws
   - Temporal dependency modeling

3. **LightGBM**
   - 49 independent binary classifiers
   - 1000 estimators, max depth 8
   - Feature importance analysis
   - Efficient gradient boosting

4. **Random Forest**
   - 300 trees, max depth 15
   - Multi-output classification
   - Out-of-bag scoring
   - Bootstrap aggregation

5. **Transformer**
   - 4 transformer blocks
   - 8 attention heads
   - Positional encoding
   - Focal loss for class imbalance

### Feature Engineering

- **Temporal Features:** Cyclical encoding, draw sequences
- **Statistical Features:** Mean, std, skewness, kurtosis
- **Frequency Features:** Rolling windows (5, 10, 20, 50, 100 draws)
- **Pattern Features:** Co-occurrence, consecutive numbers
- **Gap Analysis:** Draws since last appearance

### Reward Structure

Aligned with Singapore Pools prize tiers:

| Group | Match | Reward (Training) | Real Prize |
|-------|-------|------------------|------------|
| 1 | 6 numbers | 1,000,000 | ~38% pool (Min $1M) |
| 2 | 5 + additional | 100,000 | ~8% pool |
| 3 | 5 numbers | 10,000 | ~5.5% pool |
| 4 | 4 numbers | 1,000 | ~3% pool |
| 5 | 3 + additional | 50 | $50 fixed |
| 6 | 3 numbers | 25 | $25 fixed |

---

## 📊 Data Structure

### TOTO Game Rules
- **Draw Frequency:** Every Monday & Thursday, 6:30 PM SGT
- **Number Pool:** 1-49
- **Selection:** 6 winning numbers + 1 additional number
- **Historical Data:** From 1968 to present (~2,800+ draws)

### Database Schema
```sql
CREATE TABLE toto_draws (
    draw_id INTEGER PRIMARY KEY,
    draw_date DATE NOT NULL,
    number_1 to number_6 INTEGER (1-49),
    additional_number INTEGER (1-49),
    prize_pool DECIMAL,
    group_1_winners INTEGER,
    draw_type VARCHAR
);
```

---

## 🚀 Implementation Roadmap

### Phase 1: Data Infrastructure (Week 1-2)
- [ ] Build web scraper
- [ ] Create database
- [ ] Collect historical data (1968-present)
- [ ] Data validation pipeline

### Phase 2: Feature Engineering (Week 3)
- [ ] Implement temporal features
- [ ] Statistical features
- [ ] Pattern detection
- [ ] Feature selection

### Phase 3: Algorithm Implementation (Week 4-6)
- [ ] DNN with embeddings
- [ ] LSTM with attention
- [ ] LightGBM
- [ ] Random Forest
- [ ] Transformer

### Phase 4: Training & Optimization (Week 7-8)
- [ ] Individual model training
- [ ] Hyperparameter tuning (Optuna)
- [ ] Cross-validation (time-series split)
- [ ] Model selection

### Phase 5: Ensemble & Evaluation (Week 9)
- [ ] Ensemble strategies
- [ ] Backtesting (100+ draws)
- [ ] Statistical significance tests
- [ ] Performance reporting

### Phase 6: Deployment (Week 10)
- [ ] REST API (FastAPI)
- [ ] Docker containerization
- [ ] Monitoring dashboard
- [ ] Automated retraining

---

## 🛠️ Technology Stack

### Core ML/DL
- **PyTorch 2.1.0** - Deep learning models
- **TensorFlow 2.14.0** - Alternative DL framework
- **Scikit-learn 1.3.2** - Classical ML
- **LightGBM 4.1.0** - Gradient boosting
- **XGBoost 2.0.3** - Alternative boosting

### Data Processing
- **Pandas 2.1.3** - Data manipulation
- **NumPy 1.26.2** - Numerical computing
- **Polars 0.19.19** - Fast data processing

### Optimization
- **Optuna 3.4.0** - Hyperparameter tuning
- **MLflow 2.8.1** - Experiment tracking

### Deployment
- **FastAPI 0.104.1** - REST API
- **Docker** - Containerization
- **PostgreSQL** - Production database
- **Prometheus + Grafana** - Monitoring

---

## 📈 Evaluation Metrics

### Primary Metrics
1. **Hit Rate by Group** - Achievement rate vs. baseline
2. **Average Matches per Draw** - Mean correct predictions
3. **Cumulative Reward** - Total reward over test period
4. **Expected Value** - Average EV per bet

### Statistical Testing
- Hypothesis: Model performance > Random baseline
- Test: One-tailed t-test (α=0.05)
- Minimum samples: 100 draws

### Baseline Comparison
Random selection expected performance:
- Average matches: **0.7347 per draw**
- P(3+ matches): **1.87%**
- P(4+ matches): **0.097%**
- P(6 matches): **0.0000072%**

---

## ⚠️ Important Disclaimers

### Fundamental Limitations

1. **True Randomness**
   - TOTO draws are designed to be random
   - Past results have no bearing on future draws
   - Any patterns may be spurious correlations

2. **Expected Performance**
   - Most likely: Performance ≈ random baseline
   - Best case: Marginal improvement (0.05-0.1 matches)
   - Educational value: High
   - Practical utility: Low

3. **Ethical Considerations**
   - No gambling encouragement
   - Transparent reporting (no cherry-picking)
   - Regulatory compliance
   - Research and analysis only

---

## 📚 Educational Value

This project is excellent for learning:

- **Machine Learning Pipeline Design** - End-to-end ML system
- **Multiple Algorithm Implementation** - Diverse ML techniques
- **Time Series Analysis** - Sequential data modeling
- **Feature Engineering** - Creative feature creation
- **Ensemble Methods** - Model combination strategies
- **Evaluation Frameworks** - Rigorous testing methodology
- **MLOps Practices** - Deployment and monitoring

---

## 🔍 Statistical Baseline

For reference, random number selection (6 from 49):

| Outcome | Probability |
|---------|-------------|
| 0 matches | 43.6% |
| 1 match | 41.3% |
| 2 matches | 13.2% |
| 3 matches | 1.77% |
| 4 matches | 0.097% |
| 5 matches | 0.0018% |
| 6 matches | 0.0000072% |

**Expected matches per draw:** 0.7347

---

## 📖 How to Use This Repository

### For Learning
1. Read `toto_ml_prediction_plan.md` first for overview
2. Study `data_pipeline_design.md` for data engineering
3. Explore `ml_algorithms_implementation.md` for ML details
4. Review `project_structure.md` for implementation guide

### For Implementation
1. Follow setup instructions in `project_structure.md`
2. Implement data pipeline from `data_pipeline_design.md`
3. Build models using `ml_algorithms_implementation.md`
4. Follow roadmap in `toto_ml_prediction_plan.md`

### For Research
1. Analyze the statistical approach
2. Study feature engineering techniques
3. Compare ensemble strategies
4. Design experiments using the framework

---

## 🎓 Key Learning Outcomes

After completing this project, you will understand:

✅ How to design end-to-end ML systems
✅ Multiple deep learning architectures (DNN, LSTM, Transformer)
✅ Tree-based ensemble methods (LightGBM, Random Forest)
✅ Time series feature engineering
✅ Reward structure design
✅ Model evaluation and backtesting
✅ Statistical significance testing
✅ MLOps and deployment practices
✅ Realistic limitations of ML predictions

---

## 🤝 Contributing

This is a planning repository. Contributions welcome for:
- Improving documentation clarity
- Adding alternative approaches
- Suggesting better evaluation metrics
- Sharing research findings

---

## 📜 License

MIT License - See LICENSE file

---

## 📞 Contact & Support

For questions about this planning documentation:
- Open an issue on GitHub
- Refer to individual documents for detailed information
- Check the troubleshooting section in `project_structure.md`

---

## 🎯 Success Criteria

### Minimum Viable Success
✅ All 5 algorithms implemented
✅ Complete data pipeline functional
✅ Statistical comparison with baseline
✅ Comprehensive documentation

### Stretch Goals
🎯 Performance > random baseline (statistically significant)
🎯 Positive expected value in backtesting
🎯 Group 5-7 wins more frequent than baseline
🎯 Ensemble outperforms individual models

### Realistic Expectations
- **Most Likely:** Performance ≈ random baseline
- **Learning Value:** Very High
- **Practical Gambling Use:** Not Recommended
- **Technical Skills Gained:** Extensive

---

## 📅 Project Timeline

**Total Duration:** 10 weeks (estimated)

| Week | Phase | Focus Area |
|------|-------|------------|
| 1-2 | Data Infrastructure | Scraping, database, collection |
| 3 | Feature Engineering | Feature creation & selection |
| 4-6 | Algorithm Implementation | 5 ML models |
| 7-8 | Training & Optimization | Tuning, cross-validation |
| 9 | Ensemble & Evaluation | Integration, backtesting |
| 10 | Deployment | API, monitoring, documentation |

---

## 🌟 Why This Project?

Despite the inherent unpredictability of lottery systems, this project offers:

1. **Real-world ML Pipeline Design** - Complete end-to-end system
2. **Multiple Algorithm Comparison** - Diverse ML techniques
3. **Rigorous Evaluation** - Statistical testing, baselines
4. **Ethical ML Practice** - Transparent limitations, responsible approach
5. **Educational Excellence** - Comprehensive learning opportunity

Remember: The journey of building a robust ML system is more valuable than achieving improbable prediction accuracy. This project teaches the entire ML lifecycle while maintaining intellectual honesty about limitations.

---

**Status:** ✅ Planning Complete - Ready for Implementation
**Version:** 1.0
**Date:** 2025-11-12
**Next Step:** Begin Phase 1 - Data Infrastructure

---

## 📁 Repository Contents

```
fun/
├── README_PLANNING.md                      # This file
├── toto_ml_prediction_plan.md              # Main planning document
├── data_pipeline_design.md                 # Data engineering details
├── ml_algorithms_implementation.md         # ML model specifications
└── project_structure.md                    # Implementation guide
```

---

**Happy Learning! 🚀**

Remember: Build responsibly, learn continuously, predict skeptically.
