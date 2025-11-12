# TOTO Number Prediction ML System - Comprehensive Planning Document

## Executive Summary

### Important Disclaimer
**⚠️ Critical Understanding:** Singapore Pools TOTO is a regulated lottery system designed to produce random outcomes. Each draw is independent, and past results have no mathematical bearing on future draws. This project is for **educational and research purposes only** to explore ML techniques. The expected prediction accuracy for truly random systems approaches baseline probability.

### Project Objective
Design and implement a multi-algorithm machine learning system to explore pattern recognition in TOTO draw data, with proper reward structures aligned to Singapore Pools prize tiers.

---

## 1. TOTO Game Structure Analysis

### Draw Mechanics
- **Draw Frequency:** Every Monday and Thursday, 6:30 PM SGT
- **Number Pool:** 1-49 (49 total numbers)
- **Selection:** 6 main winning numbers + 1 additional number
- **Bet Format:** Players select 6 numbers from 1-49

### Prize Structure (7 Groups)

| Group | Match Requirement | Prize Type | Prize Amount/Share |
|-------|------------------|------------|-------------------|
| 1 | 6 numbers | Variable | ~38% of prize pool (Min $1M) |
| 2 | 5 numbers + additional | Variable | ~8% of prize pool |
| 3 | 5 numbers | Variable | ~5.5% of prize pool |
| 4 | 4 numbers | Variable | ~3% of prize pool |
| 5 | 3 numbers + additional | Fixed | $50 |
| 6 | 3 numbers | Fixed | $25 |
| 7 | Partial match | Fixed | $10 |

### Probability Baseline
- Group 1 (6/6): 1 in 13,983,816
- Group 2 (5/6 + add): 1 in 2,330,636
- Group 3 (5/6): 1 in 55,491
- Group 4 (4/6): 1 in 1,083
- Group 5-7 (3/6 variations): 1 in 61-196

---

## 2. Data Collection Strategy

### Data Sources
1. **Primary:** Singapore Pools Historical Data
   - URL: `https://www.singaporepools.com.sg/en/product/Pages/toto_results.aspx`
   - Target: All Monday/Thursday draws from 1968-present

2. **Data Points per Draw:**
   - Draw date and number
   - 6 winning numbers
   - 1 additional number
   - Prize pool amount
   - Winners per group
   - Cascade/Hongbao indicators (special draws)

### Data Collection Pipeline
```python
Required Fields:
- draw_id: Unique identifier
- draw_date: ISO format
- day_of_week: Monday/Thursday
- winning_numbers: [n1, n2, n3, n4, n5, n6] (sorted)
- additional_number: int
- prize_pool: float
- group_1_winners: int
- draw_type: normal/cascade/hongbao/special
```

### Preprocessing Requirements
1. **Data Validation:**
   - Ensure all numbers are within 1-49
   - Verify 6 unique winning numbers
   - Check additional number not in winning set
   - Validate date is Monday or Thursday

2. **Data Enrichment:**
   - Sequential draw numbering
   - Days since last draw
   - Festival/holiday flags
   - Month/quarter/year indicators

---

## 3. Five Machine Learning Algorithms

### Algorithm 1: Deep Neural Network (DNN) with Embedding
**Type:** Deep Learning
**Architecture:**
- Input Layer: Historical sequence (last 20 draws)
- Embedding Layer: 49-dimensional number embeddings
- Hidden Layers: 3 layers [512, 256, 128] with dropout (0.3)
- Output Layer: 49 neurons (probability distribution)

**Rationale:**
- Can learn complex non-linear patterns
- Embedding captures number relationships
- Sequence modeling for temporal dependencies

**Training Strategy:**
- Loss: Custom multi-label cross-entropy
- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Epochs: 100 with early stopping

---

### Algorithm 2: Recurrent Neural Network (LSTM)
**Type:** Sequential Deep Learning
**Architecture:**
- Input: Sequence of last 50 draws (timesteps=50)
- LSTM Layers: 2 layers [256, 128] with return sequences
- Attention Mechanism: Self-attention over sequence
- Dense Layers: [128, 64]
- Output: 49 probability scores

**Rationale:**
- Specialized for sequential/temporal data
- Long-term dependency modeling
- Attention focuses on relevant historical patterns

**Training Strategy:**
- Loss: Binary cross-entropy (multi-label)
- Optimizer: RMSprop (lr=0.0005)
- Sequence length: 50 draws
- Gradient clipping: 1.0

---

### Algorithm 3: Gradient Boosting (XGBoost/LightGBM)
**Type:** Ensemble Tree-Based
**Architecture:**
- Framework: LightGBM
- Trees: 500-1000 estimators
- Max depth: 8
- Learning rate: 0.05
- Multi-output: 49 binary classifiers (one per number)

**Rationale:**
- Excellent for tabular feature data
- Captures feature interactions
- Robust to overfitting
- Interpretable feature importance

**Features:**
- Number frequency (last 10, 20, 50 draws)
- Number gaps (draws since last appearance)
- Hot/cold number indicators
- Statistical features (mean, std, skewness)
- Day/month patterns

**Training Strategy:**
- Objective: Binary classification per number
- Evaluation: AUC-ROC
- Cross-validation: Time-series split (5 folds)

---

### Algorithm 4: Random Forest with Feature Engineering
**Type:** Ensemble Bagging
**Architecture:**
- Trees: 300 estimators
- Max depth: 15
- Min samples split: 10
- Bootstrap: True
- Feature subset: sqrt(n_features)

**Rationale:**
- Reduces variance through averaging
- Handles high-dimensional feature space
- Natural feature selection
- Parallel processing capability

**Advanced Features:**
- Pair/triplet co-occurrence frequencies
- Number sum and product statistics
- Odd/even ratio patterns
- Consecutive number patterns
- Positional analysis (small/large numbers)

**Training Strategy:**
- Out-of-bag error estimation
- Feature importance ranking
- Ensemble predictions with confidence scores

---

### Algorithm 5: Transformer-Based Model
**Type:** Attention-Based Deep Learning
**Architecture:**
- Input: Tokenized draw sequences
- Positional Encoding: Learned draw positions
- Transformer Blocks: 4 layers
  - Multi-head attention: 8 heads
  - FFN: [512, 256]
  - Layer norm + residual connections
- Output: 49-class multi-label prediction

**Rationale:**
- State-of-art for sequence modeling
- Parallel processing of entire sequence
- Long-range dependency capture
- Self-attention reveals pattern importance

**Training Strategy:**
- Loss: Focal loss (handles class imbalance)
- Optimizer: AdamW with warmup
- Sequence length: 100 draws
- Learning rate schedule: Cosine annealing

---

## 4. Feature Engineering Strategy

### Temporal Features
1. **Recency Features:**
   - Days since number last appeared
   - Draw count since last appearance
   - Appearance count in last [5, 10, 20, 50] draws

2. **Cyclical Features:**
   - Month (sin/cos encoding)
   - Quarter
   - Year progression
   - Draw number modulo patterns

### Statistical Features
1. **Number-Level Statistics:**
   - Historical frequency (all-time)
   - Rolling frequency (windows: 10, 20, 50, 100 draws)
   - Z-score of frequency
   - Streak counts (consecutive appearances)

2. **Distribution Features:**
   - Mean of drawn numbers
   - Standard deviation
   - Skewness and kurtosis
   - Range (max - min)
   - Odd/even ratio
   - Sum of numbers

### Pattern Features
1. **Co-occurrence Patterns:**
   - Pair frequency matrix (49x49)
   - Triplet patterns (conditional probabilities)
   - Sequence patterns (consecutive numbers)

2. **Positional Features:**
   - Position where number appeared (1st-6th)
   - Low (1-16), Mid (17-33), High (34-49) distribution
   - Decade distribution (1-10, 11-20, etc.)

### External Features
1. **Calendar Events:**
   - Public holidays (one-hot)
   - Chinese New Year periods
   - Festival indicators
   - Special draw flags

2. **Meta Features:**
   - Prize pool size (previous draw)
   - Number of winners (previous draw)
   - Draw type (normal/cascade/hongbao)

---

## 5. Reward Structure Design

### Reward Function Philosophy
The reward should reflect real-world utility aligned with prize structure, emphasizing:
1. Higher rewards for unlikely achievements (Group 1-2)
2. Exponential scaling based on match difficulty
3. Penalty for completely wrong predictions
4. Bonus for partial matches (Group 5-7)

### Proposed Reward Function

```python
def calculate_reward(predicted_numbers, winning_numbers, additional_number):
    """
    Calculate reward based on Singapore Pools prize structure
    """
    matches = len(set(predicted_numbers) & set(winning_numbers))
    has_additional = additional_number in predicted_numbers

    # Group-based rewards (scaled for ML training)
    rewards = {
        'group_1': 1000000,  # 6 matches
        'group_2': 100000,   # 5 + additional
        'group_3': 10000,    # 5 matches
        'group_4': 1000,     # 4 matches
        'group_5': 50,       # 3 + additional
        'group_6': 25,       # 3 matches
        'baseline': -10,     # Less than 3 matches
    }

    # Determine group and reward
    if matches == 6:
        return rewards['group_1']
    elif matches == 5 and has_additional:
        return rewards['group_2']
    elif matches == 5:
        return rewards['group_3']
    elif matches == 4:
        return rewards['group_4']
    elif matches == 3 and has_additional:
        return rewards['group_5']
    elif matches == 3:
        return rewards['group_6']
    else:
        return rewards['baseline']
```

### Normalized Reward for Model Training

For stable training, normalize rewards to [-1, 1]:

```python
def normalized_reward(matches, has_additional=False):
    """Normalized reward for gradient-based learning"""
    if matches == 6:
        return 1.0
    elif matches == 5 and has_additional:
        return 0.8
    elif matches == 5:
        return 0.6
    elif matches == 4:
        return 0.4
    elif matches == 3 and has_additional:
        return 0.2
    elif matches == 3:
        return 0.1
    elif matches == 2:
        return -0.3
    elif matches == 1:
        return -0.6
    else:
        return -1.0
```

### Reinforcement Learning Reward Shaping

For RL-based approaches:

1. **Immediate Reward:** Match-based reward (as above)
2. **Delayed Reward:** Cumulative over N draws
3. **Exploration Bonus:** Encourage trying different number combinations
4. **Diversity Penalty:** Avoid repeatedly predicting same numbers

```python
total_reward = immediate_reward +
               0.1 * diversity_score +
               0.05 * exploration_bonus -
               0.2 * repetition_penalty
```

### Expected Value Calculation

For realistic assessment, calculate expected value per bet:

```
EV = Σ(P(group_i) × Reward(group_i)) - Bet_Cost

Where:
- P(group_i) = Model's predicted probability of achieving group i
- Reward(group_i) = Average prize for group i
- Bet_Cost = $1 (ordinary bet)
```

---

## 6. Training Strategy

### Data Split Strategy
**Time-Series Split (No Shuffle!):**
- Training: 70% oldest draws
- Validation: 15% middle draws
- Test: 15% most recent draws

**Rationale:** Prevents data leakage; simulates real prediction scenario

### Cross-Validation
**Expanding Window CV:**
```
Fold 1: Train[0:1000]     → Validate[1000:1200]
Fold 2: Train[0:1200]     → Validate[1200:1400]
Fold 3: Train[0:1400]     → Validate[1400:1600]
Fold 4: Train[0:1600]     → Validate[1600:1800]
Fold 5: Train[0:1800]     → Validate[1800:2000]
```

### Training Protocol per Algorithm

#### Phase 1: Independent Training
Each algorithm trains independently on the same data:
- Separate hyperparameter optimization
- Algorithm-specific feature engineering
- Individual performance tracking

#### Phase 2: Ensemble Strategy
Combine predictions using:
1. **Simple Average:** Equal weight to all 5 algorithms
2. **Weighted Average:** Weight by validation performance
3. **Stacking:** Meta-learner combines predictions
4. **Voting:** Top-K numbers by vote count

### Hyperparameter Optimization
- Method: Bayesian Optimization (Optuna)
- Budget: 100 trials per algorithm
- Objective: Maximize validation reward
- Early stopping: Patience = 10

---

## 7. Evaluation Framework

### Primary Metrics

1. **Hit Rate by Group:**
   - Group 1-7 achievement rate
   - Compare vs. baseline probability

2. **Average Matches per Draw:**
   - Mean number of correct predictions
   - Baseline: ~0.73 matches (random)

3. **Cumulative Reward:**
   - Total reward over test period
   - Compare vs. random selection

4. **Expected Value:**
   - Average EV per bet
   - Must be > -$1 (break-even vs. cost)

### Secondary Metrics

1. **Coverage:**
   - % of actual winning numbers in top-K predictions

2. **Precision/Recall:**
   - Precision@K (K=6, 10, 15)
   - Recall: How many winning numbers captured

3. **Calibration:**
   - Are predicted probabilities well-calibrated?
   - Brier score for probability accuracy

### Statistical Significance Testing

**Hypothesis Test:**
- H0: Model performance = Random baseline
- H1: Model performance > Random baseline
- Test: One-tailed t-test (α=0.05)
- Minimum samples: 100 draws (200+ draws preferred)

**Random Baseline:**
```python
def random_baseline(n_simulations=10000):
    results = []
    for _ in range(n_simulations):
        predicted = random.sample(range(1, 50), 6)
        actual = random.sample(range(1, 50), 6)
        matches = len(set(predicted) & set(actual))
        results.append(matches)
    return np.mean(results), np.std(results)
```

### Backtesting Protocol

1. **Rolling Prediction:**
   - Use data up to draw N-1
   - Predict draw N
   - Evaluate against actual results
   - Retrain every M draws (M=10 or 20)

2. **Performance Tracking:**
   - Track metrics over time
   - Identify performance degradation
   - Adapt to concept drift

---

## 8. Risk Mitigation & Limitations

### Known Limitations

1. **Fundamental Randomness:**
   - True random systems cannot be predicted
   - Any patterns may be spurious correlations

2. **Overfitting Risk:**
   - High risk with limited signal
   - Extensive regularization required

3. **Data Limitations:**
   - ~2,800 draws since 1968 (limited samples)
   - Distribution may change over time

### Mitigation Strategies

1. **Conservative Approach:**
   - Never claim prediction accuracy
   - Always compare vs. random baseline
   - Report confidence intervals

2. **Regularization:**
   - Heavy dropout in neural networks
   - L1/L2 regularization
   - Early stopping
   - Cross-validation

3. **Ensemble Diversity:**
   - Use fundamentally different algorithms
   - Different feature sets
   - Reduce overfitting through averaging

4. **Responsible Reporting:**
   - Clear disclaimers
   - Statistical significance testing
   - Transparent methodology

---

## 9. Implementation Roadmap

### Phase 1: Data Infrastructure (Week 1-2)
- [ ] Build web scraper for Singapore Pools
- [ ] Create database schema
- [ ] Implement data validation pipeline
- [ ] Historical data collection (1968-present)
- [ ] Data quality checks

### Phase 2: Feature Engineering (Week 3)
- [ ] Implement all feature categories
- [ ] Feature correlation analysis
- [ ] Feature selection (remove redundant)
- [ ] Create feature engineering pipeline

### Phase 3: Algorithm Implementation (Week 4-6)
- [ ] Algorithm 1: DNN with embeddings
- [ ] Algorithm 2: LSTM with attention
- [ ] Algorithm 3: LightGBM
- [ ] Algorithm 4: Random Forest
- [ ] Algorithm 5: Transformer
- [ ] Implement reward functions

### Phase 4: Training & Optimization (Week 7-8)
- [ ] Individual model training
- [ ] Hyperparameter optimization
- [ ] Cross-validation
- [ ] Model selection

### Phase 5: Ensemble & Evaluation (Week 9)
- [ ] Ensemble strategy implementation
- [ ] Comprehensive backtesting
- [ ] Statistical significance testing
- [ ] Performance reporting

### Phase 6: Deployment & Monitoring (Week 10)
- [ ] Prediction pipeline
- [ ] Performance monitoring dashboard
- [ ] Automated retraining schedule
- [ ] Alert system for anomalies

---

## 10. Technology Stack

### Core Framework
- **Python 3.10+**
- **PyTorch / TensorFlow** (Deep Learning)
- **Scikit-learn** (Classical ML)
- **XGBoost / LightGBM** (Gradient Boosting)

### Data Processing
- **Pandas** (Data manipulation)
- **NumPy** (Numerical computing)
- **Polars** (High-performance alternative)

### Feature Engineering
- **Feature-engine**
- **Category-encoders**
- **TSFRESH** (Time-series features)

### Optimization & Experiment Tracking
- **Optuna** (Hyperparameter optimization)
- **MLflow** (Experiment tracking)
- **Weights & Biases** (Alternative)

### Deployment
- **FastAPI** (Prediction API)
- **Docker** (Containerization)
- **Prometheus + Grafana** (Monitoring)

---

## 11. Success Criteria

### Minimum Viable Success
1. System successfully predicts numbers for each draw
2. Performance statistically measured against random baseline
3. All 5 algorithms implemented and evaluated
4. Comprehensive documentation of methodology

### Stretch Goals
1. Average matches > random baseline (statistically significant)
2. Positive expected value in backtesting
3. Achieve Group 5-7 wins more frequently than baseline
4. Ensemble outperforms individual models

### Realistic Expectations
- **Most Likely Outcome:** Performance ≈ random baseline
- **Best Case:** Marginal improvement (0.05-0.1 more matches on average)
- **Educational Value:** High (excellent ML learning project)
- **Practical Utility:** Low (lottery remains unpredictable)

---

## 12. Ethical Considerations

1. **No Gambling Encouragement:**
   - Clear disclaimers about lottery randomness
   - Educational purpose emphasis
   - Responsible gambling messaging

2. **Transparency:**
   - Open-source methodology
   - Honest reporting of results
   - No cherry-picking successful predictions

3. **Regulatory Compliance:**
   - Respect Singapore Pools terms of service
   - No automated betting systems
   - Research and analysis only

---

## 13. Next Steps

1. **Review and Approval:**
   - Stakeholder review of plan
   - Feedback incorporation
   - Resource allocation

2. **Environment Setup:**
   - Development environment configuration
   - Access to compute resources (GPU)
   - Database setup

3. **Initial Data Collection:**
   - Test web scraping on recent draws
   - Validate data quality
   - Establish update pipeline

4. **Begin Implementation:**
   - Start Phase 1 (Data Infrastructure)
   - Weekly progress reviews
   - Iterative refinement

---

## Appendix A: Prize Pool Distribution

Based on Singapore Pools structure:
- **54%** of sales goes to prize pool
- **Distribution:**
  - Group 1: 38% of pool
  - Group 2: 8% of pool
  - Group 3: 5.5% of pool
  - Group 4: 3% of pool
  - Group 5-7: Fixed amounts
  - Remaining: Rolls over to Group 1

## Appendix B: Statistical Baselines

**Random Selection (6 from 49):**
- Expected matches: 0.7347 per draw
- P(0 matches): 43.6%
- P(1 match): 41.3%
- P(2 matches): 13.2%
- P(3 matches): 1.77%
- P(4 matches): 0.097%
- P(5 matches): 0.0018%
- P(6 matches): 0.0000072%

**"Smart" Random (avoiding recent numbers):**
- May achieve ~0.75-0.77 matches
- Still fundamentally random

---

**Document Version:** 1.0
**Date:** 2025-11-12
**Status:** Planning Complete - Ready for Implementation Review
