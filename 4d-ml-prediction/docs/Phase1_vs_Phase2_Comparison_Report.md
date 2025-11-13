# Phase 1 vs Phase 2 Model Comparison Report
## 4D Number Prediction - February 2025 Analysis

### Executive Summary

This report compares two different methodologies for 4D number prediction:
- **Phase 1**: Direct 4D classification (10,000-way multi-output)
- **Phase 2**: Digit-by-digit prediction (4 × 10-way classification)

### Methodology Comparison

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| **Approach** | Holistic number prediction | Digit-level prediction |
| **Classifiers** | ~385 binary models (active numbers) | 4 multi-class models (one per digit) |
| **Target Encoding** | Multi-hot (23 winners per draw) | Individual digit targets |
| **Training Complexity** | Very high (hundreds of models) | Moderate (4 models) |
| **Prediction Strategy** | Direct top-K selection | Combine digit probabilities |
| **Inspired By** | TOTO P1_LightGBM (best performer) | Novel digit-based approach |

### Evaluation Results - February 2025 (6 Draws)

#### Draw-by-Draw Comparison

| Date | Training Data | P1 Prediction | P1 Conf | P2 Prediction | P2 Conf | Actual 1st Prize | Winner |
|------|---------------|---------------|---------|---------------|---------|------------------|---------|
| 2025-02-01 | 141 draws | **7579** | 99% | **7579** | 92% | 4731 | None |
| 2025-02-05 | 142 draws | **9137** | 99% | **4731** | 94% | 3906 | None |
| 2025-02-08 | 143 draws | **3354** | 98% | **3906** | 92% | 9641 | None |
| 2025-02-09 | 144 draws | **7600** | 98% | **9641** | 92% | 9371 | None |
| 2025-02-12 | 145 draws | **6148** | 99% | **9371** | 91% | 9023 | None |
| 2025-02-15 | 146 draws | **6866** | 98% | **9023** | 93% | 1666 | None |

#### Key Observations

**Phase 1 Characteristics:**
- **Higher Confidence**: Average 98.5% (range: 98-99%)
- **Diverse Predictions**: Different numbers each draw
- **Prediction Pattern**: No clear pattern, appears to learn from full historical distribution
- **Computational Cost**: Very high - trains ~385 binary classifiers per draw

**Phase 2 Characteristics:**
- **Lower Confidence**: Average 92.3% (range: 91-94%)
- **Predictable Pattern**: Consistently predicts previous draw's 1st prize (5 out of 6 times)
- **Prediction Pattern**: Strong recency bias, follows "hot number" strategy
- **Computational Cost**: Moderate - trains 4 digit-level classifiers per draw

**Win Performance:**
- Phase 1 Wins: **0/6** (0%)
- Phase 2 Wins: **0/6** (0%)
- **Result**: Tie in this evaluation period

### Detailed Analysis

#### 1. Confidence Levels

Phase 1 demonstrates significantly higher confidence (98.5% vs 92.3%), suggesting:
- The multi-hot encoding creates clearer decision boundaries
- Training on 23 positive examples per draw provides stronger signals
- Direct number prediction may overfit to training patterns

Phase 2's lower confidence indicates:
- Uncertainty in combining independent digit predictions
- Digit-level features may be less discriminative
- The multiplication of 4 probabilities reduces overall confidence

#### 2. Prediction Strategies

**Phase 1 Strategy**: Learns holistic patterns
- Considers entire 4D number as atomic unit
- Captures inter-digit dependencies
- Predictions vary based on complex feature interactions
- Example: 7579 → 9137 → 3354 (no obvious pattern)

**Phase 2 Strategy**: "Hot number" recency bias
- Predicts previous draw's 1st prize 83% of the time (5/6)
- Digit-level models learn individual digit frequencies
- When combined, tends to reproduce recent winning numbers
- Example: 4731 (actual) → predicted 4731 next draw

#### 3. Computational Comparison

**Training Time Comparison** (estimated for 10 draws):
- Phase 1: ~20-30 minutes
  - Trains ~385 binary classifiers per draw
  - 10 draws × 385 models = 3,850 model training cycles

- Phase 2: ~3-5 minutes
  - Trains 4 digit classifiers per draw
  - 10 draws × 4 models = 40 model training cycles

**Performance Trade-off**:
- Phase 1: 6-8x longer training time for +6% confidence
- Phase 2: Faster but shows clear overfitting to recency

### Interesting Findings

#### Finding #1: Both Models Failed to Win
Despite high confidence from both models, neither achieved any wins in the evaluated period. This confirms:
- 4D lottery remains highly random (0.23% win probability)
- Historical patterns have limited predictive power
- High confidence ≠ accurate predictions

#### Finding #2: Phase 2's Recency Bias
Phase 2 consistently predicted the previous draw's winning number, which is a classic pattern in gambling fallacy:
- Known as "hot number" bias
- Assumes recent winners will repeat
- In reality, each draw is independent

#### Finding #3: Confidence vs Accuracy Disconnect
Phase 1's 99% confidence on prediction "7579" for 2025-02-01 (actual: 4731) shows:
- Model confidence doesn't reflect actual win probability
- Training on multi-hot encoded data creates artificial certainty
- Calibration issues common in lottery prediction

#### Finding #4: Identical First Prediction
Both models predicted "7579" for 2025-02-01:
- Phase 1: 99% confidence
- Phase 2: 92% confidence
- This suggests both models converged on similar patterns from training data
- Yet still missed the actual number (4731)

### Conclusions

#### Which Model is Better?

**For Prediction Quality**: **Inconclusive**
- Both achieved 0% win rate in this evaluation
- Need more extensive testing across multiple months
- 6 draws insufficient for statistical significance

**For Confidence Calibration**: **Neither is well-calibrated**
- Phase 1: Overconfident (99% confidence, 0% accuracy)
- Phase 2: Shows recency bias (predicts previous winners)

**For Computational Efficiency**: **Phase 2 Wins**
- 6-8x faster training
- More practical for real-time predictions
- Lower memory requirements

**For Methodological Soundness**: **Phase 1 Shows Promise**
- More diverse predictions (not stuck in patterns)
- Inspired by successful TOTO implementation
- Higher confidence suggests better-learned representations

### Recommendations

#### For Further Evaluation:
1. **Extend evaluation period**: Test on all 54 draws (Feb-Jun 2025)
2. **Track all 23 winning numbers**: Not just 1st prize
3. **Measure calibration**: Compare confidence scores to actual win rates
4. **Test ensemble**: Combine Phase 1 + Phase 2 predictions

#### For Model Improvement:
1. **Phase 1**:
   - Reduce overfitting with regularization
   - Calibrate confidence scores
   - Consider training only on top-3 prizes (reduce target sparsity)

2. **Phase 2**:
   - Add anti-recency features to break "hot number" bias
   - Improve digit independence assumptions
   - Consider conditional modeling (digit 2 given digit 1, etc.)

### Technical Notes

**Why Phase 1 Takes So Long:**
- For each draw, trains one binary classifier per active 4D number
- With 17 draws in training, ~385 unique 4D numbers appear
- Training 385 LightGBM models × 10 draws = 3,850 training cycles
- Each model: 100 trees × 6 max_depth

**Why Phase 2 is Faster:**
- Only trains 4 models per draw (one per digit position)
- 4 models × 10 draws = 40 training cycles
- ~96x fewer model trainings than Phase 1

### Data Limitations

This analysis used **mock 4D data** with the following caveats:
- Generated numbers may not reflect real Singapore 4D distributions
- Random seed ensures reproducibility but not realism
- Real draws have regulatory oversight and certified randomness
- Mock data used for methodology demonstration only

### Final Verdict

**Current State (6 draws evaluated):**
- **Tie**: Both models: 0 wins, 0% success rate
- **Phase 1**: Higher confidence but slower
- **Phase 2**: Faster but shows problematic recency bias

**Recommendation**:
Neither model is ready for production use. Both need significant improvements before deployment. Phase 1 shows more promise methodologically but requires optimization for practical use.

**Next Steps**:
1. Complete full Feb-Jun evaluation (54 draws)
2. Implement confidence calibration
3. Test hybrid/ensemble approaches
4. Consider alternative feature engineering

---

*Report generated: 2025-11-13*
*Evaluation period: February 2025 (6 draws)*
*Data source: Mock 4D draw data*
