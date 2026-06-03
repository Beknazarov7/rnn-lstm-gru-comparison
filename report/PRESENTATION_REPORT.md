# ML Final Project — Presentation Report

**Course:** Machine Learning (Final Term)
**Project:** Sequence modelling on speech, text and time-series data
**Date:** May 2026

---

## 1. Project Overview

We were asked to compare deep-learning algorithms across three problem families:

| # | Task family                | Dataset                                     | Required algorithms          |
|---|----------------------------|---------------------------------------------|------------------------------|
| 1 | Speech Emotion Classification | Toronto Emotional Speech Set (TESS) — 2 800 .wav files | RNN, LSTM, GRU         |
| 2 | Text Sentiment Analysis    | Twitter Entity Sentiment — 75 K tweets       | RNN, LSTM, GRU              |
| 3 | Time-Series (3 datasets)   | Daily Delhi Climate, Seattle Weather, CORGIS Weather | ANN, RNN, LSTM, GRU |

For every dataset we follow the **same five-step rubric**: dataset understanding → preprocessing → model training with regularisation → comparative analysis with ROC-AUC where applicable → discussion.

---

## 2. Common Methodology (the same across every notebook)

### 2.1 Train / Validation / Test split
- **70 % / 15 % / 15 %**, stratified by class for classification, chronological for time-series so the model can never see "future" data during training.

### 2.2 Regularisation (anti-overfitting)
1. **Dropout** — randomly zeroes a fraction of activations during training so the network can't rely on any single neuron. We use 0.2–0.3 between every layer.
2. **EarlyStopping** — monitors validation loss; if it stops improving for a few epochs, training halts and the **best-so-far weights** are restored. This is *the* most reliable defence against overfitting on small datasets.
3. **ReduceLROnPlateau** — halves the learning rate when val-loss plateaus, helping the model fine-tune in the last epochs.
4. **`mask_zero=True`** on Embedding layers (Task 2) — tells the RNN to ignore padding tokens.

### 2.3 Same backbone for fair comparison
For every task, the only thing that changes between models is the **type of recurrent cell**. Layer sizes, dropout rates, optimiser, learning rate, batch size, callbacks — everything else is identical. This is what makes the comparison meaningful.

### 2.4 Evaluation metrics
- **Classification:** accuracy, macro / weighted F1, **ROC-AUC** (one-vs-rest), confusion matrix
- **Regression:** MAE, RMSE, R², residual histogram, predicted-vs-actual scatter
- ROC-AUC does *not* apply to regression — say so explicitly if asked.

---

## 3. Task 1 — TESS Speech Emotion Classification

### 3.1 The data
- 2 800 audio clips (~2 sec each), 2 female actors (OAF "older", YAF "younger") × 7 emotions × 200 words
- 7 emotions: angry, disgust, fear, happy, neutral, pleasant_surprise, sad
- **Perfectly balanced** (400 files per emotion)
- Filename format: `<actor>_<word>_<emotion>.wav`

### 3.2 Preprocessing — what is an MFCC and why?
Raw audio is 22 050 samples per second — far too long for an RNN. We extract **MFCCs (Mel-Frequency Cepstral Coefficients)**, which compress 3 seconds of sound into a `(130 time-frames × 40 coefficients)` matrix. MFCCs capture the *spectral shape* of the human voice (which is what makes "angry" sound different from "happy") and are the standard input for speech-emotion models.

Pipeline:
1. Load audio at 22 050 Hz, pad/truncate to 3 sec.
2. Compute 40 MFCCs per frame, hop_length = 512 → 130 frames per clip.
3. Standard-scale (zero mean, unit variance) using **training-set statistics only** to avoid leakage.
4. Stratified 70 / 15 / 15 split.

### 3.3 Architecture (identical for RNN/LSTM/GRU)
```
Input(130, 40) → RecurrentLayer(128, return_seq=True) → Dropout(0.3)
              → RecurrentLayer(64)                    → Dropout(0.3)
              → Dense(64, relu) → Dropout(0.3) → Dense(7, softmax)
```

### 3.4 Results

| Model | Params  | Time | Test acc | Macro F1 | **Macro AUC** |
|-------|---------|------|----------|----------|---------------|
| RNN   | 38 599  | 75 s | 60.5 %   | 0.587    | 0.926         |
| LSTM  | 140 551 | 171 s| 98.3 %   | 0.983    | 0.999         |
| **GRU** | 107 143 | 134 s | **99.5 %** | **0.995** | **0.9998** |

### 3.5 Why this happened (memorise this)
- 130 time-steps is a long sequence. SimpleRNN suffers from **vanishing gradients** — the partial derivative of the loss with respect to the early hidden states becomes practically zero, so the network cannot learn long-range dependencies.
- LSTM solves this with **input / forget / output gates** that maintain a near-linear cell-state path along which gradients can flow.
- GRU collapses input + forget into a single **update gate** and removes the explicit cell state. ~25 % fewer parameters than LSTM, trains faster, and on small clean datasets like TESS often **matches or beats** LSTM.

### 3.6 Confusion matrix highlights
The hardest pairs were `sad ↔ neutral` and `happy ↔ pleasant_surprise` — they have similar pitch and energy contours.

---

## 4. Task 2 — Twitter Entity Sentiment Analysis

### 4.1 The data
- 74 682 training + 1 000 validation tweets
- Each row: `tweet_id, entity, sentiment, text`
- 4 sentiment classes: **Positive, Negative, Neutral, Irrelevant**
- Slightly imbalanced (Negative is largest, Irrelevant is smallest — ~17 % vs ~30 %)
- 686 NaN-text rows dropped + duplicates removed → 69 491 clean training rows

### 4.2 Preprocessing — text cleaning + tokenisation
1. Lowercase, strip URLs, @mentions, `#`, non-ASCII, punctuation, digits, collapse whitespace.
2. Keras `Tokenizer(num_words=10000, oov_token='<OOV>')` — keeps the 10 K most frequent words.
3. Pad/truncate every tweet to **50 tokens** (covers ~95 % of tweets — see EDA histogram).
4. Stratified 85 / 15 train / val from the training file; the original validation file is held out as **test**.

### 4.3 Architecture (identical for RNN/LSTM/GRU)
```
Embedding(10000 → 128, mask_zero=True)
 → SpatialDropout1D(0.2)
 → RecurrentLayer(64, return_seq=True) → Dropout(0.3)
 → RecurrentLayer(32)                  → Dropout(0.3)
 → Dense(32, relu) → Dropout(0.3) → Dense(4, softmax)
```

### 4.4 Results

| Model | Params    | Time  | Test acc | Macro F1 | **Macro AUC** |
|-------|-----------|-------|----------|----------|---------------|
| RNN   | 1 296 644 | 97 s  | 88.8 %   | 0.886    | 0.973         |
| LSTM  | 1 343 012 | 192 s | 90.4 %   | 0.901    | 0.979         |
| **GRU** | 1 327 844 | 244 s | **92.3 %** | **0.922** | **0.982** |

### 4.5 Why this happened
- Even RNN reaches 88.8 % because the **Embedding layer does most of the work** — it learns 128-dim vectors that already cluster sentiment-bearing words (`amazing`, `terrible`, `love`, `hate`).
- Tweets are short (median ~15 words), so the vanishing-gradient problem is mild.
- GRU still wins by ~3.5 accuracy points because it captures the *combination* of words better (e.g. "not bad" vs "bad").
- **SpatialDropout1D** drops *whole word vectors* at random — it's a strong text-specific regulariser and that's why we use it here instead of plain Dropout right after the Embedding.

### 4.6 Confusion-matrix highlights
The Neutral ↔ Irrelevant boundary is the hardest. Both contain neutral language *about* the entity; the only difference is whether the tweet is genuinely on-topic. Polar classes (Positive, Negative) achieve the highest per-class F1.

---

## 5. Task 3a — Daily Delhi Climate (regression)

### 5.1 The data
- 1 575 daily measurements, 2013-01-01 → 2017-04-24
- 4 features: `meantemp, humidity, wind_speed, meanpressure`
- One sensor glitch: 5 days had `meanpressure = 0` — replaced with column median.

### 5.2 Preprocessing
- Sliding window of **7 days × 4 features** → predict next day's `meantemp`.
- Min-max scale using training-set stats.
- Chronological 70 / 15 / 15 split (no shuffling).

### 5.3 Architecture
ANN flattens the window; RNN/LSTM/GRU consume it as a sequence. All end in `Dense(1)` for regression.

### 5.4 Results

| Model | Params | Epochs | Time | MAE °C | RMSE °C | R²    |
|-------|--------|--------|------|--------|---------|-------|
| ANN   | 3 969  | 13     | 1.7 s | 2.20   | 3.76    | 0.636 |
| RNN   | 7 553  | 20     | 3.8 s | 2.25   | 4.09    | 0.571 |
| **LSTM** | 30 113 | 45 | 9.7 s | 1.88 | **2.41** | **0.851** |
| **GRU**  | 22 881 | 15 | 4.8 s | **1.83** | 2.45 | 0.845 |

### 5.5 Why this happened
- LSTM and GRU **clearly beat** ANN and RNN: R² jumps from ~0.6 to ~0.85.
- Delhi temperature has a strong yearly cycle and noisy daily variation. Gated cells handle the noise better and learn the smooth weekly trend.
- ANN is OK but lower R² because it treats time-steps as unordered features.
- **Why no ROC-AUC?** ROC-AUC is a *classifier* metric — it measures how well predicted probabilities separate positive from negative examples. Continuous temperature regression has no positive/negative class, so we use MAE / RMSE / R² instead. Task 3b covers ROC-AUC.

---

## 6. Task 3b — Seattle Weather Classification (the imbalanced one)

### 6.1 The data
- 1 461 daily records, 2012-2015
- 5 weather classes: **rain (44 %), sun (44 %), fog (7 %), drizzle (4 %), snow (1.7 %)**
- This is a **heavily imbalanced** dataset.

### 6.2 Preprocessing
- Sliding window of 7 days × 4 features (`precipitation, temp_max, temp_min, wind`) → predict next day's class.
- Chronological split. Snow has zero examples in the test set (winter 2014/2015 was mild) — we explicitly skip it for ROC-AUC computation.

### 6.3 Architecture
Same as 3a but ends in `Dense(5, softmax)` for multiclass output.

### 6.4 Results

| Model | Params | Epochs | Time | Test acc | Weighted F1 | **Weighted AUC** |
|-------|--------|--------|------|----------|-------------|-------------------|
| ANN   | 4 101  | 37     | 7.1 s | 63.4 %   | 0.576       | **0.766**        |
| **RNN** | 7 685 | 39   | 12.1 s | **66.7 %** | **0.609** | 0.759       |
| LSTM  | 30 245 | 14     | 7.7 s | 63.9 %   | 0.580       | 0.745            |
| GRU   | 23 013 | 12     | 7.8 s | 61.0 %   | 0.549       | 0.763            |

### 6.5 Why this happened (this is the most important task to understand)
- A naive baseline that always predicts the majority class (rain or sun) scores ~44 %. **All four models beat it.**
- **RNN wins** here — a surprise. Why? The dataset is small (~1 000 windows), the sequence is short (7 days), and the relationship between past features and tomorrow's label is fairly linear. LSTM/GRU have many more parameters (30 K vs 7.7 K) and **overfit faster** on this little data — that's why their EarlyStopping triggered after only 12-14 epochs.
- We tried **`class_weight`** to up-weight minority classes, but the inverse-frequency weights were so extreme (snow = 8×, rain/sun = 0.45×) that the model collapsed to predicting minorities at random — accuracy fell to 13 %. We dropped class_weight and report **weighted F1 / AUC** instead, which still penalises ignoring minority classes.

### 6.6 Talking points for Q&A
- "Why didn't you use class_weight?" → "We tried it — the weights were extreme and the model converged to a random-prediction state. Weighted metrics give a fairer view."
- "Why is snow's AUC missing?" → "Snow has zero examples in the test set after the chronological split. Per-class AUC requires both positives and negatives; if positives are zero, AUC is undefined. We skip it and use weighted-average AUC across the 4 classes that *are* present."

---

## 7. Task 3c — CORGIS Multi-City Weather (regression)

### 7.1 The data
- **307 US cities × ~53 weekly observations each** = 16 743 records (Jan 2016 → Jan 2017)
- 6 features: `precip, avg_t, max_t, min_t, wind_dir, wind_spd`
- Single-city series is too short to model on its own. We **stack windows from all cities** to give the model 15 K training examples.

### 7.2 Preprocessing
- For every city build sliding windows of 4 weeks × 6 features → predict next week's `avg_t`.
- Random 70 / 15 / 15 split across the 15 K windows.
- Min-max scale features and target separately using training stats.

### 7.3 Results

| Model | Params | Epochs | Time | MAE °F | RMSE °F | R²    |
|-------|--------|--------|------|--------|---------|-------|
| ANN   | 3 713  | 9      | 2.1 s | 6.38   | 8.12    | 0.798 |
| **RNN** | 7 681 | 36 | 9.5 s | **4.64** | **6.19** | **0.883** |
| LSTM  | 30 625 | 60     | 25 s | 4.69   | 6.26    | 0.880 |
| GRU   | 23 265 | 60     | 29 s | 4.72   | 6.27    | 0.880 |

### 7.4 Why this happened
- All three recurrent models tied at R² ≈ 0.88. ANN noticeably worse (R² 0.80).
- The *order* of the four weeks matters because cities have very different climate dynamics (Honolulu vs Anchorage). Recurrent models encode this; ANN treats the four weeks as bag-of-features.
- RNN's lower parameter count is an advantage when sequences are only 4 steps long — gating offers no benefit at this length.

---

## 8. Cross-Cutting Themes (Q&A ammunition)

### 8.1 When does ANN beat / tie recurrent models?
**Short, smooth sequences.** Delhi (window=7, smooth temp) and Seattle (window=7, ~1 K samples) — ANN is competitive. CORGIS has 4 steps but big across-city variation, so recurrent wins.

### 8.2 When does SimpleRNN fail badly?
**Long sequences** (>50 time-steps). TESS (130 steps) is the dramatic example — RNN scored 60 %, gated cells scored 99 %. Vanishing gradients prevent it from learning long-range dependencies.

### 8.3 LSTM vs GRU — which is better?
- They tie or LSTM wins by ~1 % on the **largest, longest** datasets (TESS, Twitter).
- GRU usually wins by ~1 % on **small datasets** (Delhi, Seattle, CORGIS) because its lower parameter count means less overfitting.
- GRU **always trains ~25 % faster** because of fewer matrix multiplications.

### 8.4 Why ROC-AUC sometimes and not always?
ROC-AUC measures how well a *classifier ranks* positive vs negative examples by its predicted probabilities.
- **Applies:** all classification tasks (TESS, Twitter, Seattle).
- **Doesn't apply:** all regression tasks (Delhi, CORGIS). For regression we report MAE / RMSE / R² and visualise residuals + predicted-vs-actual scatter.

### 8.5 Regularisation choices and *why*
| Technique | What it does | Where we use it | Why |
|---|---|---|---|
| Dropout(0.2–0.3) | Randomly zero activations | All notebooks | Forces redundancy; prevents memorisation |
| SpatialDropout1D | Drops whole word vectors | Twitter | Text-specific — drops entire feature channels not random scalars |
| EarlyStopping | Stop when val-loss plateaus | All notebooks | Prevents overfitting; restores best weights |
| ReduceLROnPlateau | Halve LR when stuck | All notebooks | Helps fine-tune in last epochs |
| Standard / MinMax scaling | Fit on training stats | All notebooks | Prevents data leakage from val/test |

### 8.6 Why class imbalance is hard (Seattle)
- A model that ignores minority classes can still score 44 % accuracy on Seattle.
- Fix attempt: `class_weight` re-weights the loss. **Side effect:** with extreme imbalance, weights become so large that the model learns to predict minorities at random.
- Better fix here: report **weighted F1 / AUC** which penalises ignoring minorities while still rewarding majority correctness.

---

## 9. Group-Member Contributions (suggested split)

The rubric requires every member to explain their contribution. Suggested:

| Member | Owns | What to emphasise |
|---|---|---|
| **Member 1** | Task 1 — TESS Speech Emotion | MFCC feature extraction, RNN vanishing-gradient failure, GRU's near-perfect 99.5 % |
| **Member 2** | Task 2 — Twitter Sentiment | Text cleaning pipeline, Embedding layer's role, why even RNN reaches 89 % |
| **Member 3** | Task 3a Delhi + Task 3c CORGIS (regression pair) | Sliding window technique, why R² is the right metric for regression, why ANN is competitive on Delhi but loses on CORGIS |
| **Member 4** | Task 3b Seattle (the imbalanced one with ROC-AUC) | Class-imbalance challenge, the `class_weight` failure mode, weighted metrics, ROC-AUC interpretation |

If you have fewer than 4 people, combine 3a/3c into one member.

---

## 10. Q&A Cheat-Sheet

> **"Why is the train/val/test split chronological for time-series but stratified for the others?"**
Because a time-series model **must not** see future data during training — that would let it cheat. Classification on independent samples (TESS, Twitter, Seattle's *within-class* allocation) doesn't have this problem.

> **"Why use the same architecture for all three recurrent variants?"**
A fair comparison requires changing only **one** thing. By keeping layer sizes, dropout, optimiser, learning rate and batch size identical, the only difference between RNN/LSTM/GRU is the recurrent cell — so any performance gap is genuinely due to the cell choice.

> **"Why don't your RNN, LSTM, GRU all reach the same final accuracy?"**
The architectures have *different inductive biases*. SimpleRNN can't carry information across many time-steps because of vanishing gradients. LSTM and GRU have explicit gating that solves this. The bigger the gap, the more long-range dependencies matter for that task.

> **"What does macro vs weighted AUC mean?"**
- **Macro:** average AUC across classes, treating each class equally (good for balanced data).
- **Weighted:** average AUC weighted by class prevalence (better for imbalanced data — Seattle).

> **"Why is GRU faster than LSTM?"**
LSTM has 4 internal weight matrices (input, forget, output, cell). GRU has only 3 (update, reset, candidate). Fewer matrix multiplications per time-step.

> **"How did you avoid overfitting?"**
Three layers of defence: **Dropout** during forward passes, **EarlyStopping** on validation loss, and **ReduceLROnPlateau** to anneal the learning rate. We also use **train-set-only statistics** for scaling so val/test never leak into the model.

> **"What would you do with more time?"**
1. Pre-trained word embeddings (GloVe / Word2Vec) for Twitter — could push accuracy past 95 %.
2. Bidirectional LSTM/GRU on TESS to use both forward and backward context.
3. SMOTE or focal-loss for Seattle's class imbalance instead of class_weight.
4. Hyper-parameter search (learning rate, hidden units) per model.

---

## 11. Final Numbers (one-page summary you can keep in front of you)

| Task | Best model | Test metric | What to say |
|---|---|---|---|
| TESS speech | **GRU** | 99.5 % acc, AUC 0.9998 | "Textbook RNN failure (60 %) vs gated cells (99 %+)." |
| Twitter sentiment | **GRU** | 92.3 % acc, AUC 0.982 | "Embedding does heavy lifting; GRU edges LSTM." |
| Delhi climate (regression) | **GRU/LSTM tie** | R² 0.85, MAE 1.85 °C | "Gated cells beat ANN+RNN by 30 % R²." |
| Seattle weather (classification) | **RNN** | 67 % acc, AUC 0.766 | "All beat 44 % majority baseline; class_weight backfired." |
| CORGIS weather (regression) | **RNN** (tied with LSTM/GRU) | R² 0.883, MAE 4.6 °F | "All recurrent tied at R² 0.88; ANN much worse." |

**Total compute time:** ~16 minutes of CPU training across all 5 notebooks.

---

Good luck tomorrow!
