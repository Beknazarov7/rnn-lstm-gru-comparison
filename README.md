# ML Final Project — Sequence Modelling on Speech, Text & Time-Series

A comparison of recurrent deep-learning architectures (**ANN / SimpleRNN / LSTM / GRU**) across three problem families: speech-emotion classification, text-sentiment analysis, and time-series forecasting/classification.

Every model in a given task shares an **identical backbone** — only the recurrent cell type changes — so any performance gap is attributable to the cell choice alone.

## Results at a glance

| Task | Data | Best model | Headline metric |
|------|------|-----------|-----------------|
| **1 — TESS speech emotion** | 2,800 audio clips, 7 emotions | **GRU** | 99.5% acc · AUC 0.9998 |
| **2 — Twitter entity sentiment** | ~69K cleaned tweets, 4 classes | **GRU** | 92.3% acc · AUC 0.982 |
| **3a — Daily Delhi climate** (regression) | 1,575 daily records | **GRU / LSTM** | R² 0.85 · MAE 1.85 °C |
| **3b — Seattle weather** (classification) | 1,461 records, 5 classes (imbalanced) | **RNN** | 66.7% acc · AUC 0.766 |
| **3c — CORGIS multi-city weather** (regression) | 307 cities × ~53 weeks | **RNN** | R² 0.883 · MAE 4.6 °F |

Two findings worth a look: SimpleRNN's textbook failure on long sequences (60% vs 99%+ for gated cells on TESS's 130 time-steps), and the opposite case where the lighter RNN *beats* LSTM/GRU on small, short-sequence data (Seattle, CORGIS) because the gated cells overfit. Full discussion in [`report/PRESENTATION_REPORT.md`](report/PRESENTATION_REPORT.md).

## Project structure

```
.
├── task1_tess_emotion/          # Speech emotion (MFCC features → RNN/LSTM/GRU)
│   └── tess_emotion_classification.ipynb
├── task2_twitter_sentiment/     # Text sentiment (Embedding → RNN/LSTM/GRU)
│   └── twitter_sentiment_classification.ipynb
├── task3_weather_timeseries/    # Sliding-window forecasting & classification
│   ├── 3a_daily_delhi_climate.ipynb   # regression
│   ├── 3b_seattle_weather.ipynb       # imbalanced classification (ROC-AUC)
│   └── 3c_corgis_weather.ipynb        # multi-city regression
├── report/PRESENTATION_REPORT.md      # Full methodology, results & Q&A notes
├── data/                        # Datasets go here (not committed — see below)
└── requirements.txt
```

Trained `.keras` model files are committed alongside each notebook.

## Methodology highlights

- **Splits:** stratified 70/15/15 for classification; **chronological** for time-series (the model never sees future data).
- **Regularisation:** Dropout (0.2–0.3), EarlyStopping (restore best weights), ReduceLROnPlateau; SpatialDropout1D for the text task.
- **No leakage:** all scalers fit on **training statistics only**.
- **Metrics:** accuracy / macro & weighted F1 / ROC-AUC for classification; MAE / RMSE / R² for regression.

## Datasets

The `data/` folder is intentionally empty in version control. Download the datasets separately:

| Task | Dataset | Source |
|------|---------|--------|
| 1 | Toronto Emotional Speech Set (TESS) | Kaggle: `ejlok1/toronto-emotional-speech-set-tess` |
| 2 | Twitter Entity Sentiment Analysis | Kaggle: `jp797498e/twitter-entity-sentiment-analysis` |
| 3a | Daily Delhi Climate | Kaggle: `sumanthvrao/daily-climate-time-series-data` |
| 3b | Seattle Weather | Kaggle: `ananthr1/weather-prediction` |
| 3c | CORGIS Weather | https://corgis-edu.github.io/corgis/csv/weather/ |

Place the files where each notebook's loading cell expects them (paths are at the top of each notebook).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Then open any notebook and run all cells. Total training time across all five notebooks is ~16 minutes on CPU.
