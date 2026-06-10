# 🚗 CarVal — AI Car Price Prediction Platform

> An end-to-end Machine Learning web application that predicts used car prices across 11 Indian cities with **90% R² accuracy**.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GradientBoosting-orange?style=flat-square&logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/R²%20Accuracy-90%25-brightgreen?style=flat-square)
![Deploy](https://img.shields.io/badge/Deployed-Render-purple?style=flat-square)

---

## 🔗 Live Demo
**[https://car-price-prediction.onrender.com](https://car-price-prediction.onrender.com)**

---

## 📸 Preview

![CarVal Preview](https://via.placeholder.com/900x500/0a0a0f/6c63ff?text=CarVal+Preview)

---

## 🧠 About the Project

**CarVal** is an AI-powered used car price estimator built for the Indian market. It uses a trained Gradient Boosting ML model to predict fair market value based on 13 car features — helping buyers avoid overpaying and sellers price competitively.

### Key Features
- 🎯 **90% R² accuracy** — cross-validated across 10 folds
- 📍 **11 Indian cities** — Mumbai, Delhi, Bangalore, Pune, Chennai and more
- 🚗 **30 car brands** — 1,800+ models covered
- ⚡ **Auto-fill specs** — select a model and specs fill automatically
- 📊 **Market comparison** — shows real market low, average and high
- 📈 **Price meter** — visualizes where your car sits in the market
- 💰 **Dual currency** — prices in ₹ Lakhs + USD equivalent

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Gradient Boosting Regressor |
| R² Score | **0.8991** |
| Cross-Validated R² | **0.9008 ± 0.044** |
| MAE | ₹1,46,869 |
| Within ₹1 Lakh | 61.3% |
| Within ₹2 Lakhs | 81.2% |
| Training Data | 6,019 real listings |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Backend** | Python, Flask |
| **ML Model** | scikit-learn — GradientBoostingRegressor |
| **Data Processing** | Pandas, NumPy |
| **Deployment** | Render |
| **Version Control** | Git, GitHub |

---

## 📁 Project Structure

```
car-price-prediction/
│
├── application.py          # Flask backend — routes & prediction logic
├── GBModel_v2.pkl          # Trained Gradient Boosting model (90% R²)
├── meta_v2.json            # Brands, models, cities metadata
├── requirements.txt        # Python dependencies
├── Procfile                # Render deployment config
│
└── templates/
    └── index.html          # Full SaaS frontend UI
```

---

## ⚙️ Input Features

| Feature | Description |
|---------|-------------|
| Car Brand & Model | 30 brands, 1800+ models |
| Location | 11 Indian cities |
| Year | 1998 – 2019 |
| Kilometres Driven | 0 – 5,00,000 km |
| Fuel Type | Petrol, Diesel, CNG, LPG |
| Transmission | Manual / Automatic |
| Owner Type | First / Second / Third / Fourth |
| Mileage | kmpl |
| Engine | CC |
| Power | bhp |
| Seats | 2 – 10 |

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Anumishra02/car-price-prediction.git
cd car-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python application.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## 🔍 How It Works

```
User Input (13 features)
        ↓
Feature Engineering
(car_age, kms_per_year)
        ↓
OneHotEncoding (categorical features)
        ↓
GradientBoostingRegressor
(500 estimators, depth=5, lr=0.05)
        ↓
Predicted Price in ₹ Lakhs
        +
Market Comparison (Low / Avg / High)
```

---

## 📈 Model Training

The model was trained on a dataset of **6,019 real used car listings** from 11 Indian cities.

```python
model = make_pipeline(
    ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ], remainder='passthrough'),
    GradientBoostingRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
)
# Cross-validated R² = 0.9008
```

---

## 🙋‍♀️ Author

**Anu Mishra**
- 🌐 Portfolio: [anumishra.vercel.app](https://mac-os-portfolio-dun-seven.vercel.app/)
- 💼 LinkedIn: [AnuMishra](https://linkedin.com/in/anumish)
- 🐙 GitHub: [Anumishra02](https://github.com/Anumishra02)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
