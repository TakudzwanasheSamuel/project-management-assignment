# MSU Dining Analytics System

A machine learning-powered prediction and analytics system for Michigan State University dining services. This project combines a Flask REST API with an interactive Plotly Dash dashboard to predict meal service quantities and analyze dining patterns.

## 🎯 Overview

The system uses a trained Random Forest model to predict how many meals will be served based on meal components (starch, protein, side), day of week, semester, and academic events. It provides both programmatic access via API and interactive visualizations via dashboard.

## 📋 Features

### Backend API (`app.py`)
- **POST /predict** - Get meal predictions for given parameters
- **GET /history** - Retrieve historical dining data
- Input validation and preprocessing
- JSON request/response format

### Interactive Dashboard (`dashboard.py`)
- Real-time meal predictions with visual feedback
- Historical trends analysis
- Day-of-week patterns
- Semester comparisons
- Feature importance visualization
- Statistical overview cards
- Prediction comparison charts

## 📦 Project Structure

```
├── app.py                                 # Flask REST API
├── dashboard.py                           # Plotly Dash interactive dashboard
├── test_predict.py                        # API prediction test script
├── test_history.py                        # API history test script
├── msu_dining_2semesters.csv             # Historical dining data
├── random_forest_model.joblib            # Trained Random Forest model
├── categorical_unique_values.joblib      # Category mappings for preprocessing
├── X_columns.joblib                      # Expected feature column order
├── msu_dining.ipynb                      # Jupyter notebook (model training)
└── README.md                              # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone or download the project**
```bash
cd c:\Users\hp user\Downloads\model
```

2. **Install required packages**
```bash
pip install flask pandas numpy scikit-learn joblib dash plotly dash-bootstrap-components requests
```

### Running the API

Start the Flask server:
```bash
python app.py
```

Server runs on: **http://127.0.0.1:5000/**

Available endpoints:
- `POST http://127.0.0.1:5000/predict` - Make predictions
- `GET http://127.0.0.1:5000/history` - Get historical data

### Running the Dashboard

Start the Dash dashboard:
```bash
python dashboard.py
```

Dashboard runs on: **http://127.0.0.1:8050/**

## 📊 API Usage

### Prediction Endpoint

**Request:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Day_of_Week": "Monday",
    "Semester": 1,
    "Starch": "Sadza",
    "Protein": "Beef",
    "Side": "Salad",
    "Academic_Event": "Regular"
  }'
```

**Response:**
```json
{
  "Meals_Served_Prediction": 245.50
}
```

### History Endpoint

**Request:**
```bash
curl http://127.0.0.1:5000/history
```

**Response:**
```json
[
  {
    "Date": "2024-01-15",
    "Day_of_Week": "Monday",
    "Semester": 1,
    "Starch": "Rice",
    "Protein": "Chicken",
    "Side": "Vegetables",
    "Academic_Event": "Regular",
    "Meals_Served": 250
  },
  ...
]
```

## 🧪 Testing

### Test Prediction API

```bash
python test_predict.py
```

### Test History API

```bash
python test_history.py
```

## 📈 Dashboard Features

### 1. Prediction Interface
Input form with dropdowns for all meal parameters:
- **Day of Week** - Monday through Sunday
- **Semester** - 1 or 2
- **Starch** - Available options from historical data
- **Protein** - Available options from historical data
- **Side** - Available options from historical data
- **Academic Event** - Event type classification

### 2. Visualizations

| Chart | Purpose |
|-------|---------|
| **Prediction Comparison** | Compare your prediction vs. historical averages |
| **Historical Trends** | Time-series of meals served over time |
| **Day Analysis** | Average meals by day of week |
| **Semester Analysis** | Plant patterns between semesters |
| **Feature Importance** | Which inputs most impact predictions |
| **Statistics Cards** | Key metrics: avg, max, min, total records |

## 🔧 Input Parameters

### Valid Values

**Day_of_Week:** Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday

**Semester:** 1, 2

**Starch (examples):** Sadza, Rice, Bread, Potato, Pasta

**Protein (examples):** Beef, Chicken, Fish, Pork, Vegetarian

**Side (examples):** Salad, Vegetables, Beans, Corn, Greens

**Academic_Event:** Regular, Exam, Holiday, Break, Special

## 🎓 Model Details

- **Algorithm:** Random Forest Classifier
- **Training Data:** 2 semesters of dining data
- **Features:** Categorical (one-hot encoded) and numerical
- **Output:** Continuous prediction of meals served

## 📝 Data Preprocessing

The system automatically:
- Converts categorical variables to one-hot encoded format
- Applies alphabetically-based drop-first encoding
- Maintains feature column order from training
- Handles missing categories with False/0 values

## 🐛 Troubleshooting

**Error: Model files not found**
- Ensure `random_forest_model.joblib`, `categorical_unique_values.joblib`, and `X_columns.joblib` are in the same directory

**Error: Connection refused**
- Check if Flask app is running on port 5000
- Check if Dash is running on port 8050

**Error: CSV file not found**
- Ensure `msu_dining_2semesters.csv` is in the project directory

## 📦 Dependencies

```
flask==3.0.0+
pandas==2.0.0+
numpy==1.24.0+
scikit-learn==1.3.0+
joblib==1.3.0+
dash==2.14.0+
plotly==5.17.0+
dash-bootstrap-components==1.5.0+
requests==2.31.0+
```

## 🔐 Notes

- The model is pre-trained and loaded from joblib files
- The system is read-only for predictions (no model retraining in this app)
- Historical data is loaded fresh on each history request
- Dashboard stores predictions in browser memory for comparison

## 📄 License

This project is for educational purposes at Michigan State University.

## 👤 Author

Created as part of project management assignment

## 🤝 Contributing

To modify the model or add features:
1. Update the Jupyter notebook (`msu_dining.ipynb`)
2. Retrain and save new joblib files
3. Restart the Flask/Dash servers

---

**Last Updated:** March 2026
