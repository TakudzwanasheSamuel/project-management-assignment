# Dash Dashboard Setup

## Installation

Install the required dashboard dependencies:

```bash
pip install dash plotly pandas numpy scikit-learn joblib
```

## Running the Dashboard

Start the dashboard server:

```bash
python dashboard.py
```

The dashboard will be available at: **http://127.0.0.1:8050/**

## Features

### 1. **Prediction Interface**
   - Input form with dropdowns for all meal parameters
   - Real-time predictions using the Random Forest model
   - Instant results display

### 2. **Prediction vs Historical Average**
   - Compares your prediction against historical averages
   - Shows average for the same day and semester
   - Shows overall average

### 3. **Historical Trends**
   - Time-series chart of all meals served
   - Helps visualize seasonal patterns

### 4. **Day of Week Analysis**
   - Bar chart with average meals by day
   - Shows standard deviation for variability

### 5. **Semester Analysis**
   - Compares meal service between semesters
   - Identifies semester-based patterns

### 6. **Feature Importance**
   - Shows which input features have the most impact on predictions
   - Based on Random Forest model's internal calculations

### 7. **Dashboard Statistics**
   - Overview cards with key metrics:
     - Average meals served
     - Maximum meals served
     - Minimum meals served
     - Total historical records

## Files Used

- `dashboard.py` - Main Dash application
- `random_forest_model.joblib` - Trained model
- `categorical_unique_values.joblib` - Category mappings
- `X_columns.joblib` - Expected feature columns
- `msu_dining_2semesters.csv` - Historical data

## Browser Compatibility

Works best with modern browsers (Chrome, Firefox, Safari, Edge)
