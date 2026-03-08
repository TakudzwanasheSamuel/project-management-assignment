from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np # Import numpy for potential use if needed by pandas/joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model, unique categorical values, and feature column order
# These files are expected to be in the same directory as app.py
try:
    model = joblib.load('random_forest_model.joblib')
    unique_cat_values = joblib.load('categorical_unique_values.joblib')
    model_features = joblib.load('X_columns.joblib') # Load the saved column order
except FileNotFoundError:
    print("Error: Model or preprocessing files not found. Ensure 'random_forest_model.joblib', 'categorical_unique_values.joblib', and 'X_columns.joblib' are in the same directory.")
    # Exit or handle error appropriately in a production environment
    exit(1)

# Helper function for preprocessing input (from previous step)
def preprocess_input(input_data):
    # Create a DataFrame from the input dictionary, ensuring it has an index
    df_input = pd.DataFrame([input_data])

    # Ensure 'Semester' is an integer type
    if 'Semester' in df_input.columns:
        df_input['Semester'] = df_input['Semester'].astype(int)

    # Collect all processed columns into a list
    processed_cols = []

    # Add 'Semester' column if it's an expected feature
    if 'Semester' in model_features:
        # Ensure 'Semester' is present in df_input before trying to select it
        if 'Semester' in df_input.columns:
            processed_cols.append(df_input[['Semester']])
        else:
            # If Semester is missing from input, add a default (e.g., 0 or mean/mode)
            # For this context, let's assume it should be provided or handle as missing in a general way
            # For now, if missing, it will be handled by reindex fill_value=False (which will become 0)
            pass

    for col_name, categories in unique_cat_values.items():
        if col_name in df_input.columns: # If the input data contains this categorical column
            # Sort categories alphabetically to determine which one was dropped by drop_first=True
            sorted_categories = sorted(categories)
            category_to_drop = sorted_categories[0] # Alphabetically first category

            # Create dummies for the input value. Use dtype=bool for consistency.
            # We use `reindex` with `columns` based on all possible dummy columns to ensure
            # all expected columns (even if not present in the single input row) are created and filled with False.
            expected_full_dummy_cols = [f"{col_name}_{cat}" for cat in categories]
            dummies = pd.get_dummies(df_input[col_name], prefix=col_name, dtype=bool)
            dummies = dummies.reindex(columns=expected_full_dummy_cols, fill_value=False)

            # Drop the dummy column corresponding to the first category (alphabetically)
            dummy_col_to_drop = f"{col_name}_{category_to_drop}"
            if dummy_col_to_drop in dummies.columns:
                dummies = dummies.drop(columns=[dummy_col_to_drop])

            processed_cols.append(dummies)
        else:
            # If a categorical column is missing from the input, we need to add all its
            # corresponding dummy columns (except the dropped one) as False.
            sorted_categories = sorted(categories)
            category_to_drop = sorted_categories[0]

            missing_dummy_cols = [f"{col_name}_{cat}" for cat in categories if f"{col_name}_{cat}" != f"{col_name}_{category_to_drop}"]

            # Create a DataFrame of False for these missing dummy columns, ensuring proper index
            missing_df = pd.DataFrame(False, index=df_input.index, columns=missing_dummy_cols)
            processed_cols.append(missing_df)

    # Concatenate all processed columns
    if processed_cols:
        final_processed_df = pd.concat(processed_cols, axis=1)
    else:
        # If no features processed (e.g., empty input), create an empty DataFrame matching model_features
        final_processed_df = pd.DataFrame(index=df_input.index)

    # Ensure the order of columns matches the training data (`model_features`)
    # Reindex to `model_features`. Any column in `model_features` not in `final_processed_df`
    # will be filled with False (or 0 if converted to int). Any column in `final_processed_df`
    # not in `model_features` will be dropped.
    final_prediction_df = final_processed_df.reindex(columns=model_features, fill_value=False)

    # Convert boolean columns to integer (0/1) for model input
    for col in final_prediction_df.columns:
        if final_prediction_df[col].dtype == 'bool':
            final_prediction_df[col] = final_prediction_df[col].astype(int)

    return final_prediction_df


# Existing base route
@app.route('/')
def hello_world():
    return 'Hello, World!'

# Existing prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        processed_input = preprocess_input(data)
        prediction = model.predict(processed_input)
        return jsonify({'Meals_Served_Prediction': round(float(prediction[0]), 2)})

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

# New historical data endpoint
@app.route('/history', methods=['GET'])
def get_historical_data():
    try:
        # Load the dataset (or reload if the app context is short-lived)
        df_history = pd.read_csv('msu_dining_2semesters.csv')
        df_history['Date'] = pd.to_datetime(df_history['Date'])

        # Select relevant columns
        historical_cols = ['Date', 'Day_of_Week', 'Semester', 'Starch', 'Protein', 'Side', 'Academic_Event', 'Meals_Served']
        df_selected_history = df_history[historical_cols].copy()

        # Convert 'Date' column to string for JSON serialization
        df_selected_history['Date'] = df_selected_history['Date'].dt.strftime('%Y-%m-%d')

        # Convert DataFrame to a list of dictionaries (JSON format)
        historical_data_json = df_selected_history.to_dict(orient='records')

        return jsonify(historical_data_json)

    except FileNotFoundError:
        return jsonify({'error': "Historical data file 'msu_dining_2semesters.csv' not found."}), 404
    except Exception as e:
        app.logger.error(f"Historical data endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
