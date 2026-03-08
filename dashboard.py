import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load model and preprocessing data
model = joblib.load('random_forest_model.joblib')
unique_cat_values = joblib.load('categorical_unique_values.joblib')
model_features = joblib.load('X_columns.joblib')

# Load historical data
df_history = pd.read_csv('msu_dining_2semesters.csv')
df_history['Date'] = pd.to_datetime(df_history['Date'])

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "MSU Dining Analytics Dashboard"

# Preprocessing function (same as in app.py)
def preprocess_input(input_data):
    df_input = pd.DataFrame([input_data])
    
    if 'Semester' in df_input.columns:
        df_input['Semester'] = df_input['Semester'].astype(int)
    
    processed_cols = []
    
    if 'Semester' in model_features:
        if 'Semester' in df_input.columns:
            processed_cols.append(df_input[['Semester']])
    
    for col_name, categories in unique_cat_values.items():
        if col_name in df_input.columns:
            sorted_categories = sorted(categories)
            category_to_drop = sorted_categories[0]
            expected_full_dummy_cols = [f"{col_name}_{cat}" for cat in categories]
            dummies = pd.get_dummies(df_input[col_name], prefix=col_name, dtype=bool)
            dummies = dummies.reindex(columns=expected_full_dummy_cols, fill_value=False)
            
            dummy_col_to_drop = f"{col_name}_{category_to_drop}"
            if dummy_col_to_drop in dummies.columns:
                dummies = dummies.drop(columns=[dummy_col_to_drop])
            
            processed_cols.append(dummies)
        else:
            sorted_categories = sorted(categories)
            category_to_drop = sorted_categories[0]
            missing_dummy_cols = [f"{col_name}_{cat}" for cat in categories if f"{col_name}_{cat}" != f"{col_name}_{category_to_drop}"]
            missing_df = pd.DataFrame(False, index=df_input.index, columns=missing_dummy_cols)
            processed_cols.append(missing_df)
    
    if processed_cols:
        final_processed_df = pd.concat(processed_cols, axis=1)
    else:
        final_processed_df = pd.DataFrame(index=df_input.index)
    
    final_prediction_df = final_processed_df.reindex(columns=model_features, fill_value=False)
    
    for col in final_prediction_df.columns:
        if final_prediction_df[col].dtype == 'bool':
            final_prediction_df[col] = final_prediction_df[col].astype(int)
    
    return final_prediction_df

# Get unique values for dropdowns
days_of_week = sorted(df_history['Day_of_Week'].unique())
semesters = sorted(df_history['Semester'].unique())
starches = sorted(df_history['Starch'].unique())
proteins = sorted(df_history['Protein'].unique())
sides = sorted(df_history['Side'].unique())
academic_events = sorted(df_history['Academic_Event'].unique())

# App layout
app.layout = html.Div([
    dcc.Store(id='predictions-store', data=[]),
    
    # Header
    html.Div([
        html.H1("🍽️ MSU Dining Analytics Dashboard", className="text-center mb-4"),
        html.P("Predict meal service quantities and analyze dining patterns", className="text-center text-muted")
    ], className="p-4 bg-light border-bottom"),
    
    # Main content
    html.Div([
        # Prediction Input Section
        html.Div([
            html.H3("Make a Prediction", className="mb-4"),
            html.Div([
                html.Div([
                    html.Label("Day of Week:"),
                    dcc.Dropdown(id='day-dropdown', options=[{'label': d, 'value': d} for d in days_of_week], 
                                value=days_of_week[0], clearable=False)
                ], className="col-md-2"),
                html.Div([
                    html.Label("Semester:"),
                    dcc.Dropdown(id='semester-dropdown', options=[{'label': int(s), 'value': int(s)} for s in semesters],
                                value=int(semesters[0]), clearable=False)
                ], className="col-md-2"),
                html.Div([
                    html.Label("Starch:"),
                    dcc.Dropdown(id='starch-dropdown', options=[{'label': s, 'value': s} for s in starches],
                                value=starches[0], clearable=False)
                ], className="col-md-2"),
                html.Div([
                    html.Label("Protein:"),
                    dcc.Dropdown(id='protein-dropdown', options=[{'label': p, 'value': p} for p in proteins],
                                value=proteins[0], clearable=False)
                ], className="col-md-2"),
                html.Div([
                    html.Label("Side:"),
                    dcc.Dropdown(id='side-dropdown', options=[{'label': s, 'value': s} for s in sides],
                                value=sides[0], clearable=False)
                ], className="col-md-2"),
                html.Div([
                    html.Label("Academic Event:"),
                    dcc.Dropdown(id='event-dropdown', options=[{'label': e, 'value': e} for e in academic_events],
                                value=academic_events[0], clearable=False)
                ], className="col-md-2"),
            ], className="row g-2 mb-3"),
            
            html.Button("Predict", id="predict-btn", n_clicks=0, className="btn btn-primary btn-lg"),
            html.Span(id="predict-error", className="text-danger ms-2")
        ], className="p-4 bg-white border rounded mb-4"),
        
        # Prediction Results
        html.Div([
            html.Div([
                html.Div(id="prediction-result", className="text-center")
            ], className="col-md-6"),
            html.Div([
                dcc.Graph(id="comparison-chart")
            ], className="col-md-6"),
        ], className="row mb-4"),
        
        # Charts Row 1
        html.Div([
            html.Div([
                dcc.Graph(id="historical-trend-chart")
            ], className="col-md-6"),
            html.Div([
                dcc.Graph(id="day-analysis-chart")
            ], className="col-md-6"),
        ], className="row mb-4"),
        
        # Charts Row 2
        html.Div([
            html.Div([
                dcc.Graph(id="semester-analysis-chart")
            ], className="col-md-6"),
            html.Div([
                dcc.Graph(id="feature-importance-chart")
            ], className="col-md-6"),
        ], className="row mb-4"),
        
        # Statistics Cards
        html.Div([
            html.Div([
                html.H5("Average Meals Served"),
                html.H3(f"{df_history['Meals_Served'].mean():.0f}")
            ], className="col-md-3 p-4 bg-info text-white rounded"),
            html.Div([
                html.H5("Max Meals Served"),
                html.H3(f"{df_history['Meals_Served'].max():.0f}")
            ], className="col-md-3 p-4 bg-success text-white rounded"),
            html.Div([
                html.H5("Min Meals Served"),
                html.H3(f"{df_history['Meals_Served'].min():.0f}")
            ], className="col-md-3 p-4 bg-warning text-white rounded"),
            html.Div([
                html.H5("Total Records"),
                html.H3(f"{len(df_history)}")
            ], className="col-md-3 p-4 bg-secondary text-white rounded"),
        ], className="row g-3 mb-4"),
        
    ], className="p-4")
], style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh'})

# Callbacks
@callback(
    [Output('prediction-result', 'children'),
     Output('predictions-store', 'data'),
     Output('predict-error', 'children')],
    Input('predict-btn', 'n_clicks'),
    [State('day-dropdown', 'value'),
     State('semester-dropdown', 'value'),
     State('starch-dropdown', 'value'),
     State('protein-dropdown', 'value'),
     State('side-dropdown', 'value'),
     State('event-dropdown', 'value')],
    prevent_initial_call=True
)
def make_prediction(n_clicks, day, semester, starch, protein, side, event):
    try:
        input_data = {
            'Day_of_Week': day,
            'Semester': semester,
            'Starch': starch,
            'Protein': protein,
            'Side': side,
            'Academic_Event': event
        }
        
        processed_input = preprocess_input(input_data)
        prediction = model.predict(processed_input)[0]
        
        predictions_store = {
            'inputs': input_data,
            'prediction': round(float(prediction), 2),
            'timestamp': datetime.now().isoformat()
        }
        
        result_card = html.Div([
            html.Div([
                html.H4("Prediction Result", className="text-muted"),
                html.H1(f"{prediction:.0f} meals", className="text-primary fw-bold"),
                html.P(f"Confidence: {prediction:.2f} meals served", className="text-muted")
            ], className="p-4 bg-white border rounded text-center")
        ])
        
        return result_card, [predictions_store], ""
    except Exception as e:
        return html.Div(), [], f"Error: {str(e)}"

@callback(
    Output('comparison-chart', 'figure'),
    Input('predictions-store', 'data')
)
def update_comparison(predictions_data):
    if not predictions_data:
        # Show empty placeholder
        return go.Figure().add_annotation(text="Make a prediction to see comparison")
    
    pred = predictions_data[-1] if isinstance(predictions_data, list) else predictions_data
    
    # Get historical average for the same day and semester
    day = pred['inputs']['Day_of_Week']
    semester = pred['inputs']['Semester']
    
    historical_subset = df_history[
        (df_history['Day_of_Week'] == day) & 
        (df_history['Semester'] == semester)
    ]
    
    comparison_data = {
        'Category': ['Your Prediction', f'Avg for {day} (Sem {semester})', 'Overall Average'],
        'Meals Served': [
            pred['prediction'],
            historical_subset['Meals_Served'].mean() if len(historical_subset) > 0 else 0,
            df_history['Meals_Served'].mean()
        ]
    }
    
    fig = px.bar(comparison_data, x='Category', y='Meals Served', 
                 title="Prediction vs Historical Average",
                 color='Meals Served', color_continuous_scale='Viridis')
    fig.update_layout(height=400, showlegend=False)
    return fig

@callback(
    Output('historical-trend-chart', 'figure'),
    Input('predictions-store', 'data')
)
def update_historical_trend(_):
    df_sorted = df_history.sort_values('Date')
    
    fig = px.line(df_sorted, x='Date', y='Meals_Served',
                  title='Meals Served Over Time',
                  markers=True, line_shape='linear')
    fig.update_layout(height=400, hovermode='x unified')
    return fig

@callback(
    Output('day-analysis-chart', 'figure'),
    Input('predictions-store', 'data')
)
def update_day_analysis(_):
    df_day = df_history.groupby('Day_of_Week')['Meals_Served'].agg(['mean', 'std', 'count']).reset_index()
    
    # Order days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_day['Day_of_Week'] = pd.Categorical(df_day['Day_of_Week'], categories=day_order, ordered=True)
    df_day = df_day.sort_values('Day_of_Week')
    
    fig = px.bar(df_day, x='Day_of_Week', y='mean',
                 error_y='std',
                 title='Average Meals Served by Day of Week',
                 labels={'mean': 'Average Meals', 'Day_of_Week': 'Day'})
    fig.update_layout(height=400, showlegend=False)
    return fig

@callback(
    Output('semester-analysis-chart', 'figure'),
    Input('predictions-store', 'data')
)
def update_semester_analysis(_):
    df_sem = df_history.groupby('Semester')['Meals_Served'].agg(['mean', 'std']).reset_index()
    
    fig = px.bar(df_sem, x='Semester', y='mean',
                 error_y='std',
                 title='Average Meals Served by Semester',
                 labels={'mean': 'Average Meals'},
                 color='Semester', color_continuous_scale='Blues')
    fig.update_layout(height=400, showlegend=False)
    return fig

@callback(
    Output('feature-importance-chart', 'figure'),
    Input('predictions-store', 'data')
)
def update_feature_importance(_):
    # Extract feature importance from RandomForest model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = model_features
        
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(15)  # Top 15
        
        fig = px.bar(df_importance, x='Importance', y='Feature',
                     orientation='h', title='Top 15 Feature Importances')
        fig.update_layout(height=400, showlegend=False)
    else:
        fig = go.Figure().add_annotation(text="Feature importance not available")
    
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)
