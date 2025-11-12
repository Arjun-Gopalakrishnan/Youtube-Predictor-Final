from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import json

# Initialize Flask app
app = Flask(__name__)

# --- LOAD ALL YOUR MODEL FILES ---
try:
    model = joblib.load("best_model.joblib")
except FileNotFoundError:
    print("FATAL ERROR: 'best_model.joblib' not found.")
    model = None

try:
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)
except FileNotFoundError:
    print("FATAL ERROR: 'model_columns.json' not found.")
    model_columns = []

# --- NEW: Get channel names from the model_columns.json ---
# We extract the channel names from the one-hot encoded column names
# (e.g., "channel_name_CallMeShazzam TECH" -> "CallMeShazzam TECH")
def get_channel_names_from_cols(columns):
    names = []
    prefix = "channel_name_"
    for col in columns:
        if col.startswith(prefix):
            names.append(col[len(prefix):]) # Get the part after the prefix
    return names

channel_names_list = get_channel_names_from_cols(model_columns)
if not channel_names_list:
    print("FATAL ERROR: Could not find any 'channel_name_...' columns in model_columns.json")
# --- End of loading ---


@app.route('/')
def home():
    """Renders the home page."""
    # Pass the extracted channel names to the dropdown
    return render_template('index.html', channel_names=channel_names_list)


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the form."""
    
    # Check if model loaded correctly
    if model is None or not model_columns or not channel_names_list:
        error_message = "FATAL SERVER ERROR: Model or data files are missing. Check server logs."
        return render_template('index.html', prediction_text=error_message, channel_names=channel_names_list)

    try:
        # --- 1. Collect ALL raw inputs from form ---
        subscribers = float(request.form['subscribers'])
        video_count = float(request.form['video_count'])
        account_age = float(request.form['account_age'])
        post_frequency = float(request.form['post_frequency_per_year'])
        like_count = float(request.form['like_count'])
        comment_count = float(request.form['comment_count'])
        channel_name = request.form['channel_name'] # e.g., "CallMeShazzam TECH"

        # --- 2. Perform Feature Engineering (to match notebook) ---
        
        # Handle potential log(0) errors
        def safe_log10(x):
            if x <= 0:
                return 0
            return np.log10(x)

        # Create the engineered features the model expects
        subscriber_count_log = safe_log10(subscribers)
        like_count_log = safe_log10(like_count)
        comment_count_log = safe_log10(comment_count)
        
        # Use video_count as total_posts
        total_posts = video_count 
        
        # Use account_age as account_age_years
        account_age_years = account_age 
        
        # Use post_frequency as is
        post_frequency_per_year = post_frequency 
        
        # Calculate official_engagement_rate
        if subscribers > 0:
            official_engagement_rate = (like_count + comment_count) / subscribers
        else:
            official_engagement_rate = 0

        # --- 3. Build the final DataFrame ---
        
        # Create a dictionary to hold our single prediction row
        # Initialize all columns from model_columns.json to 0
        data_dict = {col: 0 for col in model_columns}

        # Set the values for the features we just calculated
        data_dict['subscriber_count_log'] = subscriber_count_log
        data_dict['like_count_log'] = like_count_log
        data_dict['comment_count_log'] = comment_count_log
        data_dict['total_posts'] = total_posts
        data_dict['account_age_years'] = account_age_years
        data_dict['post_frequency_per_year'] = post_frequency_per_year
        data_dict['official_engagement_rate'] = official_engagement_rate

        # Handle the one-hot encoding for the channel name
        selected_channel_col = f"channel_name_{channel_name}"
        if selected_channel_col in data_dict:
            data_dict[selected_channel_col] = 1
        else:
            print(f"Warning: Channel column '{selected_channel_col}' not found in model columns.")

        # Create the final DataFrame
        final_features_df = pd.DataFrame([data_dict])
        
        # Re-order columns to match the model's training order
        final_features_df = final_features_df[model_columns]

        # --- 4. Make prediction ---
        # Prediction is in log10(view_count)
        log_prediction = model.predict(final_features_df)[0]
        
        # --- 5. Inverse transform the prediction (CRITICAL!) ---
        # Convert log10(views) back to actual views (10^log_prediction)
        prediction = 10**log_prediction
        
        # Format the prediction for display
        prediction_output = f'Estimated View Count: {int(prediction):,}'

    except KeyError as e:
        # This will catch if a feature name is wrong
        prediction_output = f"Error: Feature mismatch. Missing: {e}. Please check app.py logic."
    except Exception as e:
        # Handle other errors (e.g., non-numeric input)
        prediction_output = f'Error: {str(e)}. Please check your inputs.'

    # Render the home page again, this time with the prediction text
    return render_template('index.html', prediction_text=prediction_output, channel_names=channel_names_list)


if __name__ == "__main__":
    # Run the app in debug mode
    print("Starting Flask server...")
    if model and model_columns and channel_names_list:
        print("Model and data files loaded successfully.")
        print(f"Found {len(channel_names_list)} channels.")
    else:
        print("FATAL ERROR: Could not load all model files. Check logs.")
    
    print("Access the app at http://127.0.0.1:5000")
    # This is the corrected line
    app.run(debug=True, host='0.0.0.0', port=5000)