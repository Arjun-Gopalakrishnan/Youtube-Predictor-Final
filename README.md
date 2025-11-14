YouTube View Count Predictor

🚀 Live Demo

You can try the live, deployed version of this application here:

https://youtube-predictor-final.onrender.com

(Note: The app runs on a free service, so it may take 30-60 seconds to "wake up" on the first visit.)

This is the final submission for a machine learning project that predicts YouTube view counts for Malayalam Tech YouTubers.

This repository contains a complete, end-to-end data science project: from data collection and model training in a Jupyter Notebook to a fully functional web application built with Flask and an embedded Tableau dashboard for data analysis.

✨ Features

ML Prediction: Uses a trained XGBoost Regressor model (best_model.joblib) to predict video view counts based on channel statistics.

Flask Backend (app.py): A robust Python backend that:

Loads the model and all required data files.

Performs all necessary feature engineering (log transforms, one-hot encoding) in real-time.

Calculates the inverse transform (10**prediction) to show the real view count.

Web Interface (index.html): A clean, multi-part UI:

A professional cover page to introduce the app.

A "Predictor" tab with a form for user inputs.

Automatic calculation of 'Post Frequency' using JavaScript.

A "Dashboard" tab with an embedded, interactive Tableau dashboard.

Live Data Dashboard: The "Dashboard" tab features a live Tableau Public dashboard built from the original malayalam_youtube_tech_data_final.csv dataset, showing channel comparisons and feature relationships.

🔧 How to Run This Project Locally

To run this web application on your own computer, follow these steps:

Clone the repository:

git clone [https://github.com/Arjun-Gopalakrishnan/Youtube-Predictor-Final.git](https://github.com/Arjun-Gopalakrishnan/Youtube-Predictor-Final.git)
cd Youtube-Predictor-Final


Create a virtual environment (Recommended):

python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate # On Mac/Linux


Install the required libraries:

pip install -r requirements.txt


Run the Flask app:

python app.py


Open the app in your browser:

The terminal will show Access the app at http://127.0.0.1:5000.

Open that link in your browser to see the live application.

📂 Key Files in This Repository

app.py: The "brain" of the app. The Python Flask server that handles all logic and prediction.

templates/index.html: The "face" of the app. The complete frontend with HTML, CSS, JavaScript, and the Tableau embed code.

best_model.joblib: The final, trained XGBoost Regressor model.

model_columns.json: A JSON file containing the list of all feature columns the model was trained on.

channel_join_year.json / total_posts_per_channel.json: Data files used by app.py for feature engineering.

Data_Cleaning_and_Analysis_Final.ipynb: The Jupyter Notebook showing the complete process of data cleaning, feature engineering, and model training (R² score: ~0.82).

malayalam_youtube_tech_data_final.csv: The original, raw dataset used for training and for the Tableau dashboard.

requirements.txt: A list of all Python libraries needed to run the project.
