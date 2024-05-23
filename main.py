# type: ignore
import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

app = Flask(__name__)

def load_data(file_name):
    """Load data from a CSV file."""
    try:
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, file_name)
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        return None

@app.route('/')
def index():
    """Render the index page with a list of symptoms."""
    data = load_data('training_data.csv')
    if data is None:
        return "Error: 'training_data.csv' file not found."
    symptoms = data.columns[:-2].tolist()
    return render_template('index.html', symptoms=symptoms)

@app.route('/symptoms', methods=['GET', 'POST'])
def symptoms():
    """Render the symptoms page with a list of symptoms."""
    data = load_data('training_data.csv')
    if data is None:
        return "Error: 'training_data.csv' file not found."
    symptoms = data.columns[:-2].tolist()
    return render_template('symptoms.html', symptoms=symptoms)

@app.route('/result', methods=['POST'])
def result():
    """Handle form submission, run classifiers, and return results."""
    training_data = load_data("training_data.csv")
    test_data = load_data("test_data.csv")
    
    if training_data is None or test_data is None:
        return "Error: 'training_data.csv' or 'test_data.csv' file not found."
    
    training_data = training_data.drop(["Unnamed: 133"], axis=1)
    X_train = training_data.drop('prognosis', axis=1)
    y_train = training_data['prognosis']
    
    X_test = test_data.drop('prognosis', axis=1)
    y_test = test_data['prognosis']
    
    # Collect selected symptoms from the form
    selected_symptoms = [key for key in X_train.columns if key in request.form]
    symptoms_vector = [1 if key in selected_symptoms else 0 for key in X_train.columns]
    
    # Prepare a dictionary to store results
    results = {}
    
    def evaluate_model(model, model_name):
        """Fit and evaluate a model, then store results."""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        results[model_name] = {
            "accuracy": accuracy,
            "conf_matrix": conf_matrix.tolist(),  # Convert to list for JSON serialization
            "class_report": class_report
        }
    
    # Evaluate each model
    evaluate_model(LogisticRegression(), "Logistic Regression")
    evaluate_model(DecisionTreeClassifier(), "Decision Tree")
    evaluate_model(RandomForestClassifier(), "Random Forest")
    evaluate_model(MLPClassifier(), "MLP Classifier")
    
    # Predict using the Logistic Regression model for demonstration
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict([symptoms_vector])
    
    # Extract the most predicted disease
    most_predict_disease = y_pred[0]
    
    return render_template('result.html', 
                            selected_symptoms=selected_symptoms,
                            results=results,
                            most_predict_disease=most_predict_disease)

@app.route('/get_symptoms', methods=['GET'])
def get_symptoms():
    """Return a JSON list of symptoms."""
    data = load_data('training_data.csv')
    if data is None:
        return jsonify({"error": "'training_data.csv' file not found."}), 404
    symptoms = data.columns[:-2].tolist()
    return jsonify(symptoms)

if __name__ == '__main__':
    app.run(debug=True)