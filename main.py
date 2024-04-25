# type: ignore
import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier 

app = Flask(__name__)

@app.route('/')
def index():
    try:
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'training_data.csv')
        data = pd.read_csv(file_path)
        symptoms = data.columns[:-2].tolist()
    except FileNotFoundError:
        return "Error: 'training_data.csv' file not found."
    return render_template('index.html', symptoms=symptoms)

@app.route('/symptoms', methods=['GET', 'POST'])
def symptoms():
    if request.method == 'POST':
        # Load symptom values from training_data.csv
        try:
            current_dir = os.path.dirname(__file__)
            file_path = os.path.join(current_dir, 'training_data.csv')
            data = pd.read_csv(file_path)
            symptoms = data['Symptoms'].tolist()
        except FileNotFoundError:
            return "Error: 'training_data.csv' file not found."
    return render_template('symptoms.html')

@app.route('/result', methods=['GET','POST'])
def result():
    if request.method == 'POST':
        data = pd.read_csv("training_data.csv")
        test = pd.read_csv("test_data.csv")

        data1 = data.drop(["Unnamed: 133"], axis=1)

        X_train = data1.drop('prognosis', axis=1)
        y_train = data1['prognosis']

        X_test = test.drop('prognosis', axis=1)
        y_test = test['prognosis']

        # Collect symptoms from the form
        symptoms = []
        for key in X_train.columns:
            if key in request.form:
                symptoms.append(1)
            else:
                symptoms.append(0)

        # Filter test data for the selected symptoms
        X_test_filtered = X_test[X_train.columns]
        X_test_filtered = X_test_filtered.astype(int)  # Convert to integer

        if len(X_test_filtered) == 0:
            return "Error: No symptoms selected for prediction."

        # Convert X_test_filtered to the desired format
        X_test_filtered_string = ','.join(map(str, X_test_filtered.iloc[0].tolist()))

        # Logistic Regression
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred_model = model.predict(X_test_filtered)
        accuracy_model = accuracy_score(y_test, y_pred_model)
        conf_matrix_model = confusion_matrix(y_test, y_pred_model)
        classification_report_model = classification_report(y_test, y_pred_model)

        # Decision Tree Classifier
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred_clf = clf.predict(X_test_filtered)
        accuracy_clf = accuracy_score(y_test, y_pred_clf)
        conf_matrix_clf = confusion_matrix(y_test, y_pred_clf)
        classification_report_clf = classification_report(y_test, y_pred_clf)

        # Random Forest
        model_random_forest = RandomForestClassifier()
        model_random_forest.fit(X_train, y_train)
        y_pred_random_forest = model_random_forest.predict(X_test_filtered)
        accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)
        conf_matrix_random_forest = confusion_matrix(y_test, y_pred_random_forest)
        class_report_random_forest = classification_report(y_test, y_pred_random_forest)

        # Neural Network (Multi-layer Perceptron)
        model_mlp = MLPClassifier()
        model_mlp.fit(X_train, y_train)
        y_pred_mlp = model_mlp.predict(X_test_filtered)
        accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
        conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
        class_report_mlp = classification_report(y_test, y_pred_mlp)

        # Extracting precision, recall, f1-score, and support
        lines = classification_report_model.split('\n')
        data = [line.split() for line in lines[2:-5]]
        diseases = [line[0] for line in data]
        support = [int(line[-1]) for line in data]

        # Finding the disease with the highest support
        max_support_index = np.argmax(support)
        most_predict_disease = diseases[max_support_index]

        return render_template('result.html', 
                                symptoms=symptoms,
                                X_test_filtered=X_test_filtered_string,
                                accuracy_model=accuracy_model,
                                conf_matrix_model=conf_matrix_model,
                                classification_report_model=classification_report_model,
                                accuracy_clf=accuracy_clf,
                                conf_matrix_clf=conf_matrix_clf,
                                classification_report_clf=classification_report_clf,
                                accuracy_random_forest=accuracy_random_forest,
                                conf_matrix_random_forest=conf_matrix_random_forest,
                                class_report_random_forest=class_report_random_forest,
                                accuracy_mlp=accuracy_mlp,
                                conf_matrix_mlp=conf_matrix_mlp,
                                class_report_mlp=class_report_mlp,
                                most_predict_disease=most_predict_disease)

@app.route('/get_symptoms', methods=['GET'])
def get_symptoms():
    try:
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'training_data.csv')
        data = pd.read_csv(file_path)
        symptoms = data.columns[:-2].tolist()
    except FileNotFoundError:
        return jsonify({"error": "'training_data.csv' file not found."}), 404
    return jsonify(symptoms)

if __name__ == '__main__':
    app.run(debug=True)