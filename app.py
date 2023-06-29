from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Load the trained model
    model = joblib.load('linear_regression_model.pkl')

    # Get the JSON data from the request
    data = request.json

    # Preprocess the input data
    X = pd.DataFrame(data)
    X = X[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']]

    # Make predictions
    predictions = model.predict(X)

    # Prepare the response JSON
    response = {'predictions': predictions.tolist()}

    return jsonify(response)

if __name__ == '__main__':
    # Load the dataset
    data = pd.read_excel('UCI_Real_Estate_Valuation.xlsx')
    data = data.drop('No', axis=1)
    X = data[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']]
    y = data['Y house price of unit area']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print('Mean Squared Error:', mse)
    print('R-squared:', r2)
    print('Mean Absolute Error:', mae)
    print('Root Mean Squared Error:', rmse)

    # Save the trained model
    joblib.dump(model, 'linear_regression_model.pkl')

    # Start the Flask app
    app.run(debug=True)
