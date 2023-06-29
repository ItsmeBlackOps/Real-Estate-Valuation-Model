import requests

# Define the API endpoint URL
url = 'http://127.0.0.1:5000/predict'

# Define the input data
data = {
    "X2 house age": 15,
    "X3 distance to the nearest MRT station": 500,
    "X4 number of convenience stores": 3
}

# Send a POST request to the API
response = requests.post(url, json=data)

# Check the response status code
if response.status_code == 200:
    # Get the predictions from the response JSON
    predictions = response.json()['predictions']

    # Print the predictions
    print("Predictions:", predictions)
else:
    print("Error:", response.text)
