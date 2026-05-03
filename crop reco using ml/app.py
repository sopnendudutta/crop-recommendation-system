from flask import Flask, request, render_template
import numpy as np
import pickle
import json

with open('crop_dict.json', 'w') as f:
    json.dump(crop_dict, f)

# Load the model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

# Route to new modern UI
@app.route('/')
def home():
    return render_template("index.html", result=None)

# Prediction route
@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Fetch form data
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Preprocessing
        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        scaled = ms.transform(features)
        final = sc.transform(scaled)
        prediction = model.predict(final)

        # Crop label mapping
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        crop_name = crop_dict.get(prediction[0], "Unknown")
        result = f"{crop_name} is the best crop to be cultivated right now."

    except Exception as e:
        result = "An error occurred while processing the input. Please try again."

    # Re-render page with result
    return render_template("index.html", result=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
