from flask import Flask, request, render_template
import numpy as np
import pickle
import json

# 1. Define the dictionary at the TOP (Global Scope)
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# 2. Now you can safely save it to JSON
with open('crop_dict.json', 'w') as f:
    json.dump(crop_dict, f)

# 3. Load the model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html", result=None)

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

        # Accessing the global crop_dict[cite: 3]
        crop_name = crop_dict.get(prediction[0], "Unknown")
        result = f"{crop_name} is the best crop to be cultivated right now."

    except Exception as e:
        # It's helpful to print the error to your terminal for debugging
        print(f"Error: {e}")
        result = "An error occurred while processing the input. Please try again."

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)