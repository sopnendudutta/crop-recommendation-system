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

# 2. Save to JSON
with open('crop_dict.json', 'w') as f:
    json.dump(crop_dict, f)

# 3. Load the model and scalers[cite: 2, 5, 6]
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)

# Data for Feature #4: Fertilizer Logic
crop_requirements = {
    "Rice": {"N": 80, "P": 40, "K": 40},
    "Maize": {"N": 100, "P": 45, "K": 20},
    "Jute": {"N": 50, "P": 40, "K": 40},
    "Cotton": {"N": 120, "P": 60, "K": 60},
    "Coconut": {"N": 20, "P": 15, "K": 30},
    "Papaya": {"N": 50, "P": 50, "K": 50},
    "Orange": {"N": 20, "P": 10, "K": 10},
    "Apple": {"N": 20, "P": 125, "K": 200},
    "Muskmelon": {"N": 100, "P": 15, "K": 50},
    "Watermelon": {"N": 100, "P": 15, "K": 50},
    "Grapes": {"N": 20, "P": 125, "K": 200},
    "Mango": {"N": 20, "P": 25, "K": 30},
    "Banana": {"N": 100, "P": 80, "K": 50},
    "Pomegranate": {"N": 20, "P": 15, "K": 40},
    "Lentil": {"N": 20, "P": 60, "K": 20},
    "Blackgram": {"N": 40, "P": 60, "K": 20},
    "Mungbean": {"N": 20, "P": 45, "K": 20},
    "Mothbeans": {"N": 20, "P": 45, "K": 20},
    "Pigeonpeas": {"N": 20, "P": 70, "K": 20},
    "Kidneybeans": {"N": 20, "P": 60, "K": 20},
    "Chickpea": {"N": 40, "P": 65, "K": 80},
    "Coffee": {"N": 100, "P": 30, "K": 30}
}

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

        # Preprocessing[cite: 2]
        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        scaled = ms.transform(features)
        final = sc.transform(scaled)
        
        # Feature #2: Top 3 Rankings using probability[cite: 3]
        probabilities = model.predict_proba(final)[0]
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_crops = [crop_dict.get(idx + 1) for idx in top_indices]

        # Get the primary recommendation
        crop_name = top_crops[0]
        
        # Feature #4: Fertilizer Advice Logic[cite: 3]
        tips = []
        if crop_name in crop_requirements:
            req = crop_requirements[crop_name]
            if N < req['N']:
                tips.append(f"Nitrogen is low ({N}). Consider adding Urea or Ammonium Sulfate to reach {req['N']}.")
            if P < req['P']:
                tips.append(f"Phosphorus is below optimal ({P}). Bone meal or Superphosphate could help reach {req['P']}.")
            if K < req['K']:
                tips.append(f"Potassium levels are low ({K}). Try adding Muriate of Potash to reach {req['K']}.")
        
        if not tips:
            tips.append("Your soil nutrients are perfectly balanced for this crop!")

        result = f"{crop_name}"

    except Exception as e:
        print(f"Error: {e}")
        return render_template("index.html", result="Error", tips=["An error occurred. Please check your inputs."])

    # Passing both 'result', 'tips', and 'top_crops' to the frontend
    return render_template("index.html", result=result, tips=tips, top_crops=top_crops)

if __name__ == "__main__":
    app.run(debug=True)