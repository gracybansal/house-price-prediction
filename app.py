from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        age = float(request.form['age'])
        location = request.form['location']

        # Encode location manually
        loc_delhi = 1 if location == "Delhi" else 0
        loc_noida = 1 if location == "Noida" else 0
        loc_gurgaon = 1 if location == "Gurgaon" else 0

        features = [[area, bedrooms, bathrooms, age,
                     loc_delhi, loc_noida, loc_gurgaon]]

        prediction = model.predict(features)

        return render_template('index.html',
                               prediction_text=f"Estimated Price: ₹{int(prediction[0])}")

    except:
        return render_template('index.html',
                               prediction_text="Error: Please enter valid values")

if __name__ == "__main__":
    app.run(debug=True)