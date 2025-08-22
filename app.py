from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model (after training pipeline)
model = pickle.load(open("artifacts/model.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

# Helper function to convert Product Type to numerical values
def convert_product_type(product_type_str):
    if product_type_str.lower() == 'l':
        return 0
    elif product_type_str.lower() == 'm':
        return 1
    elif product_type_str.lower() == 'h':
        return 2
    else:
        return -1 # Or handle other cases as needed

@app.route("/predict", methods=["POST"])
def predict():
    try:
        air_temp = float(request.form["air_temp"])
        process_temp = float(request.form["process_temp"])
        rot_speed = float(request.form["rot_speed"])
        torque = float(request.form["torque"])
        tool_wear = float(request.form["tool_wear"])
        twf = float(request.form["twf"])
        hdf = float(request.form["hdf"])
        
        # Convert product_type string to a numerical value
        product_type_str = request.form["product_type"]
        product_type_val = convert_product_type(product_type_str)

        input_data = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear, twf, hdf, product_type_val]])
        prediction = model.predict(input_data)[0]

        return render_template("home.html", prediction=prediction)
    except Exception as e:
        return render_template("home.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)