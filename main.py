import pickle
from flask import Flask, request, jsonify
from model.ml_model import predict_mpg

app = Flask(__name__)

@app.route('/',methods=['POST'])
def predict():
    vehical_config = request.get_json() # Take input from the end user

    with open('model/random_forest_model_v1.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()

    predictions = predict_mpg(vehical_config, model)  # make prediction using the model

    response = {                                 # Returning the response to the end user
        'mpg_prediction':list(predictions)
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)


