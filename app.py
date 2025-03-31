from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    return jsonify({"message": "AI Model Prediction Here"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
