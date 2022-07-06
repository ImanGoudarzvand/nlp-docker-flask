
from flask import Flask, request, jsonify
from utils import predict_pipeline
import pickle


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # response = request.args.get("text") -----------> return none
    response = request.get_json()

    try:
        text = response['text']
    except KeyError:
        return {"Error": "No text was sent."}

    # request.json = request.get_json()  -----> have the same result
    # request.args.get("text") --------> returns None
    # return {"text": response}
    text = [text]
    result = predict_pipeline(text)
    # print(result)
    # When returning a dictionary keys must be int not numpy object so fix it in return of predict func
    return result[0]


if __name__ == "__main__":
    app.run(debug=True)
