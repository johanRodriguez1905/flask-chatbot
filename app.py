from flask import Flask, render_template, request, jsonify
#from chat import get_response
from chat import get_response


app = Flask(__name__)


@app.get("/")
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    #Check if text is valid
    reponse = get_response(text)
    message = {"answer": reponse}
    return jsonify(message)


if __name__ == '__main__':
    app.run(debug=True)

