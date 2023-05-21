from flask import Flask, render_template, request, jsonify
from ChatBot2 import chatBot

app = Flask(__name__, static_url_path='/static')

@app.route("/chatbotEnvironment")
def home():
    return render_template("web-app.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json['msg']
    print(message)
    bot_res = chatBot(message)
    return jsonify({'msg': bot_res})

if __name__ == '__main__':
    app.run(debug=True)
