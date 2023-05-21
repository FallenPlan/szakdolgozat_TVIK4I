from flask import Flask, render_template, request, jsonify
import ChatBot2 as chatBot
from ChatBot2 import pred_label, get_response
import json

with open(r"C:\szakdolgozat_TVIK4I\ChatBot\chatbotEnvironment\intents.json") as file:
    intents = json.load(file)

app = Flask(__name__, static_folder='static')

@app.route("/")
def home():
    return render_template("web-app.html")

@app.route("/chatBot", methods=["POST"])
def predict():
    try:
        text = request.get_json().get('message')
        intents_list = pred_label(text)
        response = get_response(intents_list, intents)
        return jsonify({'message': response})
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return jsonify({'message': error_message}), 500

print("there")
if __name__ == '__main__':
    app.run(debug=True)
