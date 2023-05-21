# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:44:42 2023

@author: CSB5MC
"""

import json
import numpy as np
import spacy
import random

import activations as activation

nlp = spacy.load("en_core_web_sm")

intents = json.loads(open(r"C:\\szakdolgozat_TVIK4I\\ChatBot\\chatbotEnvironment\\intents.json").read())

#load saved datas
words = json.load(open('words.json', 'r'))
labels = json.load(open('labels.json', 'r'))

#load saved model (weights and biases)
with open('model.json', 'r') as file:
    model = json.load(file)
    
hw = np.array(model['hidden_W'])
hb = np.array(model['hidden_B'])
ow = np.array(model['output_W'])
ob = np.array(model['output_B'])

print("hw\n", ob)


def clean_text(text):
    tokens = nlp(text)
    tokens = [token.lemma_ for token in tokens]
    print("1")
    return tokens

def bag_of_words(text):
    
# =============================================================================
#     tokens = clear_text(text)
#     bow = [1 if word == w else 0 for w in tokens for word in words]
# =============================================================================
    
    tokens = clean_text(text)
    bow = [0] * len(words)
    for w in tokens:
        for i, word in enumerate(words):
            if word == w:
                bow[i] = 1
                
    # print("bow:", bow)
    print("2")
    return np.array(bow)

print("words:\n", words)

def predict(text):
    
    activations = [
        activation.ReLU(),
        activation.Softmax()
        ]
    
    print("text:\n", text)
    hidden_layer = np.dot(text, hw) + hb
    activations[0].forward(hidden_layer)
    hidden_activation = activations[0].output
    output_layer = np.dot(hidden_activation, ow) + ob
    activations[1].forward(output_layer)
    output_activation = activations[1].output
    
    # print("output_activation\n", output_activation)
    # predicted_labels = np.argmax(output_activation, axis=1)
    # print("predicted_labels\n", predicted_labels)
    # return predicted_labels
    print("3")
    return output_activation

def pred_label(text):
    bow = bag_of_words(text)
    # print("model:", model)
    # print("np.array([bow])", np.array([bow]))
    result = predict(np.array([bow]))[0]
    print("result:", result)
    threshold = 0.2
    results = [[i, r] for i, r in enumerate(result) if r > threshold]
    
    results.sort(key=lambda x: x[1], reverse=True)
    result_list = []
    for r in results:
        result_list.append({'intent': labels[r[0]], 'probability': str(r[1])})
        
    print("4")
    return result_list
    
# =============================================================================
#     results = [
#         {'intent': labels[i], 'probability': str(r)}
#         for i, r in enumerate(result)
#         if r > treshold
#         ]
#     return sorted(results, key=lambda x: x['probability'], reverse=True)
# =============================================================================

def get_response(intents_list, intents_json):
    print("5")
    print("LEN:", intents_list)
    probability = float(intents_list[0]['probability'])
    print("prob:", probability)
    if probability < 0.5:
        result = "I don't understand."
        return result
    else:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    
def chatBot(message):
    print("ChatBot")
    print("Ask me something! :)")
    #while True:
    #message = input("You: ")
    ints = pred_label(message)
    response = get_response(ints, intents)
    print("ChatBot: ", response)
    if response in ["Bye!", 
                    # "bye", 
                    "See you soon!", 
                    "I will be here if you need my further assistance :D"]:
        exit()
        #break
                
#chatBot()
    