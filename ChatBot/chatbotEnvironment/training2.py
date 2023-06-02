# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:44:08 2023

@author: CSB5MC
"""

import sys
sys.path.append('C:\szakdolgozat_TVIK4I\ChatBot\chatbotEnvironment')

import random
import spacy
import json
import numpy as np

import layers as layer
import activations as activation

# np.random.seed(0)
nlp = spacy.load("en_core_web_sm")

with open(r"C:\szakdolgozat_TVIK4I\ChatBot\chatbotEnvironment\intentsNEW.json") as file:
    intents = json.load(file)

#nlp does a split in sentences with spaces and separates punctuations
doc = nlp(sentence)

#stopwords list (326 items)
stopwords = nlp.Defaults.stop_words

words = []
labels = []
docs = []
ignore = ["?", "!", ",", ".", ":"]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
    #pattern = separet sentences in pattern
        list_of_words = nlp(pattern)
    #list_of_words = tokens (words)
    #words = words list + (pattern(sentence) --> words(list))
    #extend = add a list to a list
        words.extend(list_of_words)
    #docs = docs(list of sentences) + the next sentence
    #append = add an item to a list
    #docs will be a tuple of words and it's labels!
        docs.append((list_of_words, intent["tag"]))
    if intent["tag"] not in labels:
        labels.append(intent["tag"])
        
#Convert tokens to list
list_of_string = [i.text for i in words]

#lower case
list_of_string = [x.lower() for x in list_of_string]

sorted_list = sorted(list_of_string)

sorted_list = set(list_of_string)

word_lem = [token.lemma_ for token in words]

word_lem_nlp = []

for i in word_lem:
    s_string = nlp(i)
#if it's extend -> output list == token.Token, if append -> output list == doc.Doc
    word_lem_nlp.extend(s_string)
    
allsorted = [token.text for token in word_lem_nlp if token.is_stop != True and token.is_punct != True]
punctsorted = [token.text for token in word_lem_nlp if token.is_punct != True]

#lower case
allsorted = [x.lower() for x in allsorted]
punctsorted = [x.lower() for x in punctsorted]

allsorted = sorted(set(allsorted))
punctsorted = sorted(set(punctsorted))


words_list = punctsorted
labels_list = labels

json.dump(words_list, open('words.json', 'w'))
json.dump(labels_list, open('labels.json', 'w'))

training = []
#tags length
output = [0] * len(labels)

for doc in docs:
    bag = []
    
    wpattern = doc[0]
    wpattern = [x.lemma_ for x in wpattern]

    for word in punctsorted:
        if word in wpattern:
            bag.append(1)
        else:
            bag.append(0)
            
    output_row = list(output)
    output_row[labels.index(doc[1])] = 1
    training.append([bag, output_row])
    
train_X = []
train_Y = []
for sub_list1 in training:
    train_X.append(sub_list1[0])
    train_Y.append(sub_list1[1])
      
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

vectors = []

for sub_x in train_X:
    vectors.append(sub_x)
    

class NeuralNetwork:
    def __init__(self, input_neuron, hidden_neuron, output_neuron):
        self.layers = [
            layer.Dense(input_neuron, hidden_neuron),
            layer.Dense(hidden_neuron, output_neuron)
            ]
        self.activations = [
            activation.ReLU(),
            activation.Softmax()
            ]
        
        
    def forward(self, X):
        self.layers[0].forward(X)
        self.activations[0].forward(self.layers[0].output)
        self.hidden = self.activations[0].output
        self.layers[1].forward(self.activations[0].output)
        self.activations[1].forward(self.layers[1].output)
        self.prediction = self.activations[1].output
        return self.prediction
    

    
    def backward(self, X, y, learning_rate):
    
        pred = self.prediction
        
        output_layer = self.layers[1]
        hidden_layer = self.layers[0]
        
        hidden_activation = self.activations[0]
        
        #Backpropagation
        delta_output = 2 * (pred - y)
        
        hidden_activation.derivative(hidden_layer.output)
        dhidden = np.dot(delta_output, output_layer.weights.T) * hidden_activation.doutput
        
        #Gradient Descent
        output_layer.grad_weights = np.dot(hidden_activation.output.T, delta_output)
        output_layer.grad_biases = np.sum(delta_output, axis=0)
        hidden_layer.grad_weights = np.dot(X.T, dhidden)
        hidden_layer.grad_biases = np.sum(dhidden, axis=0)
        
        #Update Weights and Biases
        output_layer.weights -= learning_rate * output_layer.grad_weights
        output_layer.biases -= learning_rate * output_layer.grad_biases
        hidden_layer.weights -= learning_rate * hidden_layer.grad_weights
        hidden_layer.biases -= learning_rate * hidden_layer.grad_biases

        return output_layer.weights, output_layer.biases, hidden_layer.weights, hidden_layer.biases
            
    def train(self, X, y, learning_rate, epochs, batch_size, train_ratio):
            
        self.learning_rate = learning_rate
        
        batch_x = X[:batch_size, :]
        batch_y = y[:batch_size, :]

        for epoch in range(epochs):
            loss = 0.0

            #forward propagation
            f_output = self.forward(batch_x)

            #backward propagation
            ow, ob, hw, hb = self.backward(batch_x, batch_y, learning_rate)
            
            if epoch+1 == epochs:
                model_Ws_Bs = {
                    'hidden_W': hw.tolist(),
                    'hidden_B': hb.tolist(),
                    'output_W': ow.tolist(),
                    'output_B': ob.tolist()
                }
                
                # json_path = "/path/save/model.json"
                with open('model.json', 'w') as file:
                    json.dump(model_Ws_Bs, file)
                    
                print("model saved!")
            
            

        #calculate the loss/cost/error Mean Squared Error (MSE)
            loss = np.mean(1/len(batch_y) * sum(np.square(batch_y - f_output)))

        #print the loss for the epoch
            print(f"Epoch {epoch+1} / {epochs}, loss = {loss / batch_x.shape[0]}")
        
    def test(self, X, y, test_size, train_ratio):
            
        self.learning_rate = learning_rate
        
        total_rows = X.shape[0]
        train_size = int(total_rows * train_ratio)
        
        #random test group
        test_x = X[train_size:, :]
        test_y = y[train_size:, :]

        #forward propagation
        f_pred = self.forward(test_x)

        pred_matrix = np.zeros_like(test_y)
        
        for i in range(f_pred.shape[0]):
            pred_answer = np.argmax(f_pred[i])
            
            print("pred_answer ; y:", pred_answer+1, test_y[i])
            
            pred_matrix[i][pred_answer] = 1
            
        
        correct_sum = 0
        
        for i in range(f_pred.shape[0]):
            if np.array_equal(pred_matrix[i], test_y[i]):
                correct_sum += 1
        
        print("correct_sum: ", correct_sum)
        
        total_preds = round(f_pred.sum())
        
        print("total_preds: ", total_preds)
        
        accuracy = correct_sum / total_preds
        
        print("Accuracy: ", accuracy)

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

print("train_x\n", len(train_x))
print("train_y\n", len(train_y))

learning_rate = 0.005
epochs = 200
train_ratio = 0.8
batch_size = 30
test_size = round(batch_size*(1-train_ratio))
print("TEST_SIZE", test_size)
nn = NeuralNetwork(40, 64, 8)

model = nn.train(np.array(train_x), np.array(train_y), learning_rate, epochs, batch_size, train_ratio)
    
test = nn.test(np.array(train_x), np.array(train_y), test_size, train_ratio)

print("Done")
