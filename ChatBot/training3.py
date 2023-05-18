# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:44:08 2023

@author: CSB5MC
"""

import sys
# print(sys.path)
# if sys.path
sys.path.append('C:\szakdolgozat_TVIK4I\ChatBot')

import random
import spacy
import json
import numpy as np
import matplotlib.pyplot as plt

import layers as layer
import activations as activation

# np.random.seed(0)

#nlp = spacy.load("hu_core_news_lg")
nlp = spacy.load("en_core_web_sm")

with open(r"C:\\Users\\CSB5MC\\Desktop\\python\\szakdoga\\intents.json") as file:
    intents = json.load(file)

sentence = "Hello világ! Nagyon örülök, hogy találkozhattam veletek ezen a rohanó és szép napon."

testS1 = "Welcome to Great Learning, Now start learning"
testS2 = "Learning is a good practice"

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
        
# print("words type:", type(words))
#Convert tokens to list
list_of_string = [i.text for i in words]

#lower case
list_of_string = [x.lower() for x in list_of_string]

sorted_list = sorted(list_of_string)

sorted_list = set(list_of_string)

word_lem = [token.lemma_ for token in words]
# print("\nwlemma", word_lem)
# print("\n")

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
# print("allsorted:", allsorted)
punctsorted = sorted(set(punctsorted))
      
#One-Hot encoding
#Bag of words

#convert to json and save it to a file, encoding utf-8 for hungarian letters
#needs the [1] because token object is a tuple of(instance, token) so we need just the 1 index
# =============================================================================
# print("words:", words)
# print("labels:", labels)
# print("words type:", type(words[0]))
# print("labels type:", type(labels))
# =============================================================================

# words = nlp(words)
# print("words:", words)
# print("words type:", type(words))

# #Convert tokens to list
# list_of_string = [i.text for i in allsorted]
# =============================================================================
# words_list = [str(token) for token in words]
# labels_list = [str(token) for token in labels]
# 
# json.dump(words_list, open('words.json', 'w'))
# json.dump(labels_list, open('labels.json', 'w'))
# =============================================================================

training = []
#tags length
output = [0] * len(labels)

for doc in docs:
    bag = []
    
    wpattern = doc[0]
    #print("\nwpattern:", wpattern)
    wpattern = [x.lemma_ for x in wpattern]
    #print(doc[0], wpattern)

    for word in punctsorted:
        if word in wpattern:
            bag.append(1)
        else:
            bag.append(0)
            
    output_row = list(output)
    # print(output_row)
    output_row[labels.index(doc[1])] = 1
    # print(output_row)
    # training.append(bag)
    training.append([bag, output_row])
    # out.append(output_row)
    
train_X = []
train_Y = []
for sub_list1 in training:
    train_X.append(sub_list1[0])
    train_Y.append(sub_list1[1])
    
# =============================================================================
# print("train_X:\n", train_X)
# print("train_X:\n", len(train_X))
# print("train_Y:\n", train_Y)
# =============================================================================

#ReLU activation function
# =============================================================================
# def Activation_ReLU(inputs):
#     return max(0, inputs)
# =============================================================================
        
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

#Vectors for semantic analysis to the model be more accurate
# =============================================================================
# vectors = []
# for token in words: #majd a words heljett szerintem train_X kell majd és az output lesz a train_y
#     vectors.append(token.vector)
# =============================================================================
    
vectors = []

for sub_x in train_X:
    vectors.append(sub_x)
    

class NeuralNetwork:
    def __init__(self, input_neuron, hidden_neuron, output_neuron):
        # self.learning_rate = learning_rate
        self.layers = [
            layer.Dense(input_neuron, hidden_neuron),
            # layer.Dense(hidden_neuron, hidden_neuron),
            layer.Dense(hidden_neuron, output_neuron)
            ]
        self.activations = [
            activation.ReLU(),
            # activation.ReLU(),
            activation.Softmax()
            # activation.Test()
            ]
        
        
    def forward(self, X):
        # print("\nX_type:\n", type(X))
        # print("\nX:\n", X)
        self.layers[0].forward(X)
        # print("\nhidden1:\n", self.layers[0].output)
        self.activations[0].forward(self.layers[0].output)
        # print("\nrelu:\n", self.activations[0].output)
        self.hidden = self.activations[0].output
        self.layers[1].forward(self.activations[0].output)
        # print("\nhidden2:\n", self.layers[1].output)
        self.activations[1].forward(self.layers[1].output)
        # print("\nsoftmax:\n", self.activations[1].output)
        self.prediction = self.activations[1].output
        
        # self.layers[0].forward(X)
        # self.activations[0].forward(self.layers[0].output)
        # self.layers[1].forward(self.activations[0].output)
        # self.activations[1].forward(self.layers[1].output)
        # self.layers[2].forward(self.activations[1].output)
        # self.activations[2].forward(self.layers[2].output)
        # self.prediction = self.activations[2].output

        # plt.plot(test_list_x[0])
        

        # plt.plot(train_Y[0])
        plt.plot(train_Y[2])
        plt.plot(self.prediction[2])
        plt.show()
# =============================================================================
#         print("laset_pred:\n", self.prediction[-1])
# =============================================================================
        
        return self.prediction
    

    
    def backward(self, X, y, learning_rate):
    
        pred = self.prediction
        
        # batch = X.shape[0]
        
        
        output_layer = self.layers[1]
        hidden_layer = self.layers[0]
        
        # output_activation = self.activations[1]
        hidden_activation = self.activations[0]
        
        print("pred\n", pred)
        print("y\n", y)
        
        #Backpropagation
        delta_output = 2 * (pred - y)
        
        hidden_activation.derivative(hidden_layer.output)
        dhidden = np.dot(delta_output, output_layer.weights.T) * hidden_activation.doutput
        
        #Gradient Descent
        output_layer.grad_weights = np.dot(hidden_activation.output.T, delta_output) #/ batch
        output_layer.grad_biases = np.sum(delta_output, axis=0) #/ batch
        hidden_layer.grad_weights = np.dot(X.T, dhidden) #/ batch
        hidden_layer.grad_biases = np.sum(dhidden, axis=0) #/ batch
        
        # print("output_layer.weights\n", output_layer.weights)
        # print("output_layer.grad_weights\n", output_layer.grad_weights)
        #Update Weights and Biases
        output_layer.weights -= learning_rate * output_layer.grad_weights
        # print("output_layer.weights\n", output_layer.weights)
        output_layer.biases -= learning_rate * output_layer.grad_biases
        hidden_layer.weights -= learning_rate * hidden_layer.grad_weights
        hidden_layer.biases -= learning_rate * hidden_layer.grad_biases
        
        # self.output_W = output_layer.weights
        # self.output_B = output_layer.biases
        # self.hidden_W = hidden_layer.weights
        # self.hidden_B = hidden_layer.biases


        return output_layer.weights, output_layer.biases, hidden_layer.weights, hidden_layer.biases
            
            
        # output_layer = self.layers[1]
        # hidden_layer = self.layers[0]
        
        # hidden_pred = self.activations[0]
        
        # #pred = predicted value
        # derror_out = 2 * (pred - y)
        # output_layer.grad_weights = pred * (1 - pred)
        # output_layer.grad_weights = 
        # print("1\n", output_layer.grad_weights)
        # print("2\n", derror_out)
        # print("3\n", output_layer.weights)
        # output_layer.weights -= learning_rate * output_layer.grad_weights * derror_out
            
        # hidden_layer.grad_weights = hidden_pred.output * (1 - hidden_pred.output)
        # hidden_layer.weights -= learning_rate * hidden_layer.grad_weights * derror_out
        
        # diff_output = y - self.prediction
        
        
        
        # for i in reversed(range(len(self.layers))):
        #     # f_layer =  self.layers[i]
        #     # f_activation = self.activations[i]
        #     # # #Itt lehet hogy nem a relu kell
        #     # prev_layer = self.activations[i-1]


        
            # if i == len(self.layers) - 1:
                #self. <- kell ez?
                # print("1:\n", f_activation.output)
                
            # self.diff_output = y - pred
            
            # print("y:\n", y)
            # print("pred:\n", pred)
            
            
            

                
                #--------------------------------------------------------
                
# =============================================================================
#                 f_layer.grad_weights = 2 * (f_activation.output - 1)
#                 f_layer.grad_weights = np.dot(prev_layer.output.T, f_layer.grad_weights)
#                 print("2:\n", f_layer.grad_weights)
#                 f_layer.grad_biases = np.sum(self.diff_output, axis=0, keepdims=True)
#                 print("weights:\n", f_layer.weights)
#                 f_layer.weights -= learning_rate * f_layer.grad_weights
#                 print("3:\n", f_layer.weights)
#                 f_layer.biases -= learning_rate * f_layer.grad_biases
#                 
#             elif i == 0:
#                 prev_layer = X
# =============================================================================
                
                
                


# =============================================================================
#                 f_layer.grad_weights = 2 * (f_layer.output - 1)
#                 f_layer.grad_weights = np.dot(prev_layer.T, f_layer.grad_weights)
#                 f_layer.grad_biases = np.dot(self.diff_output.T, f_activation.output)
#                 f_layer.grad_biases = np.sum(f_layer.grad_biases, axis=0, keepdims=True)
#                 f_layer.weights -= learning_rate * f_layer.grad_weights
#                 f_layer.biases -= learning_rate * f_layer.grad_biases
# =============================================================================

                
                
            # else:
            
    # def train(self, training, learning_rate, epochs, batch_size):
    def train(self, X, y, learning_rate, epochs, batch_size):
        # for epoch in epochs:
            
        self.learning_rate = learning_rate
        print("LR", learning_rate)
        loss = 0.0
        
        batch_x = X
        batch_y = y
# =============================================================================
#         
#             print("INFORWARD")
#             # print("batch_x", batch_x)
#             
#             #forward propagation
#             f_output = self.forward(batch_x)
#             
#             print("INBACKWARD")
#         
#             #backward propagation
#             ow, ob, hw, hb = self.backward(batch_x, batch_y, learning_rate)
#         
#     # =============================================================================
#     #             print("epochs", epochs)
#     #             print("epoch:", epoch)
#     #             
#     #             if epoch == epochs-1:
#     #                 model_Ws_Bs = {
#     #                     'hidden_W': hw.tolist(),
#     #                     'hidden_B': hb.tolist(),
#     #                     'output_W': ow.tolist(),
#     #                     'output_B': ob.tolist()
#     #                 }
#     #                 
#     #                 # json_path = "/path/save/model.json"
#     #                 with open('model.json', 'w') as file:
#     #                     json.dump(model_Ws_Bs, file)
#     #                     
#     #                 print("model saved!")
#     # =============================================================================
#         
#         
#     
#     #calculate the loss/cost/error Mean Squared Error (MSE)
#     # =============================================================================
#     #             print("VALUE:\n",  batch_y)
#     #             print("VALUE:\n",  f_output)
#     # =============================================================================
#         # print("VALUE:\n", batch_y - f_output)
#         
#             loss = np.mean(1/len(batch_y) * sum(np.square(batch_y - f_output)))
#             # loss = np.mean((f_output - batch_y)**2)
#             # loss = np.mean(1/len(train_y) * sum((batch_y - f_output)**2))
#             # cost = -np.mean(train_Y * np.log(f_output) + (1 - train_Y) * np.log(1 - f_output))
#         
#         
#         # =============================================================================
#         #             print("last_y:", y[i])
#         # =============================================================================
#             # print("out:", f_output)
#         
#         # =============================================================================
#         #             loss = np.mean(1/len(train_Y) * sum((y[i] - f_output)**2))
#         # =============================================================================
#             
#         
#         #print the loss for the epoch
#             print(f"Epoch {epoch+1} / {epochs}, loss = {loss / batch_x.shape[0]}")
# =============================================================================
    
        # X_arr = np.array(X)
        # y_arr = np.array(y)
        
        for epoch in range(epochs):
            loss = 0.0
            
            # shuffled_data = shuffle_data(training)
            
            # batch_x = shuffled_data[0]
            # batch_y = shuffled_data[1]
            
            # print("batch_x", batch_x)
            # print("batch_y", batch_y)
            
            
            # batch_x = np.array(batch_x)
            # batch_y = np.array(batch_y)
                        
            #random batch group
            batch_x = batch_x[:batch_size, :]
            batch_y = batch_y[:batch_size, :]
            
            # print("xshape:\n", X_arr.shape[0])
            #azért kell shape és nem len, mert a len, csak az első dimenzió méretét adja meg, nem az összesét
        # for i in range('0, X_arr.shape[0], training):

            
            print("INFORWARD")
            # print("batch_x", batch_x)
            
            #forward propagation
            f_output = self.forward(batch_x)
            
            print("INBACKWARD")

            #backward propagation
            ow, ob, hw, hb = self.backward(batch_x, batch_y, learning_rate)
            
# =============================================================================
#             print("epochs", epochs)
#             print("epoch:", epoch)
#             
#             if epoch == epochs-1:
#                 model_Ws_Bs = {
#                     'hidden_W': hw.tolist(),
#                     'hidden_B': hb.tolist(),
#                     'output_W': ow.tolist(),
#                     'output_B': ob.tolist()
#                 }
#                 
#                 # json_path = "/path/save/model.json"
#                 with open('model.json', 'w') as file:
#                     json.dump(model_Ws_Bs, file)
#                     
#                 print("model saved!")
# =============================================================================
            
            

        #calculate the loss/cost/error Mean Squared Error (MSE)
# =============================================================================
#             print("VALUE:\n",  batch_y)
#             print("VALUE:\n",  f_output)
# =============================================================================
            # print("VALUE:\n", batch_y - f_output)
            
            loss = np.mean(1/len(batch_y) * sum(np.square(batch_y - f_output)))
            # loss = np.mean((f_output - batch_y)**2)
            # loss = np.mean(1/len(train_y) * sum((batch_y - f_output)**2))
            # cost = -np.mean(train_Y * np.log(f_output) + (1 - train_Y) * np.log(1 - f_output))


# =============================================================================
#             print("last_y:", y[i])
# =============================================================================
            # print("out:", f_output)

# =============================================================================
#             loss = np.mean(1/len(train_Y) * sum((y[i] - f_output)**2))
# =============================================================================
            
        
        #print the loss for the epoch
            print(f"Epoch {epoch+1} / {epochs}, loss = {loss / batch_x.shape[0]}")

        # return model_Ws_Bs
        
test_list_x = np.array([[1, 0, 1], [0, 1, 1], [0, 1, 0]])
# [[1, 0, 1], [0, 1, 1]]
test_list_y = np.array([[1, 0], [1, 0], [0, 1]])

# =============================================================================
# testx = [[1, 0], [0, 1], [1, 1], [0, 0]]
# testy = [[1], [1], [0], [0]]
# =============================================================================

# =============================================================================
# testx = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], 
#          [0,0,0], [1,1,1]]
# testy = [[0,1], [0,1], [0,1], [1,0], [1,0], [1,0],
#          [0,0], [1,1]]
# =============================================================================

# test_activation = activation.Softmax()

# x = np.array([[1, 0, 0],
#               [0, 1, 0],
#               [1, 0, 0],
#               [0, 0, 1]])

# test_activation.forward(x)
# print("x_pred:\n", test_activation.output)


# =============================================================================
# def shuffle_data(training):
#     random.shuffle(training)
#     training = np.array(training, dtype=object)
#     
#     train_x = list(training[:, 0])
#     train_y = list(training[:, 1])
#     
#     return train_x, train_y
# =============================================================================
    
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

print("tX\n", train_x)
print("tY\n", train_y)

# print("tx", shuffle_data(training).train_x)
# print("ty", shuffle_data(training).train_y)

learning_rate = 0.01 #0.1
epochs = 100
batch_size = 5
nn = NeuralNetwork(32, 2, 5)

model = nn.train(np.array(train_x), np.array(train_y), learning_rate, epochs, batch_size)
    


# model = nn.train(training, learning_rate, epochs, batch_size) #len(test_list_x)
# =============================================================================
# model = nn.train(np.array(train_x), np.array(train_y), learning_rate, epochs, batch_size)
# =============================================================================

# print("model:\n", 



# test1 = nn.forward([1,0,0]) #[0,1]
# test2 = nn.forward([0,1,0]) #[0,1]
# test3 = nn.forward([0,0,0]) #[0,0]
# test4 = nn.forward([1,1,1]) #[1,1]

# print("test1:", test1) #[1]
# print("test2:", test2) #[1]
# print("test3:", test3) #[0]
# print("test4:", test4) #[0]


print("Done")
