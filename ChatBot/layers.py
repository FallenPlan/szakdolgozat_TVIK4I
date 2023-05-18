import spacy
import numpy as np

class Dense:
    def __init__(self, n_input, n_neuron):
        print("\ninput", n_input)
        print("neuron", n_neuron)
        self.weights = np.random.randn(n_input, n_neuron) #* 0.1
        # print("weights:\n", self.weights)
        self.biases = np.zeros((1, n_neuron)) #, dtype=int
        
        self.grad_weights = []
        self.grad_biases = []
        
        # self.weight_hod = np.array([])
        # self.bias_o =  np.array([])
        # self.weight_ihd =  np.array([])
        # self.bias_h =  np.array([])
# =============================================================================
#         print("biases:\n", self.biases)
# =============================================================================
        
    def forward(self, _input):
# =============================================================================
#         print("\n_inputTTT:\n", _input)
#         print("self.weigthsSSS:\n", self.weights)
# =============================================================================
        self.output = np.dot(_input, self.weights) + self.biases
# =============================================================================
#     
# #ReLU activation function
# class ReLU:
#     def forward(self, _input):
#         self.output = np.maximum(0, _input)
# 
# class Softmax:
#     def forward(self, inputs):
#         exp_values = np.exp(inputs)
#         # print("\nexp:\n", exp_values)
#         probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
#         self.output = probs
# 
# class Loss:
#     def calculate(self, output, y):
#         losses = self.forward(output, y)
#         data_loss = np.mean(losses)
#         return data_loss
#     
# class CategoricalCrossentropy(Loss):
#     def forward(self, y_pred, y_true):
#         sample = len(y_pred)
#         clipped = np.clip(y_pred, 1e-7, 1-1e-7)
#     
#         if len(y_true.shape) == 1:
#             confidence = clipped[range(sample), y_true]
#         elif len(y_true.shape) == 2:
#             confidence = np.sum(clipped*y_true, axis=1)
#             
#         neg_log = -np.log(confidence)
#         return neg_log
# =============================================================================
