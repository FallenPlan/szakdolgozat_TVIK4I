import numpy as np

#ReLU activation function
class ReLU:
    def forward(self, _input):
        self.output = np.maximum(0, _input)
        
    def derivative(self, output):
        self.doutput = np.where(output > 0, 1, 0)
        
def drelu(relu):
    return np.where(relu > 0, 1, 0)
        
class Test:
    def forward(self, x):
        self.output = x

class Softmax:
    def forward(self, inputs):
        # exp_values = np.exp(inputs)
        #In case of exp_values has inf. in it
        exp_values = np.exp(inputs - inputs.max(axis=1, keepdims=True))
        # print("\nexp:\n", exp_values)
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probs
        
    def derivative(self, softmax):
        # print("output:\n", softmax)
        # print("soft:\n", soft)
        self.doutput = softmax * (1 - softmax)
        # print("self.doutput:\n", self.doutput)
        # print("self.doutput_type:\n", type(self.doutput))
        
        
        
        
        
# =============================================================================
#         print("asd0:\n", output)
#         print("asd:\n", np.diag(output))
#         jacobin_matrix = output
#         print("asd2", len(jacobin_matrix))
#         length = len(jacobin_matrix)
#         print("asd3", range(length))
#         for i in range(length):
#             print("i: ", i)
#             for j in range(length):
#                 print("j: ", j)
#                 if i == j:
#                     jacobin_matrix[i][j] = output[i] * (1-output[i])
#                 else:
#                     jacobin_matrix[i][j] = -output[i] * output[j]
#                     
#         self.doutput = jacobin_matrix
# =============================================================================

# =============================================================================
# def softmax(inputs):
#         inputs -= np.max(inputs)
#         probs = (np.exp(inputs).T / np.sum(np.exp(inputs), axis=0)).T
#         return probs
# =============================================================================

def dsoftmax(softmax):
    jacobin_matrix = np.diag(softmax)
    length = len(jacobin_matrix)
    
    for i in range(length):
        for j in range(length):
            if i == j:
                jacobin_matrix[i][j] = softmax[i] * (1-softmax[i])
            else:
                jacobin_matrix[i][j] = -softmax[i] * softmax[j]
                
    return jacobin_matrix

# =============================================================================
# instance = Softmax()
# 
# x = np.array([1, 2])
# instance.forward(x)
# print("der1:\n", instance.output)
# der = dsoftmax(instance.output)
# print("der2:\n", der)
# =============================================================================
        
# =============================================================================
#         print("softmax_B:\n", self.output)
#         # 2 dimensional 1 row array [[1, 2, 3]]
#         #np.reshape(self.output, -1) => 1D & 1 row array [1, 2, 3]
#         # softmax = self.output.reshape(1, -1) #=> same than under
#         self.softmax = np.reshape(self.output, (1, -1))
#         print("softmax_A:\n", self.softmax)
#         self.grad = np.reshape(inputs, (1, -1))
#         print("inputs:\n", inputs)
#         print("grad:\n", self.grad)
# =============================================================================
        
    # creating 1 row matrixes/vectors to match the dimensions
    # def dActivation(self, inputs):
        
import sympy as sp        

# x = 3
# y = 1

# test = (x - y)**2
# print("test", test)

# test2 = 2 * (x - y)
# print("test2", test2)

# f = (x - y)**2
# test3 = sp.diff(f, x)
# print("test3", test3)



# def dActivation(x):
#     return x * (1 - x)

class Loss:
    def calculate(self, output, y):
        losses = self.forward(output, y)
        data_loss = np.mean(losses)
        return data_loss
    
class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        clipped = np.clip(y_pred, 1e-7, 1-1e-7)
    
        if len(y_true.shape) == 1:
            confidence = clipped[range(sample), y_true]
        elif len(y_true.shape) == 2:
            confidence = np.sum(clipped*y_true, axis=1)
            
        neg_log = -np.log(confidence)
        return neg_log