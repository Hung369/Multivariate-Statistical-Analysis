import torch.nn as nn
import torch

def sigmoid(value): # hàm sigmoid activation
    sig = 1 / (1 + torch.exp(-value))
    return sig

def sigmoid_derivative(value): # đạo hàm của hàm sigmoid activation
    d = value * (1 - value)
    return d

class FFNN(nn.Module):
    # initialization function
    def __init__(self):
        # init function of base class
        super(FFNN, self).__init__()

        # corresponding size of each layer
        self.inputlayer = 3 # number of perceptrons at input layer
        self.hiddenlayer = 12 # number of perceptrons at hidden layer
        self.outputlayer = 1 # number of perceptrons at output layer

        # random weights from a normal distribution
        self.W1 = torch.randn(self.inputlayer, self.hiddenlayer)   # 3 X 12 tensor
        self.W2 = torch.randn(self.hiddenlayer, self.hiddenlayer)  # 12 X 12 tensor
        self.W3 = torch.randn(self.hiddenlayer, self.outputlayer)  # 12 X 1 tensor

        self.z = None # processing result between input layer and 1st hidden layer
        self.a = None # sigmoid activation result of z

        self.z_activation = None # save sigmoid activation result of z
        self.z_activation_derivative = None # save the gradient descent sigmoid activation result of z


        self.z2 = None # processing result between 1s hidden layer and 2nd hidden layer
        self.a2 = None # sigmoid activation result of z2

        self.z3 = None # processing result between 1s hidden layer and 2nd hidden layer
        self.a3 = None # sigmoid activation result of z3

        self.w3_error = None # Error value between correct result and predicted result
        self.w3_delta = None # result of error * derivative of activation function of ai result (i = i-th layer) (i=3)

        self.w2_error = None # Error value between correct result and predicted result
        self.w2_delta = None # result of error * derivative of activation function of ai result (i = 2)

        self.w1_error = None # Error value between correct result and predicted result
        self.w1_delta = None # result of error * derivative of activation function of ai result (i = 1)

        # a = sigmoid(z); z = x.T * W       

    # activation function using sigmoid
    def activation(self, z):
        self.z_activation = sigmoid(z)
        return self.z_activation

    # derivative of activation function
    def activation_derivative(self, z):
        self.z_activation_derivative = sigmoid_derivative(z)
        return self.z_activation_derivative

    def forward(self, X):
        # multiply input X and weights W1 from input layer to 1st hidden layer
        self.z = torch.matmul(X, self.W1)
        self.a = self.activation(self.z)  # activation function

        # multiply current tensor and weights W2 from 1st hidden layer to 2nd hidden layer
        self.z2 = torch.matmul(self.a, self.W2)
        self.a2 = self.activation(self.z2)  # activation function

        # multiply current tensor and weights W3 from 2nd hidden layer to output layer
        self.z3 = torch.matmul(self.a2, self.W3)
        self.a3 = self.activation(self.z3)  # activation function
        return self.a3

    def backward(self, X, y, res, rate):
        # W3 adjustment rate
        self.w3_error = y - res  # error in output
        self.w3_delta = self.w3_error * self.activation_derivative(res)  # derivative of activation to error

        # W2 adjustment rate
        self.w2_error = torch.matmul(self.w3_delta, torch.t(self.W3))
        self.w2_delta = self.w2_error * self.activation_derivative(self.a2)

        # W1 adjustment rate
        self.w1_error = torch.matmul(self.w2_delta, torch.t(self.W2))
        self.w1_delta = self.w1_error * self.activation_derivative(self.a)

        # update weights from delta of error and learning rate
        self.W1 += torch.matmul(torch.t(X), self.w1_delta) * rate
        self.W2 += torch.matmul(torch.t(self.a2), self.w2_delta) * rate
        self.W3 += torch.matmul(torch.t(self.a3), self.w3_delta) * rate

    # training function with learning rate parameter
    def train(self, X, y, rate, e):
        # forward + backward pass for training
        result = self.forward(X)
        print("Epoch #" + str(e) + " MSE Loss: " + str(torch.mean(torch.pow(y - result, 2))))
        self.backward(X, y, result, rate)

    # predict function
    def predict(self, x_predict):
        print("Predict: " + str(self.forward(x_predict)))
    
    def fit(self, x, y, epochs, lr):
        for i in range(epochs):           
            self.train(x, y, lr, i) # lr = learning rate