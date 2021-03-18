import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    # constructs the neural network
    def __init__(self, nn_inputs, nn_outputs, nn_epochs):
        self.inputs = nn_inputs
        self.outputs = nn_outputs
        self.epochs = nn_epochs
        self.hidden = 0
        self.error = 0

        # seeds e random number generator
        np.random.seed(1)
        # gets synaptic weights from -1 ro 1
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        self.error_history = []
        self.epoch_list = []

    # Using the sigmoid function
    def sigmoid(self, x, derivative=False):
        if not derivative:
            return 1 / (1 + np.exp(-x))
        else:
            # returns derivative of sigmoid function
            return x * (1 - x)

    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.synaptic_weights))

    # going backwards through the network to update weights
    def backpropagation(self):
        self.error = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, derivative=True)
        self.synaptic_weights += np.dot(self.inputs.T, delta)

    # trains model to make accurate predictions while continually adjusting weights
    def train(self):
        for epoch in range(self.epochs):
            # go forward and produce an output
            self.feed_forward()
            # go back through the network and make corrections based on the output
            self.backpropagation()

            # keep track of input data
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.synaptic_weights))
        return prediction


def run(run_inputs, run_outputs, run_iterations, run_new_inputs):
    # initializes neural network class
    neural_network = NeuralNetwork(run_inputs, run_outputs, run_iterations)

    # trains network
    neural_network.train()

    # print the predictions for new inputs
    for i in range(len(run_new_inputs)):
        print(run_new_inputs[i])
        print(neural_network.predict(run_new_inputs[i]), ' - Correct: ', run_new_inputs[i][0])

    # plot the error over the entire training duration
    plt.figure(figsize=(15, 5))
    plt.plot(neural_network.epoch_list, neural_network.error_history)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()


# provides all possible datasets
def data():
    data_input = []

    for i1 in range(0, 2):
        for i2 in range(0, 2):
            for i3 in range(0, 2):
                data_input.append([i1, i2, i3])
    return data_input


# gets the outputs for a set of inputs
def get_outputs(get_outputs_inputs):
    get_outputs_outputs = []
    for i in range(0, len(get_outputs_inputs)):
        get_outputs_outputs.append([get_outputs_inputs[i][0]])
    return get_outputs_outputs


inputs = np.array(data())

iterations = 15000
new_inputs = np.array([[0, 0, 1],
                       [1, 0, 1],
                       [0, 0, 0],
                       [1, 1, 1],
                       [1, 1, 0],
                       [0, 1, 1],
                       [1, 0, 1],
                       [1, 0, 1]])

run(inputs, np.array(get_outputs(inputs)), iterations, new_inputs)
