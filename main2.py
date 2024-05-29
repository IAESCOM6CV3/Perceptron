class Perceptron:
    def __init__(self, learningRate = 0.01, neurons = 10) -> None:
        self.learningRate = learningRate
        self.neurons = neurons
    def Train(self, data, labels):
        