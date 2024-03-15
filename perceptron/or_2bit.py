import nn
import numpy as np

# OR Logic Function
# w1 = 1, w2 = 1, b = -0.5
def OR_logicFunction(x):
    w = np.array([1, 1])
    b = -0.5
    return nn.perceptronModel(x, w, b)

def main():
    # testing the Perceptron Model
    dataset = np.array([
        [0,0], [0, 1], [1, 0], [1, 1]
    ])

    print("OR Gate 2 Input:")
    for data in dataset:
        print("OR({}, {}) = {}".format(data[0], data[1], OR_logicFunction(data)))

if __name__ == '__main__':
   main()