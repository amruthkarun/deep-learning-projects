import nn
import numpy as np

# AND Logic Function
# w1 = 1, w2 = 1, b = -2
def AND_logicFunction(x):
    w = np.array([1, 1])
    b = -2
    return nn.perceptronModel(x, w, b)

def main():
    # testing the Perceptron Model
    dataset = np.array([
        [0,0], [0, 1], [1, 0], [1, 1]
    ])

    print("AND Gate 2 Input:")
    for data in dataset:
        print("AND({}, {}) = {}".format(data[0], data[1], AND_logicFunction(data)))

if __name__ == '__main__':
   main()