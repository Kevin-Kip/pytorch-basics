import math as m

def perceptron(x1, x2):
    result = (4 * x1) + (5 * x2) - 9
    x = sigmoid(result)
    return result, x

def sigmoid(x):
    return 1 / (1 + m.exp(-x))

if __name__ == '__main__':
    x1 = int(input("Input 1: "))
    x2 = int(input("Input 2: "))
    value = perceptron(x1, x2)
    percent = value[1] * 100
    print("Value is {0} and prediction is {1}, which is {2}%".format(value[0],value[1],int(percent)))
