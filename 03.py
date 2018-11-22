import nupy as np

def perceptron(x1,x2):
    pass

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

if __name__ == '__main__':
    pass
