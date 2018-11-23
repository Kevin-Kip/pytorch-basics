import numpy as np

e = np.e

# SOftmax using numpy
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# A very inneficient version of softmax
def plainmax(x):
    probs = 0 # this holds the (e^z1+e^z2+e^z3)
    results = []
    for item in x:
        probs += (e ** item)

    for each in x:
        results.append((e ** each) / probs)

    return results

if __name__ == '__main__':
    print(softmax([2,2]))
    print(plainmax([2,2]))
