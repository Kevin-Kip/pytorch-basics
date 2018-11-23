import numpy as np

e = np.e
results = []

# SOftmax using numpy
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# A very inneficient version of softmax
def plainmax(x):
    probs = 0 # this holds the (e^z1+e^z2+e^z3)
    for item in x:
        probs += (e ** item)

    for each in x:
        results.append((e ** each) / probs)

    return results

if __name__ == '__main__':
    input_array = [2,3]
    plainmax(input_array)
    item = np.max(input_array)
    probability = np.max(results)
    percentage = (np.max(input_array) / np.sum(input_array))*100
    print("Most likely is {0} and its probability {1}, which translates to {2}%".format(item,probability, percentage))
