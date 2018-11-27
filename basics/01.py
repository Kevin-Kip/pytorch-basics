def perceptron(grade, test):
    return (3 * grade) + test - 18

def step(grade, test):
    result = perceptron(grade, test)
    if result > 0:
        print("{0} : Student is accepted".format(result))
    else:
        print("{0} : Student is rejected".format(result))

if __name__ == '__main__':
    grade = int(input("Grade: "))
    test = int(input("Test: "))
    print(step(grade,test))
