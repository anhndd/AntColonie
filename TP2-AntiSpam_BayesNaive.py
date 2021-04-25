import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2

trainingFilename = "/home/dell/PycharmProjects/AI2/TP2/Uses_Cases/Uses_Cases/Spam/Spam detection - For model creation.csv"
testFilename = "/home/dell/PycharmProjects/AI2/TP2/Uses_Cases/Uses_Cases/Spam/Spam detection - For prediction.csv"


def exp_function(u, X_train, Y_train):
    countYes = 0
    countNo = 0
    lenX = len(X_train[0])
    for i in range(0, len(Y_train)):
        X = X_train[i]
        y = Y_train[i]
        if y == 0:
            countNo += 1
        elif y == 1:
            countYes += 1
        for j in range(0, lenX):
            u[j][y] += X[j]
    for j in range(0, lenX):
        u[j][0] /= countNo
        u[j][1] /= countYes


def var_function(u, q, X_train, Y_train):
    countYes = 0
    countNo = 0
    lenX = len(X_train[0])
    for i in range(0, len(Y_train)):
        X = X_train[i]
        y = Y_train[i]
        if y == 0:
            countNo += 1
        elif y == 1:
            countYes += 1
        for j in range(0, lenX):
            q[j][y] += ((X[j] - u[j][y]) ** 2)
    for j in range(0, lenX):
        q[j][0] /= (countNo - 1)
        q[j][1] /= (countYes - 1)


def predict_function(x, u, q):
    f_0 = 1
    f_1 = 1
    for i in range(0, len(x)):
        f_0 *= np.exp(-(x[i] - u[i][0]) ** 2 / (2 * q[i][0])) / np.sqrt(q[i][0])
        f_1 *= np.exp(-(x[i] - u[i][1]) ** 2 / (2 * q[i][1])) / np.sqrt(q[i][1])
    t = f_1 / f_0
    if t > 1:
        return 1
    return 0


def accuracy_func(X, Y, u, q):
    count = 0
    for i in range(0, len(Y)):
        x = X[i]
        y = Y[i]
        y_predict = predict_function(x, u, q)
        if y_predict == y:
            count += 1
    print("accuracy: ", count / len(Y) * 100, "%")


if __name__ == '__main__':
    file = csv.reader(open(trainingFilename, "rt"))
    next(file)
    testFile = csv.reader(open(testFilename, "rt"))
    next(testFile)
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    a = 0.8

    for row in file:
        row_value = row[0].split(";")
        y = 0
        if row_value[0] == 'Yes':
            y = 1
        x = [float(x) for x in row_value[1:]]
        X_train.append(x)
        Y_train.append(y)

    for row in testFile:
        y = int(row[-1])
        x = [float(x) for x in row[0:len(row) - 1]]
        X_test.append(x)
        Y_test.append(y)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    expectation = [[0, 0]] * len(X_train[0])
    variance = [[0, 0]] * len(X_train[0])
    exp_function(expectation, X_train, Y_train)
    var_function(expectation, variance, X_train, Y_train)
    accuracy_func(X_test, Y_test, expectation, variance)
    # print("accuracy:", acc * 100, "%")
