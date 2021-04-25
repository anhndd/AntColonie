import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2

trainingFilename = "/home/dell/PycharmProjects/AI2/TP2/Uses_Cases/Uses_Cases/Spam/Spam detection - For model creation.csv"
testFilename = "/home/dell/PycharmProjects/AI2/TP2/Uses_Cases/Uses_Cases/Spam/Spam detection - For prediction.csv"

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
    model = Sequential([
      Dense(units=1, activation='sigmoid', kernel_regularizer=l2(0.), input_shape=(len(X_train[0]),))
    ])

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=1, verbose=0)
    _,acc = model.evaluate(X_test, Y_test)
    print("accuracy:",acc*100,"%")
