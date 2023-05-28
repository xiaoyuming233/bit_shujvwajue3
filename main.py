import csv
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def judge3(a):
    if a == "Yes":
        return 1
    elif a == "No":
        return 0
    return 2


def judge2(a):
    if a == "Yes":
        return 1
    elif a == "No":
        return 0
    print("Wrong!")
    return 2


def judge8(a):
    if a == "DSL":
        return 1
    elif a == "Fiber optic":
        return 2
    elif a == "No":
        return 3
    print("Wrong!")
    return 0


def judge15(a):
    if a == "Month-to-month":
        return 1
    elif a == "One year":
        return 2
    elif a == "Two year":
        return 3
    print("Wrong!")
    return 0


def judge17(a):
    if a == "Electronic check":
        return 1
    elif a == "Mailed check":
        return 2
    elif a == "Bank transfer (automatic)":
        return 3
    elif a == "Credit card (automatic)":
        return 4
    print("Wrong!")
    return 0


def judgeMode(a, b):
    if a == 1:
        if b == 1:
            return 3
        if b == 0:
            return 2
    if a == 0:
        if b == 1:
            return 1
        if b == 0:
            return 0
    print("Wrong!")
    return 4


def calculateS(a):
    y_bar = sum(a) / len(a)
    sons = 0
    for yi in a:
        sons += (yi - y_bar) ** 2
    return sons ** 0.5
    # 创建一个训练数据集


def readData():
    with open('WA_Fn-UseC_-Telco-Customer-Churn.csv', encoding='utf-8-sig') as f:
        allInfo = []
        allX = []
        allY = []
        indexJudge3 = [7, 9, 10, 11, 12, 13, 14]
        for row in csv.reader(f, skipinitialspace=True):
            storage = True
            for item in row:
                if len(item) == 0:
                    storage = False
                    break
            if storage:
                allInfo.append(row)
        allInfo.pop(0)
        for line in allInfo:
            allY.append(judge2(line[20]))
            newLine = []
            if line[1] == "Female":
                newLine.append(1)
            elif line[1] == "Male":
                newLine.append(0)
            else:
                print("Wrong!")
            newLine.append(int(line[2]))
            newLine.append(int(line[5]))
            newLine.append(judge2(line[3]))
            newLine.append(judge2(line[4]))
            newLine.append(judge2(line[6]))
            newLine.append(judge2(line[16]))
            for j in indexJudge3:
                newLine.append(judge3(line[j]))
            newLine.append(judge8(line[8]))
            newLine.append(judge15(line[15]))
            newLine.append(judge17(line[17]))
            newLine.append(float(line[18]))
            newLine.append(float(line[19]))
            allX.append(newLine)
    f.close()
    return allX, allY


X, Y = readData()
Sy = calculateS(Y)
for index in range(len(X[0])):
    newX = []
    for line in X:
        newX.append(line[index])
    xBar = sum(newX) / len(newX)
    yBar = sum(Y) / len(Y)
    Sx = calculateS(newX)
    son = 0
    for i in range(len(X)):
        son += newX[i] * Y[i]
    son -= len(X) * xBar * yBar
    R = son / (Sy * Sx)
    print(index, R)

# X = np.delete(X, [0, 5, 7, 14], axis=1)
amount = len(X)
trainNum = (amount // 10) * 8
x_train = X[0:trainNum]
y_train = Y[0:trainNum]
x_test = X[trainNum:]
y_test = Y[trainNum:]
# 创建一个决策树模型
reg = DecisionTreeRegressor()

# 训练模型
reg.fit(x_train, y_train)

# 预测新数据
predicted_y = reg.predict(x_test)

Tp = 0
Fp = 0
Fn = 0
Tn = 0
for i in range(len(predicted_y)):
    temp = judgeMode(int(predicted_y[i]), y_test[i])
    if temp == 3:
        Tp += 1
    if temp == 2:
        Fp += 1
    if temp == 1:
        Fn += 1
    if temp == 0:
        Tn += 1

acc = (Tp + Tn) / len(predicted_y)
pre = Tp / (Tp + Fp)
rec = Tp / (Tp + Fn)
F1 = (2 * pre * rec) / (pre + rec)

# 打印预测结果
print(acc)
print(pre)
print(rec)
print(F1)
# for i in predicted_y:
#     print(int(i))
