import numpy as np
import random, sys, time
from tp1ACO import test_model, number_ant,alpha,beta,Gamma,Q,p_evaporation,number_node


# graph = [[0, 3, 31, 90], [34, 0, 35, 3], [9, 49, 0, 10], [59, 34, 97, 0]];number_node=4  # [(0, 1), (1, 3), (3,0)] 65 6
# graph = [[0, 38, 54, 88], [66, 0, 69, 40], [63, 3, 0, 67], [9, 65, 30, 0]];number_node=4 # [(0, 1), (1, 3), (3, 0)] 87 78
# graph = [[0, 3, 31, 60], [34, 0, 35, 90], [9, 49, 0, 94], [59, 34, 97, 0]];number_node=4 # [[0, 3],[3,0]] 119 60
# graph = [[-1, 1, 2, 5, 7],[ 2,-1, 2, 4, 3],[ 2, 2,-1, 1, 2],[ 5, 4, 1,-1, 5],[ 7, 2, 2, 5,-1]];number_node = 5 #equal total but success [(0, 2), (2, 4), (4, 1), (1, 0)] 8 4
# graph = [[0, 76, 23, 60, 74], [27, 0, 85, 16, 30], [79, 52, 0, 64, 53], [65, 73, 96, 0, 30], [12, 94, 16, 37, 0]];number_node=5 # [(0, 4), (4, 0)] 86 74
# graph = [[-1, 2, 2, 5, 7],[ 2,-1, 2, 4, 2],[ 2, 2,-1, 1, 2],[ 5, 4, 1,-1, 5],[ 7, 2, 2, 5,-1]];number_node = 5 #[(0, 1), (1, 4), (4, 2), (2, 0)] 8 4
# graph = [[0, 45, 34, 1, 34], [27, 0, 7, 85, 17], [38, 34, 0, 74, 92], [88, 40, 96, 0, 36], [13, 87, 91, 2, 0]];number_node=5 #[(0, 4), (4, 0)] 47 34
# graph = [[0, 1, 30, 35, 96], [41, 0, 12, 98, 61], [97, 12, 0, 71, 90], [50, 87, 25, 0, 23], [22, 54, 37, 37, 0]];number_node=5 #[(0, 3), (3, 4), (4, 0)] 80 58

# check 3 case
# graph = [[4, 38, 54, 88], [66, 1, 69, 40], [63, 3, 3, 67], [9, 65, 30, 46]];number_node=4
# graph = [[19, 63, 37, 19, 82, 88, 31, 24, 89, 57], [83, 61, 20, 55, 3, 40, 2, 25, 44, 3], [27, 90, 60, 16, 79, 27, 16, 7, 90, 16], [40, 87, 76, 65, 34, 34, 29, 10, 12, 16], [70, 13, 49, 94, 1, 56, 77, 60, 27, 15], [52, 5, 34, 56, 16, 56, 67, 85, 96, 82], [39, 74, 81, 42, 22, 59, 92, 86, 10, 91], [95, 81, 40, 20, 1, 73, 90, 99, 3, 64], [54, 86, 70, 36, 40, 51, 69, 67, 86, 45], [81, 13, 95, 87, 91, 80, 58, 80, 44, 32]];number_node=10 # [(0, 3), (3, 9), (9, 1), (1, 6), (6, 0)] 89 35

# graph = [[0, 56, 2, 96, 60, 72, 3, 24, 96, 81], [68, 0, 2, 9, 99, 8, 77, 92, 13, 98], [14, 30, 0, 56, 50, 21, 68, 30, 53, 90], [88, 8, 45, 0, 19, 47, 87, 90, 69, 72], [92, 56, 62, 53, 0, 76, 44, 89, 57, 31], [38, 62, 99, 94, 85, 0, 56, 87, 73, 17], [47, 49, 94, 3, 95, 77, 0, 73, 84, 88], [1, 18, 54, 54, 19, 75, 15, 0, 83, 86], [26, 31, 71, 96, 2, 50, 46, 33, 0, 53], [60, 87, 70, 24, 41, 63, 27, 1, 74, 0]]
# graph = [[0, 56, 2, 96, 60, 72, 3, 24, 96, 81], [68, 0, 2, 9, 99, 8, 77, 92, 13, 98], [14, 15, 0, 56, 18, 21, 68, 4, 53, 41], [88, 8, 45, 0, 19, 47, 87, 90, 69, 72], [92, 56, 62, 53, 0, 76, 44, 89, 57, 31], [38, 62, 99, 94, 85, 0, 56, 87, 73, 17], [47, 49, 94, 3, 95, 77, 0, 73, 84, 88], [1, 18, 54, 54, 19, 75, 15, 0, 83, 86], [26, 31, 71, 96, 2, 50, 46, 33, 0, 53], [60, 87, 70, 24, 41, 63, 27, 1, 74, 0]] # always wrong cause 1 node has many approximate length with the shortest path
# graph = [[0, 49, 36, 82, 15, 27, 75, 8, 95, 30], [77, 0, 96, 3, 37, 52, 7, 13, 38, 39],[50, 79, 0, 71, 80, 43, 81, 70, 81, 45], [65, 95, 93, 0, 14, 44, 5, 15, 23, 27],[14, 92, 33, 15, 0, 43, 22, 33, 73, 23], [99, 33, 61, 68, 18, 0, 65, 88, 84, 67],[32, 92, 51, 94, 61, 12, 0, 72, 48, 65], [58, 83, 43, 89, 23, 63, 33, 0, 20, 4],[83, 50, 99, 2, 34, 69, 61, 93, 0, 45],[79, 23, 75, 61, 93, 69, 18, 56, 12, 0]]  # 54 12 success more with number ant = 300
# graph = [[0, 50, 93, 36, 42], [37, 0, 36, 96, 64], [85, 92, 0, 99, 6], [18, 93, 20, 0, 4], [55, 39, 24, 79, 0]];number_node=5 # wrong 95 and 97 approximate
# graph = [[0, 21, 82, 98, 81], [30, 0, 9, 57, 89], [89, 3, 0, 18, 43], [77, 23, 49, 0, 32], [13, 61, 28, 97, 0]];number_node=5 # [0, 4] has approximate length with the shortest path and has the most ant at the beginning
# graph = [[0, 57, 25, 73], [36, 0, 88, 9], [6, 31, 0, 99], [92, 15, 80, 0]]; number_node =4
# graph = [[0, 90, 97, 17, 83], [43, 0, 13, 77, 83], [10, 57, 0, 58, 71], [51, 47, 66, 0, 95], [68, 68, 50, 20, 0]]; number_node =5
# graph = [[0, 38, 73, 3, 91], [11, 0, 25, 4, 41], [31, 61, 0, 54, 12], [1, 61, 40, 0, 12], [71, 22, 50, 50, 0]]; number_node =5 # loop infinity [3][0] = 1 too short
# graph = [[0, 83, 37, 64], [64, 0, 50, 73], [36, 2, 0, 7], [96, 95, 14, 0]];number_node = 4
# graph = [[0, 36, 41, 67, 70], [86, 0, 9, 90, 14], [7, 96, 0, 84, 91], [98, 89, 65, 0, 73], [84, 40, 78, 63, 0]];number_node = 5
# graph = [[0, 54, 52, 81, 29, 36, 31, 87, 24, 57], [57, 0, 5, 31, 51, 94, 7, 43, 64, 15], [3, 69, 0, 64, 90, 78, 76, 63, 23, 97], [75, 55, 14, 0, 84, 52, 50, 94, 18, 55], [94, 24, 69, 75, 0, 50, 7, 80, 56, 27], [52, 1, 42, 91, 41, 0, 29, 32, 20, 84], [79, 3, 92, 6, 84, 28, 0, 72, 47, 81], [78, 44, 63, 61, 62, 98, 51, 0, 84, 78], [58, 57, 7, 16, 64, 78, 45, 75, 0, 37], [99, 2, 29, 90, 15, 57, 23, 63, 68, 0]]
args = sys.argv
isCircuit = False
if len(args) > 1:
    isCircuit = args[1]
    isCircuit = bool(int(isCircuit))

def readData(number_node):
    graphdata = []
    fileread = open("graph-"+str(number_node),"r")
    for row in fileread:
        graph = []
        count = 0
        rowarray = row.split()
        for i in range(0,number_node):
            graphtemp = []
            for j in range(0,number_node):
                graphtemp.append(int(rowarray[count].replace('[','').replace(',','').replace(']','')))
                count+=1
            graph.append(graphtemp)
        graphdata.append(graph)
    labels = readLabel(number_node, len(graphdata))
    return graphdata, labels

def createData(number_node):
    file = open("graph-"+str(number_node), "a+")
    for number_graph in range(0,100):
        graph = [[random.randrange(1, 100) if x != y else 0 for x in range(0, number_node)] for y in range(0, number_node)]
        file.write(str(graph)+"\n")
    file.close()
    file = open("graph-"+str(number_node)+"-labels", "w")
    file.close()

def updateLabel(number_node, shortest_lengths):
    global isCircuit
    strCircuit = "-circuit" if isCircuit else ""
    file = open("graph-"+str(number_node)+"-labels"+strCircuit, "w")
    file.write(str(shortest_lengths))
    file.close()

def readLabel(number_node, length):
    global isCircuit
    strCircuit = "-circuit" if isCircuit else ""
    file = open("graph-"+str(number_node)+"-labels"+strCircuit, "r")
    rowarray = file.readline().split()
    graphtemp = []
    for i in range(0,len(rowarray)):
        graphtemp.append(int(rowarray[i].replace('[','').replace(',','').replace(']','')))
    if graphtemp == []:
        graphtemp = [np.inf]*length
    return graphtemp

number_node = int(args[2])
# createData(number_node)
# print(readData(number_node))
# updateLabel(5,[1,2,3])
graphlist, labels = readData(number_node)
countFail = 0
totalGraph = len(graphlist)
totalTimeExecute = 0
for i in range(0,totalGraph):
    graph = graphlist[i]
    start_time = time.time()
    label, max_length_found = test_model(graph,number_node)
    time_excecute = time.time() -  start_time
    totalTimeExecute += time_excecute
    if label < labels[i]:
        print(label, labels[i])
        print("update label" ,i)
        labels[i] = label
    else:
        label = labels[i]
    if max_length_found != label:
        print(i, "FAIL", graph)
        countFail+=1
    print("label done:",i)
updateLabel(number_node, labels)
acc = 1 - (countFail*1./totalGraph)
print("Accuracy",acc, "--- %s seconds ---" % (totalTimeExecute), "mean:",totalTimeExecute / totalGraph)

# file = open("result","a+")
# file.write(str(isCircuit) + ", number_ant="+str(number_ant)+", alpha="+str(alpha)+", beta="+ str(beta) +", Gamma="+str(Gamma)+ ", Q="+str(Q)+", p_evaporation="+str(p_evaporation)+", number_node="+str(number_node)+", Gamma="+str(Gamma)+",accuracy="+str(acc)+"\n")
