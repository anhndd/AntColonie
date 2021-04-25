import numpy as np
import random, sys, time
from ant import Ant

args = sys.argv
isCircuit = False
number_node = 10

# setup finding shortest circuit or shotest path and how many node
if len(args) > 1:
    number_node = int(args[1])

    if len(args) > 2:
        isCircuit = args[2]
        if (isCircuit == 1) | (isCircuit == "circuit"):
            isCircuit = True
        else:
            isCircuit = False

number_ant = 300
alpha = 1.035
beta = 0.026
Gamma = 0.02
Q = 100
p_evaporation = 0.9
track = [[0] * number_node for x in range(0, number_node)]

graph = [[random.randrange(1, 100) if x != y else 0 for x in range(0, number_node)] for y in range(0, number_node)]

# test specific case
if number_node == 4:
    graph = [[0, 57, 46, 20], [67, 0, 18, 13], [83, 7, 0, 18], [58, 76, 88, 0]]; number_node = 4
elif number_node == 5:
    graph = [[77, 99, 51, 35, 99], [18, 94, 25, 24, 74], [94, 68, 98, 33, 6], [96, 76, 26, 56, 82], [55, 9, 43, 12, 63]]; number_node = 5
elif number_node == 10:
    graph = [[0, 36, 94, 57, 43, 89, 6, 30, 55, 80], [60, 0, 33, 65, 13, 64, 56, 78, 23, 64], [84, 56, 0, 8, 46, 8, 25, 32, 50, 69], [20, 97, 86, 0, 84, 27, 80, 8, 99, 66], [82, 15, 45, 52, 0, 57, 24, 84, 86, 41], [68, 56, 72, 58, 12, 0, 26, 21, 10, 93], [45, 22, 60, 33, 64, 46, 0, 34, 62, 71], [19, 4, 50, 24, 64, 52, 60, 0, 89, 17], [84, 94, 5, 24, 98, 62, 53, 24, 0, 5], [13, 10, 84, 58, 6, 27, 8, 77, 20, 0]]; number_node = 10


def calculate_propability(from_node, path):
    p = [0] * number_node
    #  calculate choosing node probability
    total = 0
    for to_node in range(0, number_node):
        if (from_node, to_node) in path:
            p[to_node] = 0
        else:
            d = graph[from_node][to_node]
            if d > 0:  # if destination node is neighbor of current node
                p[to_node] = Gamma + (track[from_node][to_node] ** alpha) * ((1 / d) ** beta)
                total += p[to_node]
            else:
                p[to_node] = 0
    for to_node in range(0, number_node):
        if p[to_node] > 0:
            p[to_node] /= total
    return p


def pick_next_node(p):
    # pick the node
    r = random.uniform(0, 1)
    tot = 0
    for j in range(0, number_node):
        tot += p[j]
        if tot >= r:
            return j


def update_tracking_total(ants):
    for i in range(0, number_node):
        for j in range(0, number_node):
            # evaporation
            track[i][j] = track[i][j] * (1 - p_evaporation)
    for ant in ants:
        contribution = Q / ant.total_length
        for i in range(0, number_node):
            for j in range(0, number_node):
                if (i, j) in ant.path:
                    track[i][j] += contribution

if __name__ == '__main__':
    start_time = time.time()
    # graph in format array
    print(graph)

    # setup objective
    destination = number_node - 1

    # for the objective
    shortest_path_length = np.inf
    objective_path = []
    objective_loop = -1

    # for the result of algorithm
    count_converge = 0
    path_converge = []
    max_length_converge = np.inf

    t = 0

    # start algorithm
    while True:
        most_path = {}
        ants = [Ant() for x in range(0, number_ant)]
        count = 0

        # each ant go to the destination
        for ant in ants:
            from_node = 0
            while True:
                p = calculate_propability(from_node, ant.path)  # calculate probability for picking next node
                to_node = pick_next_node(p)  # get next node

                if (from_node, to_node) in ant.path:  # this is the error case (should not be true)
                    print("error", to_node, ant.path)
                    break

                ant.path.append((from_node, to_node))  # flag the new node already explored for each ant
                ant.total_length += graph[from_node][to_node]  # calculate total length

                from_node = to_node

                # In mode finding the shortest circuit, check IF an ant reach the destination
                if isCircuit & (to_node == destination):
                    ant.reach_destination = True

                # If an ant finishs the tour, save result
                if (isCircuit & (to_node == 0) & ant.reach_destination) | ((not isCircuit) & (to_node == destination)):

                    # save temporarily the path each time an ant finish its tour
                    if str(ant.path) in most_path:
                        most_path[str(ant.path)][0] += 1
                    else:
                        most_path[str(ant.path)] = [1, ant.total_length]

                    # Save the possible shortest path in the graph for checking in the end of the program
                    if shortest_path_length > ant.total_length:
                        shortest_path_length = ant.total_length
                        if ant.path != objective_path:
                            objective_path = ant.path
                            objective_loop = t
                    break

        # print all path ants have traveled
        max_temp = max(most_path.values())
        path_temp = max(most_path, key=most_path.get)
        sumAnt = sum([x[0] for x in most_path.values()])
        most_path = dict(sorted(most_path.items(), key=lambda item: item[1]))
        print("--------------------------------------------------------")
        for x in most_path.keys():
            print(round(most_path[x][0] / sumAnt * 100, 2), "%", "Path", x, " length:", most_path[x][1])

        # update the path with the most number of ants
        if count_converge < max_temp[0]:
            count_converge = max_temp[0]
            path_converge = path_temp
            max_length_converge = most_path[path_converge][1]
            print("Loop", t, "Max ant travel the same way", path_temp, ":", max_temp[0], "ants")

        # if algorithm is converged (All ants travel the same way) => End
        if max_temp[0] == number_ant:
            break
        # if not, continue update tracks in each path
        update_tracking_total(ants)
        t += 1

    # End algorithm, print the objective and hope the algorithm converge in the objective path
    print("\nSUCCESS" if (
            (path_converge == str(objective_path)) | (max_length_converge == shortest_path_length)) else "FAIL",
          "objective shortest path:", objective_path,
          "shortest length:", shortest_path_length, "LOOP:", objective_loop)

    # graph in format matrix
    row_labels = [x for x in range(0, number_node)]
    graph_color = np.insert(np.insert(graph, 0, row_labels, axis=1), 0, [0] + row_labels, axis=0)
    print("\ngraph:")
    print(np.matrix(graph_color))

    print("--- %s seconds ---" % (time.time() - start_time))

    input("press enter to exit")
    # graph in format array
    print(graph)
