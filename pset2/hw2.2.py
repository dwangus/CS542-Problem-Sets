import numpy as np
import scipy.io as sio
import math
from operator import itemgetter

def a():
    extract = sio.loadmat('detroit.mat')
    data = extract['data']
    
    def basisfunc(x):
        return x
    
    all_design_matrices = []
    
    output_vector = []
    for i in range(len(data)):
        output_vector.append(data[i][-1])
    output = np.array(output_vector)
    
    for k in range(1,8):
        design_matrix = []
        for i in range(len(data)):
            row = []
            row.append(basisfunc(data[i][0]))
            row.append(basisfunc(data[i][8]))
            row.append(basisfunc(data[i][k]))
            design_matrix.append(row)
        all_design_matrices.append(design_matrix)
    
    errors = []
    for i in range(7):
        design = np.array(all_design_matrices[i])
        
        phi_avg = [0.0, 0.0, 0.0]
        t_avg = 0.0
        for j in range(len(design)):
            phi_avg += basisfunc(design[j])
            t_avg += basisfunc(output[j])
        phi_avg /= len(design)
        t_avg /= len(output)
        
        weights = ((np.linalg.inv((design.T).dot(design))).dot(design.T)).dot(output)
        
        result = 0.0
        for j in range(len(phi_avg)):
            result += weights[j]*phi_avg[j]
        w0 = t_avg - result
        
        loss = 0.0
        for n in range(len(design)):
            third_term = (weights.T).dot(design[n])
            loss += (output[n] - w0 - third_term) ** 2
            '''
            msum = 0.0
            for m in range(len(design[0])):
                msum += design[n][m]*weights[m]
            loss += (output[n] - w0 - msum) ** 2
            '''
        loss /= 2
        errors.append(loss)
    
    print("Errors of each corresponding column, starting with Errors[0] = UEMP")
    print(errors)
    lowest_err_col = errors.index(min(errors)) + 1
    if lowest_err_col == 1:
        print("FTP, WE, + Third variable in determining HOM is UEMP")
    elif lowest_err_col == 2:
        print("FTP, WE, + Third variable in determining HOM is MAN")
    elif lowest_err_col == 3:
        print("FTP, WE, + Third variable in determining HOM is LIC")
    elif lowest_err_col == 4:
        print("FTP, WE, + Third variable in determining HOM is GR")
    elif lowest_err_col == 5:
        print("FTP, WE, + Third variable in determining HOM is NMAN")
    elif lowest_err_col == 6:
        print("FTP, WE, + Third variable in determining HOM is GOV")
    elif lowest_err_col == 7:
        print("FTP, WE, + Third variable in determining HOM is HE")

def b():
    def process(filename, name = "lenses"):
        values = []
        with open(filename, "r") as filestream:
            for line in filestream:
                values.append(line.split(","))
                if name == 'crx':
                    if values[-1][1] != '?':
                        values[-1][1] = float(values[-1][1])
                    if values[-1][2] != '?':
                        values[-1][2] = float(values[-1][2])
                    if values[-1][7] != '?':
                        values[-1][7] = float(values[-1][7])
                    if values[-1][10] != '?':
                        values[-1][10] = float(values[-1][10])
                    if values[-1][13] != '?':
                        values[-1][13] = float(values[-1][13])
                    if values[-1][14] != '?':
                        values[-1][14] = float(values[-1][14])
                    values[-1][15] = values[-1][15][0]
                else:
                    values[-1][4] = values[-1][4][0]
        return values
        
    def process_crx(arr):
        cag = [[[0,0],[0,0]], 0.0, 0.0, [[0,0,0,0],[0,0,0,0]], [[0,0,0],[0,0,0]], [[0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0]], [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]], 0.0, [[0,0],[0,0]], [[0,0],[0,0]], 0.0, [[0,0],[0,0]], [[0,0,0],[0,0,0]], 0.0, 0.0, [0,0]]
        for line in arr:
            if line[1] != "?":
                cag[1] += line[1]
            if line[2] != "?":
                cag[2] += line[2]
            if line[7] != "?":
                cag[7] += line[7]
            if line[10] != "?":
                cag[10] += line[10]
            if line[13] != "?":
                cag[13] += line[13]
            if line[14] != "?":
                cag[14] += line[14]
            if line[-1] == "+":
                if line[0] == 'a':
                    cag[0][0][0] += 1
                elif line[0] == 'b':
                    cag[0][0][1] += 1
                
                if line[3] == "u":
                    cag[3][0][0] += 1
                elif line[3] == "y":
                    cag[3][0][1] += 1
                elif line[3] == "l":
                    cag[3][0][2] += 1
                elif line[3] == "t":
                    cag[3][0][3] += 1
                
                if line[4] == "g":
                    cag[4][0][0] += 1
                elif line[4] == "p":
                    cag[4][0][1] += 1
                elif line[4] == "gg":
                    cag[4][0][2] += 1
                
                if line[5] == "c":
                    cag[5][0][0] += 1
                elif line[5] == "d":
                    cag[5][0][1] += 1
                elif line[5] == "cc":
                    cag[5][0][2] += 1
                elif line[5] == "i":
                    cag[5][0][3] += 1
                elif line[5] == "j":
                    cag[5][0][4] += 1
                elif line[5] == "k":
                    cag[5][0][5] += 1
                elif line[5] == "m":
                    cag[5][0][6] += 1
                elif line[5] == "r":
                    cag[5][0][7] += 1
                elif line[5] == "q":
                    cag[5][0][8] += 1
                elif line[5] == "w":
                    cag[5][0][9] += 1
                elif line[5] == "x":
                    cag[5][0][10] += 1
                elif line[5] == "e":
                    cag[5][0][11] += 1
                elif line[5] == "aa":
                    cag[5][0][12] += 1
                elif line[5] == "ff":
                    cag[5][0][13] += 1
                
                if line[6] == "v":
                    cag[6][0][0] += 1
                elif line[6] == "h":
                    cag[6][0][1] += 1
                elif line[6] == "bb":
                    cag[6][0][2] += 1
                elif line[6] == "j":
                    cag[6][0][3] += 1
                elif line[6] == "n":
                    cag[6][0][4] += 1
                elif line[6] == "z":
                    cag[6][0][5] += 1
                elif line[6] == "dd":
                    cag[6][0][6] += 1
                elif line[6] == "ff":
                    cag[6][0][7] += 1
                elif line[6] == "o":
                    cag[6][0][8] += 1
                
                if line[8] == "t":
                    cag[8][0][0] += 1
                elif line[8] == "f":
                    cag[8][0][1] += 1
                
                if line[9] == "t":
                    cag[9][0][0] += 1
                elif line[9] == "f":
                    cag[9][0][1] += 1
                
                if line[11] == "t":
                    cag[11][0][0] += 1
                elif line[11] == "f":
                    cag[11][0][1] += 1
                
                if line[12] == "g":
                    cag[12][0][0] += 1
                elif line[12] == "p":
                    cag[12][0][1] += 1
                elif line[12] == "s":
                    cag[12][0][2] += 1
            else:
                if line[0] == 'a':
                    cag[0][1][0] += 1
                elif line[0] == 'b':
                    cag[0][1][1] += 1
                
                if line[3] == "u":
                    cag[3][1][0] += 1
                elif line[3] == "y":
                    cag[3][1][1] += 1
                elif line[3] == "l":
                    cag[3][1][2] += 1
                elif line[3] == "t":
                    cag[3][1][3] += 1
                
                if line[4] == "g":
                    cag[4][1][0] += 1
                elif line[4] == "p":
                    cag[4][1][1] += 1
                elif line[4] == "gg":
                    cag[4][1][2] += 1
                
                if line[5] == "c":
                    cag[5][1][0] += 1
                elif line[5] == "d":
                    cag[5][1][1] += 1
                elif line[5] == "cc":
                    cag[5][1][2] += 1
                elif line[5] == "i":
                    cag[5][1][3] += 1
                elif line[5] == "j":
                    cag[5][1][4] += 1
                elif line[5] == "k":
                    cag[5][1][5] += 1
                elif line[5] == "m":
                    cag[5][1][6] += 1
                elif line[5] == "r":
                    cag[5][1][7] += 1
                elif line[5] == "q":
                    cag[5][1][8] += 1
                elif line[5] == "w":
                    cag[5][1][9] += 1
                elif line[5] == "x":
                    cag[5][1][10] += 1
                elif line[5] == "e":
                    cag[5][1][11] += 1
                elif line[5] == "aa":
                    cag[5][1][12] += 1
                elif line[5] == "ff":
                    cag[5][1][13] += 1
                
                if line[6] == "v":
                    cag[6][1][0] += 1
                elif line[6] == "h":
                    cag[6][1][1] += 1
                elif line[6] == "bb":
                    cag[6][1][2] += 1
                elif line[6] == "j":
                    cag[6][1][3] += 1
                elif line[6] == "n":
                    cag[6][1][4] += 1
                elif line[6] == "z":
                    cag[6][1][5] += 1
                elif line[6] == "dd":
                    cag[6][1][6] += 1
                elif line[6] == "ff":
                    cag[6][1][7] += 1
                elif line[6] == "o":
                    cag[6][1][8] += 1
                
                if line[8] == "t":
                    cag[8][1][0] += 1
                elif line[8] == "f":
                    cag[8][1][1] += 1
                
                if line[9] == "t":
                    cag[9][1][0] += 1
                elif line[9] == "f":
                    cag[9][1][1] += 1
                
                if line[11] == "t":
                    cag[11][1][0] += 1
                elif line[11] == "f":
                    cag[11][1][1] += 1
                
                if line[12] == "g":
                    cag[12][1][0] += 1
                elif line[12] == "p":
                    cag[12][1][1] += 1
                elif line[12] == "s":
                    cag[12][1][2] += 1
        nominal = [1,2,7,10,13,14]
        means = []
        stds = []
        for index in nominal:
            mean = cag[index]
            count = 0.0
            for item in arr:
                if item[index] != "?":
                    count += 1.0
            mean /= count
            std = 0.0
            for item in arr:
                if item[index] != "?":
                    std += (item[index] - mean) ** 2
            std = math.sqrt(std/(count - 1))
            means.append(mean)
            stds.append(std)
            for x in range(len(arr)):
               if  arr[x][index] == "?":
                   arr[x][index] = mean
               else:
                   arr[x][index] = (arr[x][index] - mean)/std
        
        for i in range(len(arr)):
            for j in range(len(arr[0]) - 1):
                if j not in nominal:
                    if arr[i][-1] == "+":
                        index = cag[j][0].index(max(cag[j][0]))
                    else:
                        index = cag[j][1].index(max(cag[j][1]))
                    if arr[i][j] == '?':
                        if j == 0:
                            if index == 0:
                                arr[i][j] = 'a'
                            else:
                                arr[i][j] = "b"
                        elif j == 3:
                            if index == 0:
                                arr[i][j] = 'u'
                            elif index == 1:
                                arr[i][j] = 'y'
                            elif index == 2:
                                arr[i][j] = 'l'
                            else:
                                arr[i][j] = "t"
                        elif j == 4:
                            if index == 0:
                                arr[i][j] = 'g'
                            elif index == 1:
                                arr[i][j] = 'p'
                            else:
                                arr[i][j] = "gg"
                        elif j == 5:
                            if index == 0:
                                arr[i][j] = 'c'
                            elif index == 1:
                                arr[i][j] = 'd'
                            elif index == 2:
                                arr[i][j] = 'cc'
                            elif index == 3:
                                arr[i][j] = 'i'
                            elif index == 4:
                                arr[i][j] = 'j'
                            elif index == 5:
                                arr[i][j] = 'k'
                            elif index == 6:
                                arr[i][j] = 'm'
                            elif index == 7:
                                arr[i][j] = 'r'
                            elif index == 8:
                                arr[i][j] = 'q'
                            elif index == 9:
                                arr[i][j] = 'w'
                            elif index == 10:
                                arr[i][j] = 'x'
                            elif index == 11:
                                arr[i][j] = 'e'
                            elif index == 12:
                                arr[i][j] = 'aa'
                            else:
                                arr[i][j] = "ff"
                        elif j == 6:
                            if index == 0:
                                arr[i][j] = 'v'
                            elif index == 1:
                                arr[i][j] = 'h'
                            elif index == 2:
                                arr[i][j] = 'bb'
                            elif index == 3:
                                arr[i][j] = 'j'
                            elif index == 4:
                                arr[i][j] = 'n'
                            elif index == 5:
                                arr[i][j] = 'z'
                            elif index == 6:
                                arr[i][j] = 'dd'
                            elif index == 7:
                                arr[i][j] = 'ff'
                            else:
                                arr[i][j] = "o"
                        elif j == 8:
                            if index == 0:
                                arr[i][j] = 't'
                            else:
                                arr[i][j] = "f"
                        elif j == 9:
                            if index == 0:
                                arr[i][j] = 't'
                            else:
                                arr[i][j] = "f"
                        elif j == 11:
                            if index == 0:
                                arr[i][j] = 't'
                            else:
                                arr[i][j] = "f"
                        elif j == 12:
                            if index == 0:
                                arr[i][j] = 'g'
                            elif index == 1:
                                arr[i][j] = 'p'
                            else:
                                arr[i][j] = "s"
        return arr
    
    def DL2(p1, p2):
        dist = 0.0
        for i in range(len(p1) - 1):
            #print(p1[i])
            #print(p2[i])
            if type(p1[i]) is float or type(p2[i]) is float:
                dist += (p1[i] - p2[i]) ** 2
            else:
                if p1[i] != p2[i]:
                    dist += 1
        return math.sqrt(dist)
    def training(test_point, train_set):
        distances = []
        for i in range(len(train_set)):
            distances.append((i, DL2(train_set[i], test_point)))
        distances = sorted(distances, key=itemgetter(1))
        return distances
    def output(test_set, train_set, k, filename = "lenses"):
        neighbors = []
        closest = []
        predicts = []
        for test in test_set:
            dist = training(test, train_set)
            neighbors.append(dist)
        for n in neighbors:
            closei = []
            for i in range(k):
                closei.append(n[i])
            closest.append(closei)
        for neigh in closest:
            if filename == "lenses":
                three = 0
                two = 0
                one = 0
                for category in neigh:
                    if train_set[category[0]][-1] == "3":
                        three += 1
                    elif train_set[category[0]][-1] == "2":
                        two += 1
                    else:
                        one += 1
                if three > two and three > one:
                    predicts.append("3")
                elif two > three and two > one:
                    predicts.append("2")
                else:
                    predicts.append("1")
            else:
                plus = 0
                minus = 0
                for category in neigh:
                    if train_set[category[0]][-1] == "+":
                        plus += 1
                    else:
                        minus += 1
                if plus > minus:
                    predicts.append("+")
                else:
                    predicts.append("-")
        return predicts
    def accuracy(predict, testing_set):
        actual = []
        for item in testing_set:
            actual.append(item[-1])
        correct = 0.0
        for i in range(len(predict)):
            if predict[i] == actual[i]:
                correct += 1
        return float(correct / len(predict))
        
            
    lenses_training = process("lenses.training.txt")
    lenses_testing = process("lenses.testing.txt")
    lens3 = output(lenses_testing, lenses_training, 3)
    lens5 = output(lenses_testing, lenses_training, 5)
    print("\n\nTesting Lenses Data, w/ K = 3:")
    print(lens3)
    print("Accuracy: {0}".format(accuracy(lens3, lenses_testing)))
    print("Testing Lenses Data, w/ K = 5:")
    print(lens5)
    print("Accuracy: {0}".format(accuracy(lens5, lenses_testing)))

    crx_training = process_crx(process("crx.data.training.txt", "crx"))
    crx_testing = process_crx(process("crx.data.testing.txt", "crx"))
    crx5 = output(crx_testing, crx_training, 5, "crx")
    crx10 = output(crx_testing, crx_training, 10, "crx")
    print("\n\nTesting CRX Data, w/ K = 5:")
    print(crx5)
    print("Accuracy: {0}".format(accuracy(crx5, crx_testing)))
    print("Testing CRX Data, w/ K = 10:")
    print("Accuracy: {0}".format(accuracy(crx10, crx_testing)))
    print(crx10)

a()
b()
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
