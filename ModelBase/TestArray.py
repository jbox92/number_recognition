data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
dataToUseI = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for d in range(len(data)):
    for j in range(len(data[d])):
        dataToUseI[d * 3 + j] += data[d][j]

print(dataToUseI)