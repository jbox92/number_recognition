allData = [[[1, 2], [5, 6], [8, 9],], [["b", "c"], ["b", "c"], ["b", "c"]]]

dataToUse = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

for i in range(len(allData)):
    data = allData[i]
    for d in range(len(data)):
        for j in range(len(data[d])):
            dataToUse[i][d * 2 + j] = data[d][j]

print(dataToUse)

import numpy as np
testnumpy = np.zeros(shape=(2,9))
print(testnumpy)