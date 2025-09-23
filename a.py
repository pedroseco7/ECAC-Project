import numpy as np
import os
import matplotlib.pyplot as plt

def loadData(filePath):
    data = np.loadtxt(filePath, delimiter=',')
    return data

def loadBoxPlotAndVariable(data):
    return


def main():
    base_dir = 'dataset'

    dict = {}
    for i in range(0, 15):
        for j in range(1, 6):
            dict[f"part{i}dev{j}"] = {}

    for i in range(0, 15):
        folder = os.path.join(base_dir, f"part{i}")
        for j in range(1, 6):
            file = f"part{i}dev{j}.csv"
            path = os.path.join(folder, file)
            data = loadData(path)
            if data is None:
                continue

            dict[f"part{i}dev{j}"] = data

    for values in dict.values():
        print(values)
        break

if __name__ == "__main__":
    main()