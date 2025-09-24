import numpy as np
import os
import matplotlib.pyplot as plt

#Exercício 2: Descarregue os dados através do link indicado em cima e elabore uma rotina que carregue os dados relativos a um indivíduo e os devolva num Array NumPy.
def loadData():
    base_dir = 'dataset'

    dict = {}

    for i in range(0, 15):
        folder = os.path.join(base_dir, f"part{i}")
        for j in range(1, 6):
            file = f"part{i}dev{j}.csv"
            path = os.path.join(folder, file)
            data = np.loadtxt(path, delimiter=',')
            if data is None:
                continue

            dict[f"part{i}dev{j}"] = data
    return dict

def loadBoxPlotAndVariable(data):
    return

def main():

    dict = loadData()

    for key, values in dict.items():
        print(f"Key: {key}, Values: {values}")
        break

if __name__ == "__main__":
    main()