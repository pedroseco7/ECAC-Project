import numpy as np
import os
import matplotlib.pyplot as plt

#Global variables
activities = {
        1: 'STAND',
        2: 'SIT',
        3: 'SIT AND TALK',
        4: 'WALK',
        5: 'WALK AND TALK',
        6: 'CLIMB STAIR (UP/DOWN)',
        7: 'CLIMB STAIR (UP/DOWN) AND TALK',
        8: 'STAND->SIT',
        9: 'SIT->STAND',
        10: 'STAND->SIT AND TALK',
        11: 'SIT->STAND AND TALK',
        12: 'STAND->WALK',
        13: 'WALK->STAND',
        14: 'STAND->CLIMB STAIRS (UP/DOWN), STAND->CLIMB STAIRS (UP/DOWN) AND TALK',
        15: 'CLIMB STAIRS (UP/DOWN)->WALK',
        16: 'CLIMB STAIRS (UP/DOWN) AND TALK->WALK AND TALK'
    }

#Exercício 2: Descarregue os dados através do link indicado em cima e elabore uma rotina que carregue os dados relativos a um indivíduo e os devolva num Array NumPy.
def loadData():
    """Carrega dados de todos os ficheiros CSV com tratamento de erros"""
    base_dir = 'dataset'
    dataset = {}
    
    for i in range(0, 15):
        folder = os.path.join(base_dir, f"part{i}")
        for j in range(1, 6):
            file = f"part{i}dev{j}.csv"
            path = os.path.join(folder, file)
            data = np.loadtxt(path, delimiter=',')
            if data is None:
                continue

            dataset[f"part{i}dev{j}"] = data
    return dataset

#Exercício 3.1
def calculateModuleVariable(value1,value2,value3):
    return np.sqrt(value1**2 + value2**2 + value3**2)

def loadBoxPlotActivityAndVariable(data, sensor_type):

    if data is None:
        return
    
    # Definir colunas para cada sensor
    sensor_columns = {
        'acceleration': (1, 2, 3),
        'gyroscope': (4, 5, 6),
        'magnetometer': (7, 8, 9)
    }
    
    if sensor_type not in sensor_columns:
        print(f"Sensor type '{sensor_type}' não suportado.")
        return
    
    #obter as colunas corretas
    col_x, col_y, col_z = sensor_columns[sensor_type]
    activity_data = {}

    for key, values in data.items():
        for row in values:
            activity = int(row[11]) #buscar cada label de atividade
            
            #buscar os valores x,y,z do sensor que queremos
            x_val = row[col_x]
            y_val = row[col_y]
            z_val = row[col_z]
            
            #calcular o módulo com a formula que dão
            module = calculateModuleVariable(x_val, y_val, z_val)
            
            #adicionar o módulo ao dicionário de atividades
            if activity not in activity_data:
                activity_data[activity] = []
            activity_data[activity].append(module)
    
    
    #ordenar atividades para garantir consistência (o copilot aconselhou)
    sorted_activities = sorted(activity_data.keys())
    
    boxplot_data = [activity_data[activity] for activity in sorted_activities]
    activity_labels = [activities.get(activity, f'Activity {activity}') for activity in sorted_activities]
    
    #fazer o boxplot
    plt.figure()
    plt.boxplot(boxplot_data)
    plt.xticks(range(1, len(activity_labels) + 1), activity_labels, rotation=45)
    plt.ylabel(f'{sensor_type.capitalize()} Module')
    plt.title(f'Boxplot - {sensor_type.capitalize()} Module by Activity')
    plt.show()
    
    return

def main():

    #2.
    dataset = loadData()
    
    if not dataset:
        print("Nenhum dado carregado!")
        return
    else:
        for key, values in dataset.items():
            print(f"Key: {key}, Values: {values}")
            break

    #3.1
    print("\n=== Acelerómetro ===")
    loadBoxPlotActivityAndVariable(dataset, 'acceleration')

if __name__ == "__main__":
    main()