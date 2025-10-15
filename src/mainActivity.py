import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import zscore

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
def loadData(deviceId):
    """Carrega dados de todos os ficheiros CSV com tratamento de erros"""
    base_dir = 'dataset'
    dataset = {}
    
    if deviceId == None:
        for i in range(0, 15):
            folder = os.path.join(base_dir, f"part{i}")
            for j in range(1, 6):
                file = f"part{i}dev{j}.csv"
                path = os.path.join(folder, file)
                data = np.loadtxt(path, delimiter=',')
                if data is None:
                    continue

                dataset[f"part{i}dev{j}"] = data

    else:
        for i in range(0, 15):
            folder = os.path.join(base_dir, f"part{i}")
            file = f"part{i}dev{deviceId}.csv"
            path = os.path.join(folder, file)
            data = np.loadtxt(path, delimiter=',')

            if data is None:
                continue

            dataset[f"part{i}dev{deviceId}"] = data

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

#Exercício 3.2
def calculateDensity(no,nr):
    return (no/nr)*100

def outlierDensity(sensor_type):
    
    # Carregar dados apenas do device 2 (pulso direito)  
    dataset_right_pulse = loadData(2)
    
    if not dataset_right_pulse:
        print("Nenhum dado do pulso direito carregado!")
        return
    
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

    for key, values in dataset_right_pulse.items():
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

    for activity in sorted(activity_data.keys()):
        data_points = np.array(activity_data[activity])
        nr = len(data_points) #numero total de pontos da atividade

        #Calcular Q1, Q3 e IQR para detetar os outliers
        Q1 = np.percentile(data_points,25)
        Q3 = np.percentile(data_points,75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        #detetar os outliers
        outliers = data_points[(data_points < lower) | (data_points > upper)]
        no = len(outliers)

        density = calculateDensity(no, nr)

        print(f"Atividade: {activities.get(activity, f'Activity {activity}')}, Densidade de Outliers: {density:.2f}% (Outliers: {no}, Total: {nr})")
    
    return

#Exercício 3.3
def Zscore(x, mean, std, k):

    # X é a amostra 
    zscore = (x-mean) / std

    # Caso esteja acima ou abaixo de K, retorna 1 (identifica o outlier)
    if(np.abs(zscore) > k):
        return 1
    
    else:
        return 0
    

def outliers(data, sensor_type, k):

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


    for activity in sorted(activity_data.keys()):
        data_points = np.array(activity_data[activity])
        nr = len(data_points) #numero total de pontos da atividade

        outliers_counter_activity = 0
        mean = np.mean(activity_data[activity])
        std = np.std(activity_data[activity])

        for i in activity_data[activity]:

            if(Zscore(i, mean, std, k)):
                outliers_counter_activity += 1
        
        print(f"Atividade: {activities.get(activity, f'Activity {activity}')}, Outliers: {outliers_counter_activity}, Total: {nr}")
    

# Exercício 3.6

def kmeans(X, n_clusters, max_iters=300):

    # Inicializar os centróides aleatoriamente, escolhendo n_clusters pontos de dados aleatoriamente

    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    for _ in range(max_iters):

        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])
        if np.all(centroids == new_centroids):
            print("ESTABILIZOU")
            break
        centroids = new_centroids

    print("Centróides finais:")
    print(centroids)

    # Separar os pontos de dados para cada centróide considerando a distância

    clusters = [X[labels == k] for k in range(n_clusters)]

    return centroids, clusters, labels

def kmeansVisualization(data):

    sensors_types = ['acceleration', 'gyroscope', 'magnetometer']
    activities_data = []

    if data is None:
        return
    
    for sensor_type in sensors_types:
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
        
        activities_data.append(activity_data)

    for activity in sorted(activity_data.keys()):
        data_acceleration = np.array(activities_data[0][activity])  #usar os dados do acelerómetro para k-means
        data_gyroscope = np.array(activities_data[1][activity])
        data_magnetometer = np.array(activities_data[2][activity])

        # Montar pontos de dados 3D

        X = np.vstack((data_acceleration, data_gyroscope, data_magnetometer)).T

        centroids, clusters, labels = kmeans(X, n_clusters=4)

        print("Centroides Finais: ")
        print(centroids)
        print(clusters)
        print(labels)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['blue', 'green', 'orange', 'purple']

        for i, cluster in enumerate(clusters):
            color = colors [i % len(colors)]
            ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], c=color, marker='.', label=f'Cluster {i}')

        ax.set_xlabel('Acceleration Module')
        ax.set_ylabel('Gyroscope Module')
        ax.set_zlabel('Magnetometer Module')
        ax.set_title(f'K-Means Clustering - {activities.get(activity, f"Activity {activity}")}')
        ax.legend()
        plt.show()
        break


def main():

    #2.
    #"""
    dataset = loadData(None)
    
    if not dataset:
        print("Nenhum dado carregado!")
        return
    else:
        for key, values in dataset.items():
            print(f"Key: {key}, Values: {values}")
            break
    
    #"""
    
    #3.1
    print("\n=== Acelerómetro ===")
    #loadBoxPlotActivityAndVariable(dataset, 'acceleration')

    print("\n=== Giroscópio ===")
    #loadBoxPlotActivityAndVariable(dataset, 'gyroscope')

    print("\n=== Magnetómetro ===")
    #loadBoxPlotActivityAndVariable(dataset, 'magnetometer')
    
    #3.2 - Análise de densidade de outliers
    print("\n=== Outliers Acelerómetro ===")
    #outlierDensity('acceleration')

    print("\n=== Outliers Giroscópio ===")
    #outlierDensity('gyroscope')
    
    print("\n=== Outliers Magnetómetro ===")
    #outlierDensity('magnetometer')


    """
    K = (3, 3.5, 4)
    for i in K:
        print(f"\n=== Detecção de Outliers Acelerómetro Para K = {i}===")
        outliers(dataset, 'acceleration', i)
        break
    """
    #3.6 - K-Means
    
    kmeansVisualization(dataset)
    

if __name__ == "__main__":
    main()