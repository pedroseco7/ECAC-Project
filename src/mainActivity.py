import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis, iqr, entropy
from scipy.fft import fft
from scipy.stats import kstest, ttest_ind, kruskal 

from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


from ReliefF import ReliefF

import pandas as pd

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
    
    
    #ordenar atividades para garantir consistência
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

        centroids, clusters, labels = kmeans(X, n_clusters=3)

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


#3.7.1
def dbscan(data, epslon, min_samples):

    #normalizar os dados
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    #aplicar o dbscan
    dbscan = DBSCAN(eps=epslon, min_samples=min_samples)
    labels = dbscan.fit_predict(data_scaled) #cluster labels para cada ponto no dataset

    existe_outlier = 0

    if -1 in labels:
        existe_outlier = 1

    n_clusters = len(set(labels)) - existe_outlier
    n_noise = list(labels).count(-1)

    return labels, n_clusters, n_noise

def dbscanVisualization(data):

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
    
    #testar diferentes valores do epslon e do min_samples para ver o que dá melhor
    valores_epslon = [0.3,0.5, 1.0]
    valores_min_samples = [5,10,15]


    for activity in sorted(activity_data.keys()):
        data_acceleration = np.array(activities_data[0][activity]) 
        data_gyroscope = np.array(activities_data[1][activity])
        data_magnetometer = np.array(activities_data[2][activity])

        X = np.vstack((data_acceleration, data_gyroscope, data_magnetometer)).T
        
        # REDUZIR MEMÓRIA: Se houver muitos dados, fazer amostragem
        max_samples = 30000  # Limitar a 30k pontos para evitar MemoryError
        if X.shape[0] > max_samples:
            print(f" Atividade {activities.get(activity, f'Activity {activity}')} tem {X.shape[0]} pontos. Reduzindo para {max_samples}...")
            indices = np.random.choice(X.shape[0], max_samples, replace=False)
            X = X[indices]

        #procurar a melhor configuração do dbscan, testar com vários valores de epslon e min_samples
        best_eps = 0
        best_min_samples = 0
        best_n_clusters = 0
        
        print(f"\n🔍 Testando configurações DBSCAN para atividade: {activities.get(activity, f'Activity {activity}')} ({X.shape[0]} pontos)")

        for eps in valores_epslon:
            for min_samples in valores_min_samples:
                print(f"   Testando eps={eps}, min_samples={min_samples}...", end=" ")
                labels, n_clusters, n_noise = dbscan(X, eps, min_samples)
                print(f"✓ clusters={n_clusters}, noise={n_noise}")

                if n_clusters > best_n_clusters and n_clusters <= 4:
                    best_n_clusters = n_clusters
                    best_eps = eps
                    best_min_samples = min_samples
        
        #aplicar a melhor configuração do dbscan
        labels_dbscan, n_clusters_dbscan, n_noise_dbscan = dbscan(X, best_eps, best_min_samples)

        print(f"Atividade: {activities.get(activity, f'Activity {activity}')}, Epslon: {best_eps}, Min Samples: {best_min_samples}, Clusters: {n_clusters_dbscan}, Outliers: {n_noise_dbscan}")

        #visualizar os resultados do DBSCAN
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        unique_labels = set(labels_dbscan)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Pontos de ruído em preto
                class_member_mask = (labels_dbscan == label)
                xy = X[class_member_mask]
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], 
                            c='black', marker='x', s=20, alpha=0.6, label='Noise')
            else:
                class_member_mask = (labels_dbscan == label)
                xy = X[class_member_mask]
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], 
                            c=[color], marker='o', s=20, alpha=0.8, label=f'Cluster {label}')

        
        ax.set_xlabel('Acceleration')
        ax.set_ylabel('Gyroscope')
        ax.set_zlabel('Magnetometer')
        ax.set_title(f'DBSCAN Clustering - {activities.get(activity, f"Activity {activity}")}')
        ax.legend()
        plt.show()

        


'''
TODO:
4.1

1. Calcular médias por atividade
2. Testar normalidade (Kolmogorov-Smirnov)
3. Se NORMAL → usar Student's t-test
   Se NÃO NORMAL → usar Kruskal-Wallis
4. Interpretar p-values
'''


def estatisticalSignificance(data):

    sensors_types = ['acceleration', 'gyroscope', 'magnetometer']

    if data is None:
        return
    
    for sensor_type in sensors_types:
        # Definir colunas para cada sensor
        print("Sensor:", sensor_type)
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
        
        #1. calcular a média por atividade
        means_by_activity = {}

        for activity in sorted(activity_data.keys()):  # FIX: era activities_data.keys()
            data_array = np.array(activity_data[activity])

            #guardar a info das atividades numa struct
            means_by_activity[activity] = {
                'mean': np.mean(data_array),
                'std': np.std(data_array),
                'n': len(data_array),
                'data': data_array
            }

        print("1. Médias por atividade:")
        for activity, stats in means_by_activity.items():
            print(f"   {activities.get(activity, f'Activity {activity}'):45} | Média: {stats['mean']:.4f} | Std: {stats['std']:.4f} | N: {stats['n']}")

        #2. testar a normalidade com o kstest
        print("\n2. Teste de Normalidade (Kolmogorov-Smirnov):")
        print("   (p-value > 0.05 = Normal, p-value ≤ 0.05 = Não Normal)\n")
        
        normal_activities = []
        non_normal_activities = []

        for activity, stats in means_by_activity.items():
            data_array = stats['data']

            # Testar normalidade (KS test)
            ks_stat, ks_pvalue = kstest(data_array, 'norm', args=(stats['mean'], stats['std']))
            
            is_normal = ks_pvalue > 0.05
            
            if is_normal:
                normal_activities.append(activity)
                status = "✓ NORMAL"
            else:
                non_normal_activities.append(activity)
                status = "✗ NÃO NORMAL"
            
            print(f"   {activities.get(activity, f'Activity {activity}'):45} | KS: {ks_stat:.4f} | p-value: {ks_pvalue:.4f} | {status}")

        #3. Testes estatísticos (comparação entre atividades)
        print("\n3. Testes de Significância Estatística:")
        
        if len(normal_activities) >= 2:
            print(f"\n   📈 T-TEST (Student's t-test) para atividades NORMAIS ({len(normal_activities)} atividades):")
            # Comparar pares de atividades normais
            for i, act1 in enumerate(normal_activities):
                for act2 in normal_activities[i+1:]:
                    data1 = means_by_activity[act1]['data']
                    data2 = means_by_activity[act2]['data']
                    
                    t_stat, p_value = ttest_ind(data1, data2)
                    
                    # Interpretar p-value
                    if p_value < 0.01:
                        interpretation = "*** ALTAMENTE SIGNIFICATIVO"
                    elif p_value < 0.05:
                        interpretation = "** SIGNIFICATIVO"
                    elif p_value < 0.1:
                        interpretation = "* MARGINALMENTE SIGNIFICATIVO"
                    else:
                        interpretation = "NÃO SIGNIFICATIVO"
                    
                    act1_name = activities.get(act1, f'Activity {act1}')
                    act2_name = activities.get(act2, f'Activity {act2}')
                    print(f"      {act1_name} vs {act2_name}")
                    print(f"         t-stat: {t_stat:.4f} | p-value: {p_value:.4f} | {interpretation}")
        
        if len(non_normal_activities) >= 2:
            print(f"\n   📉 KRUSKAL-WALLIS TEST para atividades NÃO NORMAIS ({len(non_normal_activities)} atividades):")
            
            # Kruskal-Wallis requer pelo menos 2 grupos
            groups = [means_by_activity[act]['data'] for act in non_normal_activities]
            
            if len(groups) >= 2:
                h_stat, p_value = kruskal(*groups)
                
                # Interpretar p-value
                if p_value < 0.01:
                    interpretation = "*** ALTAMENTE SIGNIFICATIVO (há diferenças entre os grupos)"
                elif p_value < 0.05:
                    interpretation = "** SIGNIFICATIVO (há diferenças entre os grupos)"
                elif p_value < 0.1:
                    interpretation = "* MARGINALMENTE SIGNIFICATIVO"
                else:
                    interpretation = "NÃO SIGNIFICATIVO (grupos são similares)"
                
                print(f"      Atividades: {[activities.get(act, f'Activity {act}') for act in non_normal_activities]}")
                print(f"      H-statistic: {h_stat:.4f} | p-value: {p_value:.4f}")
                print(f"      Resultado: {interpretation}")
        
        print("\n" + "="*60)







#4.2 
#Funções de Features Temporais (Por Eixo) do artigo
'''
Features Temporais:

Mean (média)
Median (mediana)
Standard deviation (desvio padrão)
Variance (variância)
Root Mean Square (RMS)
Averaged derivatives (média das derivadas de primeira ordem)
Skewness (assimetria da distribuição)
Kurtosis (curtose da distribuição)
Interquartile Range (amplitude interquartílica)
Zero Crossing Rate (taxa de cruzamento por zero)
Mean Crossing Rate (taxa de cruzamento pela média)
Pairwise Correlation (correlação entre eixos dos sensores)
Spectral Entropy (entropia espectral) — transição entre o domínio temporal e espectral.
'''

'''
Features Espectrais:
AI - Mean of Movement Intensity
VI - Variance of Movement Intensity
SMA - Signal Magnitude Area normalizada
EVA - Eigenvalues of Dominant Directions (2 maiores autovalores)
CAGH - Correlation between Acceleration along Gravity and Heading Directions
AVH - Averaged Velocity along Heading Direction
AVG - Averaged Velocity along Gravity Direction
ARATG - Averaged Rotation Angles related to Gravity Direction
DF - Dominant Frequency (frequência dominante no espectro FFT)
ENERGY - Soma dos quadrados das magnitudes FFT normalizada pela janela
AAE - Averaged Acceleration Energy (média das energias das 3 componentes de aceleração)
ARE - Averaged Rotation Energy (média das energias das 3 componentes do giroscópio)
'''

SAMPLING_RATE = 50  # Hz
WINDOW_SECS = 5
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_SECS)
OVERLAP_RATE = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_RATE))

def segment_data(data, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Segmenta os dados em janelas deslizantes e descarta janelas que contenham mais do que uma atividade
    """

    segments = []
    labels = []

    for i in range(0, len(data) - window_size + 1, step_size):
        
        window = data[i:i + window_size]

        activity_labels_in_window = window[:,11]

        #verificar se todas as labels na janela são iguais
        if np.all(activity_labels_in_window == activity_labels_in_window[0]):
            segments.append(window)
            labels.append(activity_labels_in_window[0])

    return segments, labels

#funcoes de features temporais
def f_mean(data):
    """Média."""
    return np.mean(data)

def f_median(data):
    """Mediana."""
    return np.median(data)

def f_std(data):
    """Desvio Padrão."""
    return np.std(data)

def f_var(data):
    """Variância."""
    return np.var(data)

def f_rms(data):
    """Root Mean Square (RMS)."""
    return np.sqrt(np.mean(data**2))

def f_avg_deriv(data):
    """Média das derivadas de primeira ordem."""
    return np.mean(np.diff(data))

def f_skew(data):
    """Skewness (Assimetria)."""
    return skew(data)

def f_kurt(data):
    """Kurtosis (Curtose)."""
    return kurtosis(data)

def f_iqr(data):
    """Interquartile Range (Amplitude Interquartílica)."""
    return iqr(data)

def f_zcr(data):
    """Zero Crossing Rate."""
    # Conta o número de vezes que o sinal cruza o zero
    return np.sum(np.diff(np.signbit(data)) != 0)

def f_mcr(data):
    """Mean Crossing Rate."""
    # Conta o número de vezes que o sinal cruza a sua própria média
    mean_centered_data = data - np.mean(data)
    return f_zcr(mean_centered_data)

def f_spectral_entropy(data):
    """Spectral Entropy."""
    fft_vals = np.abs(fft(data))
    fft_vals = fft_vals[:len(fft_vals) // 2] # Apenas frequências positivas
    power_spectrum = fft_vals**2
    
    # Evitar divisão por zero se a janela for silenciosa (energia 0)
    if np.sum(power_spectrum) == 0:
        return 0.0
        
    ps_normalized = power_spectrum / np.sum(power_spectrum)
    return entropy(ps_normalized)

# --- Funções de Features Temporais (Combinadas / Eixos Múltiplos) ---

def f_corr(data1, data2):
    """Pairwise Correlation (Correlação entre dois eixos)."""
    # np.corrcoef pode retornar NaN se um eixo for constante (ex: std=0)
    corr_matrix = np.corrcoef(data1, data2)
    if np.isnan(corr_matrix).any():
        return 0.0 # Retorna 0 se a correlação não puder ser calculada
    return corr_matrix[0, 1]

def f_sma(data_x, data_y, data_z):
    """Signal Magnitude Area (normalizada pela janela)."""
    # Nota: Esta é uma feature temporal, apesar de estar na lista de espectrais do artigo.
    # É a média da soma das magnitudes absolutas dos 3 eixos.
    sma = np.mean(np.abs(data_x) + np.abs(data_y) + np.abs(data_z))
    return sma

def f_eva(data_x, data_y, data_z):
    """Eigenvalues of Dominant Directions (2 maiores autovalores)."""
    # Combina os 3 eixos numa matriz (N_amostras, 3)
    data_3d = np.vstack((data_x, data_y, data_z)).T
    # Calcula a matriz de covariância
    cov_matrix = np.cov(data_3d, rowvar=False)
    # Calcula os autovalores (usamos 'h' para matriz simétrica/Hermitiana)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues.sort() # Ordena do menor para o maior
    # Retorna os 2 maiores
    return eigenvalues[-1], eigenvalues[-2] 

# --- Funções de Features Espectrais (Base) ---

def f_dominant_freq(data, fs):
    """Dominant Frequency (Frequência Dominante)."""
    n = len(data)
    if n < 2:
        return 0.0
        
    fft_vals = np.abs(fft(data))[:n // 2]
    fft_freqs = np.fft.fftfreq(n, d=1/fs)[:n // 2]
    
    # Ignora a componente DC (índice 0) para encontrar a frequência dominante
    # (a componente DC é apenas o 'offset' ou média do sinal)
    dominant_idx = np.argmax(fft_vals[1:]) + 1 # +1 para compensar o slicing [1:]
    return fft_freqs[dominant_idx]

def f_energy(data):
    """ENERGY (Soma dos quadrados das magnitudes FFT normalizada)."""
    fft_vals = np.abs(fft(data))
    # Energia de Parseval, normalizada pelo tamanho da janela (N)
    return np.sum(fft_vals**2) / len(data)


#função de extração

def extract_features(data, window_size, step_size, fs):
    """
    Função principal que aplica o janelamento e extrai o vetor de features
    para cada segmento válido.
    Retorna um array NumPy e uma lista com os nomes das features.
    """

    # 1. Segmentar os dados
    print(f"A segmentar os dados... (Janela: {window_size}, Passo: {step_size})")
    segments, labels = segment_data(data, window_size, step_size)
    print(f"Segmentação concluída. {len(segments)} segmentos válidos encontrados.")
    
    if not segments:
        print("Nenhum segmento válido encontrado.")
        # Retorna um array vazio e uma lista de nomes vazia
        return np.array([]), []
    
    feature_list_of_dicts = [] # Lista de dicionários, cada dict é uma linha (janela)

    axis_indices = {
        'acc_x': 1, 'acc_y': 2, 'acc_z': 3,
        'gyro_x': 4, 'gyro_y': 5, 'gyro_z': 6,
        'mag_x': 7, 'mag_y': 8, 'mag_z': 9,
    }
    sensors = ['acc', 'gyro', 'mag']
    axes = ['x', 'y', 'z']

    for i, window in enumerate(segments):
        features = {} # Dicionário para esta janela
        window_data = {} # Armazena os dados brutos dos eixos desta janela
        
        # 1. Armazena os 9 eixos de dados
        for name, col_idx in axis_indices.items():
            window_data[name] = window[:, col_idx]
        
        # 2. Calcular Features Temporais (por eixo)
        for key, data_axis in window_data.items():
            features[f'{key}_mean'] = f_mean(data_axis)
            features[f'{key}_median'] = f_median(data_axis)
            features[f'{key}_std'] = f_std(data_axis)
            features[f'{key}_var'] = f_var(data_axis)
            features[f'{key}_rms'] = f_rms(data_axis)
            features[f'{key}_avg_deriv'] = f_avg_deriv(data_axis)
            features[f'{key}_skew'] = f_skew(data_axis)
            features[f'{key}_kurt'] = f_kurt(data_axis)
            features[f'{key}_iqr'] = f_iqr(data_axis)
            features[f'{key}_zcr'] = f_zcr(data_axis)
            features[f'{key}_mcr'] = f_mcr(data_axis)
            features[f'{key}_spectral_entropy'] = f_spectral_entropy(data_axis)
            
        # 3. Calcular Features Espectrais (por eixo)
        for key, data_axis in window_data.items():
            features[f'{key}_dom_freq'] = f_dominant_freq(data_axis, fs)
            features[f'{key}_energy'] = f_energy(data_axis)

        # 4. Calcular Features Combinadas (entre eixos)
        for sensor in sensors:
            x, y, z = window_data[f'{sensor}_x'], window_data[f'{sensor}_y'], window_data[f'{sensor}_z']
            
            # Correlações
            features[f'{sensor}_corr_xy'] = f_corr(x, y)
            features[f'{sensor}_corr_xz'] = f_corr(x, z)
            features[f'{sensor}_corr_yz'] = f_corr(y, z)
            
            # SMA (Signal Magnitude Area)
            features[f'{sensor}_sma'] = f_sma(x, y, z)
            
            # EVA (Eigenvalues)
            eva1, eva2 = f_eva(x, y, z)
            features[f'{sensor}_eva1'] = eva1
            features[f'{sensor}_eva2'] = eva2
        
        # 5. Calcular Features Espectrais Agregadas (AAE, ARE)
        # AAE - Averaged Acceleration Energy
        e_ax = features['acc_x_energy']
        e_ay = features['acc_y_energy']
        e_az = features['acc_z_energy']
        features['aae'] = np.mean([e_ax, e_ay, e_az])

        # ARE - Averaged Rotation Energy (Giroscópio)
        e_gx = features['gyro_x_energy']
        e_gy = features['gyro_y_energy']
        e_gz = features['gyro_z_energy']
        features['are'] = np.mean([e_gx, e_gy, e_gz])
        
        # 6. Adicionar a label da atividade
        features['activity_label'] = int(labels[i])
        
        # 7. Adicionar o dicionário de features à lista
        feature_list_of_dicts.append(features)

    #Conversão para NumPy Array
    
    feature_names = list(feature_list_of_dicts[0].keys())
    
    # 2. Criar uma lista de listas (linhas de dados)
    data_rows = []
    for feature_dict in feature_list_of_dicts:
        # Adiciona os valores na ordem correta
        data_rows.append([feature_dict[name] for name in feature_names])
    
    # 3. Converter para NumPy array
    np_features = np.array(data_rows, dtype=np.float64)
    
    # Retornar o array NumPy e os nomes das features
    return np_features, feature_names

def perform_pca(features, n_components=0.75):

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)

    print(f"Explained variance ratio by PCA components: {pca.explained_variance_ratio_}")
    print(f"Total explained variance by selected components: {np.sum(pca.explained_variance_ratio_):.4f}")

    return features_pca, pca, scaler

# https://medium.com/@yashdagli98/feature-selection-using-relief-algorithms-with-python-example-3c2006e18f83

def perform_reliefF(X, y, n_neighbors=100, n_features_to_select=10):
    fs = ReliefF(n_neighbors=n_neighbors, n_features_to_keep=n_features_to_select)
    X_train = fs.fit_transform(X, y)

    feature_scores = fs.feature_scores
    return X_train, feature_scores, fs

def perform_Fisher_score(X, y, n_features_to_select=10):

    selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
    X_new = selector.fittransform(X, y)

    scores = selector.scores
    top_features_idx = selector.get_support(indices=True)
    return X_new, scores, top_features_idx 



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
    
    #kmeansVisualization(dataset)

    #3.7.1
    #dbscanVisualization(dataset)
    
    #4.1 - Significância Estatística
    #statisticalSignificance(dataset)

    #4.2 - Extração de Features
    
    all_features = []
    all_feature_names = None
    
    # Processar todos os ficheiros
    total_files = len(dataset)
    processed_count = 0
    
    for key, data in dataset.items():
        processed_count += 1
        print(f"[{processed_count}/{total_files}] Processando {key}...")
        
        if data is not None and len(data) > 0:
            np_features, feature_names = extract_features(data, WINDOW_SIZE, STEP_SIZE, SAMPLING_RATE)
            
            if np_features.size > 0:
                print(f" {np_features.shape[0]} segmentos extraídos")
                all_features.append(np_features)
                
                if all_feature_names is None:
                    all_feature_names = feature_names
    
    # Concatenar tudo
    if all_features:
        np_features = np.vstack(all_features)
        feature_names = all_feature_names
        
        print(f"\nTotal de segmentos: {np_features.shape[0]}")
        print(f"Total de features: {np_features.shape[1]}")
        
        # Contagem por atividade
        label_col_index = feature_names.index('activity_label')
        activity_labels_vector = np_features[:, label_col_index]
        unique_labels, counts = np.unique(activity_labels_vector, return_counts=True)
        
        print("\nDistribuição por atividade:")
        for label, count in zip(unique_labels, counts):
            activity_name = activities.get(int(label), f'Activity {int(label)}')
            print(f"  {activity_name:45} | {count:5d} segmentos")
        
        # PCA
        print("\n=== PCA ===")
        feature_data = np.delete(np_features, label_col_index, axis=1)
        features_pca, pca_model, scaler_model = perform_pca(feature_data, n_components=0.75)
        print(f"Dimensões após PCA: {features_pca.shape}")

        # Agora vamos aplicar o PCA para um único ponto do dataset, para ver como
        # muda as dimensões

        print("\n --- Análise de um Ponto Individual com PCA --- ")

        instant_point = feature_data[0]
        print("Len do Instant Point antes do PCA:")
        print(len(instant_point))
        print("10 primeiros valores do Instant Point antes do PCA:")
        print(instant_point[:10])
        
        single_point_pca, pca_model_useless, scaler_model_useless = perform_pca(instant_point.reshape(1, -1), n_components=0.75)
        
        print("Len do Instant Point depois do PCA:")
        print(len(single_point_pca))
        print("10 primeiros valores do Instant Point depois do PCA:")
        print(single_point_pca[:10])
        
        '''
        # ReliefF
        print("\n=== ReliefF ===")
        X = np.delete(np_features, label_col_index, axis=1)
        y = np_features[:, label_col_index].astype(int)
        
        X_train_reliefF, features_scores, fs_model = perform_reliefF(X, y, n_neighbors=100, n_features_to_select=10)
        
        top_features_idx = np.argsort(features_scores)[::-1][:10]
        print("\nTop 10 Features (ReliefF):")
        for i, idx in enumerate(top_features_idx):
            print(f"{i+1:2d}. {feature_names[idx]} - Score: {features_scores[idx]:.5f}")
        
        # Fisher Score
        print("\n=== Fisher Score ===")
        X_train_fisher, fisher_scores, top_idx_fisher = perform_Fisher_score(X, y, n_features_to_select=10)
        
        print("\nTop 10 Features (Fisher):")
        for i, idx in enumerate(top_idx_fisher):
            print(f"{i+1:2d}. {feature_names[idx]} - Score: {fisher_scores[idx]:.5f}")
        
    else:
        print("\nNenhuma feature extraída!")

    # Realizar a seleção de features usando o algoritmo ReliefF

    '''

if __name__ == "__main__":
    main()