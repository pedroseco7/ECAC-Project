import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis, iqr, entropy

from scipy.stats import kstest, ttest_ind, kruskal 

from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler

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
        max_samples = 70000  # Limitar a 70k pontos para evitar MemoryError
        if X.shape[0] > max_samples:
            print(f"⚠️  Atividade {activities.get(activity, f'Activity {activity}')} tem {X.shape[0]} pontos. Reduzindo para {max_samples}...")
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

def calc_mean(window_axis):
    return np.mean(window_axis)

def calc_median(window_axis):
    return np.median(window_axis)

def calc_std(window_axis):
    return np.std(window_axis)

def calc_variance(window_axis):
    return np.var(window_axis)

def calc_rms(window_axis):
    """Calcula o Root Mean Square (RMS)"""
    return np.sqrt(np.mean(window_axis**2))

def calc_avg_derivative(window_axis):
    """Calcula a média das derivadas de primeira ordem (Averaged derivatives)"""
    return np.mean(np.diff(window_axis))

def calc_skewness(window_axis):
    """Calcula a assimetria (Skewness)"""
    return skew(window_axis)

def calc_kurtosis(window_axis):
    """Calcula a curtose (Kurtosis)"""
    return kurtosis(window_axis)

def calc_iqr(window_axis):
    """Calcula o Intervalo Interquartil (Interquartile Range)"""
    return iqr(window_axis)

def calc_zcr(window_axis):
    """Calcula a Taxa de Cruzamento de Zero (Zero Crossing Rate)"""
    T = len(window_axis)
    crossings = np.sum((window_axis[:-1] * window_axis[1:]) < 0)
    return crossings / T

def calc_mcr(window_axis):
    """Calcula a Taxa de Cruzamento da Média (Mean Crossing Rate)"""
    T = len(window_axis)
    mean = np.mean(window_axis)
    centered_signal = window_axis - mean
    crossings = np.sum((centered_signal[:-1] * centered_signal[1:]) < 0)
    return crossings / T


#Funções de Features Temporais (Multi-Eixo) do artigo

def calc_pairwise_correlation(window_multi):
    """Calcula a correlação entre pares de eixos (Pairwise Correlation)"""
    # window_multi é N_samples x 3_eixos
    corr_matrix = np.corrcoef(window_multi, rowvar=False)
    # Retorna as correlações do triângulo superior (XY, XZ, YZ)
    return corr_matrix[0, 1], corr_matrix[0, 2], corr_matrix[1, 2]

def calc_ai_vi(window_accel):
    """Calcula a Média (AI) e Variância (VI) da Intensidade do Movimento (Movement Intensity)"""
    # Nota: O artigo remove a gravidade, mas não especifica como.
    # Esta é uma implementação da magnitude total.
    mi = np.sqrt(np.sum(window_accel**2, axis=1))
    ai = np.mean(mi) 
    vi = np.var(mi)
    return ai, vi

def calc_sma(window_accel):
    """Calcula a Área de Magnitude do Sinal Normalizada (Normalized Signal Magnitude Area)"""
    # Fórmula: (1/T) * (sum(|ax|) + sum(|ay|) + sum(|az|))
    # Isto é matematicamente equivalente a sum(mean(abs(eixos)))
    T = len(window_accel)
    sum_abs_axes = np.sum(np.abs(window_accel), axis=0)
    sma = np.sum(sum_abs_axes) / T
    return sma

def calc_eva(window_accel):
    """Calcula os dois maiores valores próprios (Eigenvalues) da matriz de covariância"""
    cov_matrix = np.cov(window_accel, rowvar=False)
    # Usar eigvalsh para matrizes simétricas (como a covariância)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues.sort() # Ordena do menor para o maior
    return eigenvalues[-1], eigenvalues[-2] # Retorna os dois maiores


# Funções de Features Espectrais (Por Eixo) do artigo

def calc_energy(window_axis, fs):
    """Calcula a Energia (Energy) do sinal """
    T = len(window_axis)
    fft_vals = np.abs(np.fft.rfft(window_axis))
    fft_vals = fft_vals[1:] # Exclui a componente DC (índice 0) 
    energy = np.sum(fft_vals**2) / T
    return energy

def calc_dominant_frequency(window_axis, fs):
    """Calcula a Frequência Dominante (Dominant Frequency)"""
    T = len(window_axis)
    fft_vals = np.abs(np.fft.rfft(window_axis))**2 
    fft_freq = np.fft.rfftfreq(T, d=1.0/fs)
    dominant_idx = np.argmax(fft_vals)
    return fft_freq[dominant_idx]

def calc_spectral_entropy(window_axis, fs):
    """Calcula a Entropia Espectral (Spectral Entropy)"""
    fft_vals = np.abs(np.fft.rfft(window_axis))
    psd = fft_vals**2 # Power Spectral Density
    # Normaliza a PSD para que a soma seja 1 (necessário para entropia)
    psd_norm = psd / np.sum(psd)
    # Usa a função de entropia da scipy (base 2)
    return entropy(psd_norm, base=2)

# Funções de Features Espectrais (Multi-Eixo) do artigo

def calc_aae(window_accel, fs):
    """Calcula a Energia Média do Acelerómetro (Averaged Acceleration Energy)"""
    energy_x = calc_energy(window_accel[:, 0], fs)
    energy_y = calc_energy(window_accel[:, 1], fs)
    energy_z = calc_energy(window_accel[:, 2], fs)
    return np.mean([energy_x, energy_y, energy_z])

def calc_are(window_gyro, fs):
    """Calcula a Energia Média de Rotação (Averaged Rotation Energy)"""
    energy_x = calc_energy(window_gyro[:, 0], fs)
    energy_y = calc_energy(window_gyro[:, 1], fs)
    energy_z = calc_energy(window_gyro[:, 2], fs)
    return np.mean([energy_x, energy_y, energy_z])



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
    estatisticalSignificance(dataset)

if __name__ == "__main__":
    main()