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
import pickle
import re

from ReliefF import ReliefF

import pandas as pd


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

def main():

    feature_file = "features.pkl"
    
    np_features = None
    feature_names = None
    
    dataset = loadData(None)

    print("\n--- A Iniciar Extração de Features")

    all_features = []
    all_feature_names = None

    total_files = len(dataset)
    processed_count = 0

    for key, data in dataset.items():
        processed_count += 1

        match = re.search(r'part(\d+)', key)
        if match:
            subject_id = int(match.group(1))
        else:
            print("Não consegui extrair")
            continue

        print(f"[{processed_count}/{total_files}] Processando {key} com subject_id = {subject_id}...")

        features_temp, names_temp = extract_features(data, WINDOW_SIZE, STEP_SIZE, SAMPLING_RATE)

        if features_temp.size > 0:
            
            num_rows = features_temp.shape[0]

            subject_col = np.full((num_rows, 1), subject_id, dtype=float)
                
            # --- PASSO 3: Juntar tudo ---
            # Adiciona a coluna do ID à direita das features
            # O resultado fica: [Feature1, ..., Label, SubjectID]
            features_with_subject = np.hstack((features_temp, subject_col))
            
            all_features.append(features_with_subject)
            
            # Atualizar a lista de nomes apenas na primeira vez
            if all_feature_names is None:
                all_feature_names = names_temp + ['subject_id']

    # 3. Consolidar e Guardar
    if all_features:
        np_features = np.vstack(all_features)
        feature_names = all_feature_names
        
        print(f"\nTotal de segmentos: {np_features.shape[0]}")
        print(f"Total de features (incluindo Label e ID): {np_features.shape[1]}")
        
        # Verificar se a última coluna é realmente o ID
        print(f"Exemplo da última coluna (IDs): {np_features[:10, -1]}")
        
        print(f"\n[INFO] A guardar dados em '{feature_file}'...")
        
        # --- MUITO IMPORTANTE ---
        # Guarda como um dicionário para manteres os nomes das colunas associados aos dados
        with open(feature_file, 'wb') as f:
            pickle.dump({'features': np_features, 'names': feature_names}, f)
            
        print("[INFO] Guardado com sucesso! Podes agora fazer o Split por Subject.")
        
    else:
        print("\nNenhuma feature extraída!")

