import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import re
import random
import torch
import pickle

from scipy.stats import skew, kurtosis, iqr, entropy
from scipy.fft import fft
from scipy.stats import kstest, ttest_ind, kruskal 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import re

from ReliefF import ReliefF

import pandas as pd

from scipy.stats import wilcoxon

class our_KNN_Classifier:
    def __init__(self,k,distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self,X_train,Y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(Y_train)
        return self
    
    def predict(self,X_test):
        #prever as labels de multiplas amostras
        X_test = np.array(X_test)
        predictions = []
        for x in X_test:
            # (x - self.X_train) subtrai x a TODAS as linhas de treino de uma vez
            if self.distance_metric == 'euclidean':
                distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            elif self.distance_metric == 'manhattan':
                distances = np.sum(np.abs(self.X_train - x), axis=1)
            
            else:
                raise ValueError("Métrica desconhecida.")
            
            #obter os índices dos k vizinhos mais próximos
            k_nearest_indices = np.argsort(distances)[:self.k]
            
            #buscar as labels correspondentes
            k_nearest_labels = self.y_train[k_nearest_indices]

            #votar na label mais comum
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            most_common_idx = np.argmax(counts)
            predictions.append(unique_labels[most_common_idx])

        return np.array(predictions)
    

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

    
SAMPLING_RATE = 50  # Hz
WINDOW_SECS = 5
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_SECS)
OVERLAP_RATE = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_RATE))

def f_spectral_entropy(data):
    # (Igual ao treino)
    fft_vals = np.abs(fft(data))
    fft_vals = fft_vals[:len(fft_vals) // 2]
    power_spectrum = fft_vals**2
    if np.sum(power_spectrum) == 0: return 0.0
    ps_normalized = power_spectrum / np.sum(power_spectrum)
    return entropy(ps_normalized)

def extract_manual_features_single(segment, fs=50):
    """
    Extrai features de um único segmento (256, 9).
    GARANTIA DE ORDEM: Segue a mesma sequência do extract_features original.
    """
    feature_vector = []
    
    # 1. Mapeamento das colunas (Igual ao treino)
    # 0,1,2=Acc | 3,4,5=Gyro | 6,7,8=Mag
    axis_indices = {
        'acc_x': 0, 'acc_y': 1, 'acc_z': 2,
        'gyro_x': 3, 'gyro_y': 4, 'gyro_z': 5,
        'mag_x': 6, 'mag_y': 7, 'mag_z': 8,
    }
    sensors = ['acc', 'gyro', 'mag']

    window_data = {} 

    for ax_name, col_idx in axis_indices.items():
        data_axis = segment[:, col_idx]
        window_data[ax_name] = data_axis
        
        feature_vector.append(np.mean(data_axis))                   
        feature_vector.append(np.median(data_axis))                 
        feature_vector.append(np.std(data_axis))                    
        feature_vector.append(np.var(data_axis))                    
        feature_vector.append(np.sqrt(np.mean(data_axis**2)))       
        feature_vector.append(np.mean(np.diff(data_axis)))          
        feature_vector.append(skew(data_axis))                      
        feature_vector.append(kurtosis(data_axis))                  
        feature_vector.append(iqr(data_axis))                       
        
        
        feature_vector.append(np.sum(np.diff(np.signbit(data_axis)) != 0)) 
        
        feature_vector.append(np.sum(np.diff(np.signbit(data_axis - np.mean(data_axis))) != 0)) 
        
        feature_vector.append(f_spectral_entropy(data_axis))
        
        fft_vals = np.abs(fft(data_axis))[:len(data_axis)//2]
        fft_freqs = np.fft.fftfreq(len(data_axis), d=1/fs)[:len(data_axis)//2]
        
        if len(fft_vals) > 1:
            dom_idx = np.argmax(fft_vals[1:]) + 1
            dom_freq = fft_freqs[dom_idx]
        else:
            dom_freq = 0.0
            
        feature_vector.append(dom_freq)
        feature_vector.append(np.sum(fft_vals**2) / len(data_axis)) 


    energies_acc = []
    energies_gyro = []

    for sensor in sensors:
        x = window_data[f'{sensor}_x']
        y = window_data[f'{sensor}_y']
        z = window_data[f'{sensor}_z']
        
        # Correlações
        feature_vector.append(np.corrcoef(x, y)[0, 1] if np.std(x)>0 and np.std(y)>0 else 0)
        feature_vector.append(np.corrcoef(x, z)[0, 1] if np.std(x)>0 and np.std(z)>0 else 0)
        feature_vector.append(np.corrcoef(y, z)[0, 1] if np.std(y)>0 and np.std(z)>0 else 0)
        
        # SMA
        feature_vector.append(np.mean(np.abs(x) + np.abs(y) + np.abs(z)))
        
        # EVA
        data_3d = np.vstack((x, y, z)).T
        cov_matrix = np.cov(data_3d, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues.sort()
        feature_vector.append(eigenvalues[-1])
        feature_vector.append(eigenvalues[-2])
        
        e_x = np.sum(np.abs(fft(x))[:len(x)//2]**2)/len(x)
        e_y = np.sum(np.abs(fft(y))[:len(y)//2]**2)/len(y)
        e_z = np.sum(np.abs(fft(z))[:len(z)//2]**2)/len(z)
        
        if sensor == 'acc': energies_acc = [e_x, e_y, e_z]
        if sensor == 'gyro': energies_gyro = [e_x, e_y, e_z]

    feature_vector.append(np.mean(energies_acc)) 
    feature_vector.append(np.mean(energies_gyro)) 
    
    return np.array(feature_vector).reshape(1, -1)

#importar funções do embeddings_extractor.py
#============================================
def load_model():
  ''' Loads the model from the github repo and obtains just the feature encoder. '''

  repo = 'OxWearables/ssl-wearables'
  # class_num não interessa para extrair features; mas o hub pede este arg
  model = torch.hub.load(repo, 'harnet5', class_num=5, pretrained=True)
  model.eval()

  # Passo crucial: ficar só com a parte auto-supervisionada
  # O README diz que há um 'feature_extractor' (pré-treinado) e um 'classifier' (não treinado). :contentReference[oaicite:14]{index=14}
  feature_encoder = model.feature_extractor
  feature_encoder.to("cpu")
  feature_encoder.eval()

  return feature_encoder

def get_embedding_single(segment, fs_in_hz=50.0):
    """
    Versão DEPLOY da embedding_features.
    Recebe (256, 9) -> Devolve (1, 512).
    """
    repo = 'OxWearables/ssl-wearables'
    model = torch.hub.load(repo, 'harnet5', class_num=5, pretrained=True)
    encoder = model.feature_extractor
    encoder.to("cpu")
    encoder.eval()

    acc_data = segment[:, 0:3] # Só Acc
    
    # Resample 50Hz -> 30Hz
    t_in = np.arange(acc_data.shape[0]) / fs_in_hz
    t_out = np.arange(0, 5.0, 1.0/30.0)
    
    acc_resampled = np.zeros((len(t_out), 3), dtype=np.float32)
    for axis in range(3):
        acc_resampled[:, axis] = np.interp(t_out, t_in, acc_data[:, axis])
        
    x_tensor = torch.from_numpy(acc_resampled.T).float().unsqueeze(0)
    
    with torch.no_grad():
        embedding = encoder(x_tensor).numpy()
        
    return embedding

def predict_activity_from_segment(segment, pipeline_path='best_model_pipeline.pkl'):
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Ficheiro {pipeline_path} não encontrado.")
        
    with open(pipeline_path, 'rb') as f:
        model_pkg = pickle.load(f)
        
    # 2. USAR AS FUNÇÕES SINGLE (CORREÇÃO)
    if model_pkg['data_type'] == 'manual':
        X = extract_manual_features_single(segment) # <-- USA A NOVA
    else:
        X = get_embedding_single(segment) # <-- USA A NOVA
        
    # 3. Normalização e Redução (Igual)
    X = model_pkg['scaler'].transform(X)
    
    if model_pkg['reducer'] is not None:
        try:
            X = model_pkg['reducer'].transform(X)
        except AttributeError:
            if hasattr(model_pkg['reducer'], 'top_features'):
                cols = model_pkg['reducer'].top_features[:15]
                X = X[:, cols]
                
    pred = model_pkg['knn_model'].predict(X)
    return int(pred[0]), model_pkg['model_name']

def main():
    print("--- INICIANDO DEPLOYMENT SIMULADO ---")
    
    # 1. Obter Dados Reais
    full_dataset = loadData(None)
    
    # Vamos usar o primeiro ficheiro disponível para teste
    key = list(full_dataset.keys())[0]
    data = full_dataset[key]
    
    # 2. Filtrar Atividades (1 a 7 apenas)
    print(f"Dados brutos: {data.shape}")
    data = data[data[:, 11] <= 7]
    print(f"Dados filtrados (Atividades 1-7): {data.shape}")
    
    if len(data) < 256:
        print("Erro: Não há dados suficientes.")
        exit()
        
    # 3. Selecionar um Segmento Aleatório
    # Colunas 1 a 10 (exclusivo) = índices 1..9 (Acc, Gyro, Mag)
    start = np.random.randint(0, len(data) - 256)
    segment_raw = data[start : start + 256, 1:10] # Shape (256, 9)
    true_label = int(data[start, 11])
    
    print(f"\nSegmento Selecionado: Linhas {start}-{start+256}")
    
    # 4. EXECUTAR O MODELO
    try:
        predicted_label, model_name = predict_activity_from_segment(segment_raw)
        
        # Mapa para display
        acts = {
            1: 'STAND', 2: 'SIT', 3: 'SIT AND TALK', 4: 'WALK', 5: 'WALK AND TALK',
            6: 'CLIMB STAIR', 7: 'CLIMB STAIR AND TALK'
        }
        
        print("\n" + "="*40)
        print(f"MODELO USADO: {model_name}")
        print("="*40)
        print(f"PREVISÃO:   {predicted_label} -> {acts.get(predicted_label, 'Unknown')}")
        print(f"VERDADEIRO: {true_label} -> {acts.get(true_label, 'Unknown')}")
        
        if predicted_label == true_label:
            print("\nRESULTADO:  SUCESSO")
        else:
            print("\nRESULTADO:  ERRO")
            
    except Exception as e:
        print(f"\n[ERRO] Falha no deployment: {e}")


if __name__ == "__main__":
    main()