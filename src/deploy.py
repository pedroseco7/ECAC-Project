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

import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Ignorar especificamente o aviso de versões incompatíveis
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

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
    É importante que respeite a ordem de dois loops do treino.
    """
    feature_vector = []
    
    # 9 colunas: Acc(0-2), Gyro(3-5), Mag(6-8)
    # Índices: 0..8
    
    # --- LOOP 1: Features Temporais (Média ... Spectral Entropy) ---
    # No treino, este loop corre para todos os eixos primeiro.
    for col_idx in range(9):
        d = segment[:, col_idx]
        feature_vector.extend([
            np.mean(d), np.median(d), np.std(d), np.var(d), 
            np.sqrt(np.mean(d**2)), np.mean(np.diff(d)), 
            skew(d), kurtosis(d), iqr(d),
            np.sum(np.diff(np.signbit(d)) != 0),
            np.sum(np.diff(np.signbit(d - np.mean(d))) != 0),
            f_spectral_entropy(d)
        ])

    # --- LOOP 2: Features Espectrais (Dom Freq, Energy) ---
    # No treino, este loop corre para todos os eixos DEPOIS do primeiro loop acabar.
    for col_idx in range(9):
        d = segment[:, col_idx]
        fft_vals = np.abs(fft(d))[:len(d)//2]
        fft_freqs = np.fft.fftfreq(len(d), d=1/fs)[:len(d)//2]
        
        if len(fft_vals) > 1:
            dom_idx = np.argmax(fft_vals[1:]) + 1
            dom_freq = fft_freqs[dom_idx]
        else:
            dom_freq = 0.0
            
        feature_vector.append(dom_freq)
        feature_vector.append(np.sum(fft_vals**2) / len(d)) # Energy

    # --- LOOP 3: Features Combinadas (Corr, SMA, EVA) ---
    energies = {'acc': [], 'gyro': []}
    sensors = ['acc', 'gyro', 'mag']
    
    for i, sensor in enumerate(sensors):
        start = i * 3
        x, y, z = segment[:, start], segment[:, start+1], segment[:, start+2]
        
        # Correlações
        feature_vector.append(np.corrcoef(x, y)[0, 1] if np.std(x)>0 and np.std(y)>0 else 0)
        feature_vector.append(np.corrcoef(x, z)[0, 1] if np.std(x)>0 and np.std(z)>0 else 0)
        feature_vector.append(np.corrcoef(y, z)[0, 1] if np.std(y)>0 and np.std(z)>0 else 0)
        
        # SMA
        feature_vector.append(np.mean(np.abs(x) + np.abs(y) + np.abs(z)))
        
        # EVA
        evals = np.linalg.eigvalsh(np.cov(np.vstack((x, y, z)).T, rowvar=False))
        feature_vector.extend([evals[-1], evals[-2]])
        
        # Guardar energia para passo 4 (Recalcular para garantir)
        if sensor in energies:
            energies[sensor] = [np.sum(np.abs(fft(axis))[:len(axis)//2]**2)/len(axis) for axis in [x,y,z]]

    # --- PASSO 4: Agregadas (AAE, ARE) ---
    feature_vector.append(np.mean(energies['acc']))
    feature_vector.append(np.mean(energies['gyro']))

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

def predict_activity_from_segment(segment, pipeline_path='./src/best_model_pipeline.pkl'):
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Ficheiro {pipeline_path} não encontrado.")
        
    with open(pipeline_path, 'rb') as f:
        model_pkg = pickle.load(f)
    
    # 2. USAR AS FUNÇÕES SINGLE
    if model_pkg['data_type'] == 'manual':
        X = extract_manual_features_single(segment) 
    else:
        X = get_embedding_single(segment)
        
    # 3. Normalização e Redução
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
        
    # ... (depois de carregar dados) ...

    print(f"\nA testar 1000 segmentos aleatórios...")
    correct = 0
    total = 1000
    
    for i in range(total):
        start = np.random.randint(0, len(data) - 256)
        segment_raw = data[start : start + 256, 1:10]
        true_label = int(data[start, 11])
        
        try:
            pred_label, _ = predict_activity_from_segment(segment_raw)
            if pred_label == true_label:
                correct += 1
                print(f"Seg {i+1}: ✅ ({true_label})")
            else:
                print(f"Seg {i+1}: ❌ Real: {true_label} vs Pred: {pred_label}")
        except:
            pass
            
    print(f"\nAccuracy no Teste Rápido: {correct/total*100:.1f}%")


if __name__ == "__main__":
    main()