import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import re
import random
import torch
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import re

from ReliefF import ReliefF

import pandas as pd

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

def analyze_data(data):

    """Analisar o equilibrio entre os dados"""
    """Vamos ver quantas amostras temos de cada atividade"""

    """Vamos aproveitar para excluir atividades com label > 7 do dataset"""
    activity_counts = {}
    for key, values in data.items():

        filtered_rows = values[values[:, 11] <= 7]
        data[key] = filtered_rows
        for row in filtered_rows:
            activity = int(row[11]) #buscar cada label de atividade
            if activity > 7:
                continue
            if activity in activity_counts:
                activity_counts[activity] += 1
            else:
                activity_counts[activity] = 1

    return activity_counts, data
    

def smote_activity(dataset, activity_label, K):
    """Aplicar o SMOTE a uma atividade específica do dataset"""

    activity_samples = []
    for key, values in dataset.items():
        for row in values:
            if int(row[11]) == activity_label:
                activity_samples.append(row)

    if len(activity_samples) == 0:
        print("Não foram encontradas samples para a atividade especificada, abortar SMOTE.")
        return np.empty((0, 12))

    # Concatenar todas as samples da atividade num único array
    activity_samples = np.vstack(activity_samples)

    # Separar características e labels
    X = activity_samples[:, :11] 

    N = X.shape[0]

    if N < 2:
        print("Número insuficiente de samples para aplicar SMOTE.")
        return np.empty((0, 12))
    
    # Vamos definir o número de vizinhos que devem ser considerados
    # É importante considerar que deve ser pelo menos 2

    k_neighbors = min(K, N - 1)

    neighbors = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X)

    samples_sinteticas = []

    for _ in range(K):
        
        # Vamos escolher uma amostra aleatória
        idx = random.randrange(N)
        X_idx = X[idx].reshape(1, -1)

        # Encontrar os K vizinhos mais próximos

        _, indices = neighbors.kneighbors(X_idx, n_neighbors=k_neighbors + 1)

        neighbors_indices = indices[0][1:]

        # Vamos escolher um vizinho aleatório

        n_idx = random.choice(neighbors_indices)
        X_n = X[n_idx]

        # Vamos interpolar linearmente para criar uma nova amostra
        # A fórmula é X_new = X_idx + random(0,1) * (X_n - X_idx)

        diff = X_n - X[idx]
        gap = random.random()

        X_new = X[idx] + gap * diff

        sample_sintetica_com_label = np.append(X_new, activity_label)
        samples_sinteticas.append(sample_sintetica_com_label)
    
    return np.array(samples_sinteticas)

def visualize_smote(dataset, synthetic_samples):
    """
    2D scatter plot dos dados originais e sintéticos.
    1: Aceleração X
    2: Aceleração Y
    11: Activity Label
    """
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    highlighted_color_for_synthetic = 'orange'
    plt.figure(figsize=(10, 8))
    
    # Agrupar pontos originais por atividade (1..7)
    activity_sets = {a: [] for a in range(1, 8)}
    for key, values in dataset.items():
        for row in values:
            activity = int(row[11])
            if activity > 7:
                continue
            activity_sets[activity].append(row)

    # Plotar uma vez por atividade (Dados Originais)
    for activity, rows in activity_sets.items():
        if len(rows) == 0:
            continue
        arr = np.vstack(rows)

        plt.scatter(arr[:, 1], arr[:, 2], c=colors[activity - 1], marker='o', s=8, alpha=0.5, label=f'Atividade {activity}')

    # Plotar pontos sintéticos por atividade
    synth = np.array(synthetic_samples)
    
    # Verificar se existem samples sintéticas antes de tentar plotar
    if synth.ndim > 1 and synth.shape[0] > 0:
        for activity in range(1, 8):
            rows = synth[synth[:, 11] == activity]
            if rows.shape[0] == 0:
                continue
            
            plt.scatter(rows[:, 1], rows[:, 2], c=highlighted_color_for_synthetic, marker='x', s=50, linewidths=2, label=f'Sintética Atividade {activity}')

    plt.title('Visualização de Samples Originais e Sintéticas (Feature 1 vs Feature 2)')
    plt.xlabel('Eixo X do Acelerometro')
    plt.ylabel('Eixo Y do Acelerometro')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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

def acc_segmentation(data):
  ''' Estract ACC segments and their activities '''

  TIMESTAMP_COL = 10
  MIN_SEGMENT_SIZE = 20
  fs_in_hz = 51.5
  win_size = 5000
  start_time = data[0,TIMESTAMP_COL]
  end_time = start_time + win_size

  activities = []
  segments = []

  while end_time < data[-1,TIMESTAMP_COL]:
    mask = (data[:,TIMESTAMP_COL] >= start_time) & (data[:,TIMESTAMP_COL] < end_time)

    if np.sum(mask) > MIN_SEGMENT_SIZE and np.all(data[mask, -1] == data[mask, -1][0]):

      acc_xyz = data[mask,1:4]
      activity = data[mask, -1][0]
      
      activities.append(activity)
      segments.append( acc_xyz )
      

    start_time = end_time - win_size/2
    end_time = start_time + win_size
  
  
  return segments, activities

def resample_to_30hz_5s(acc_xyz, fs_in_hz):
    """
    acc_xyz: np.ndarray shape (N, 3) em m/s^2 (ou g), amostrado a fs_in_hz (float)
    devolve:
      acc_resampled: np.ndarray shape (M, 3) já a 30 Hz
      fs_target: 30.0
    """
    fs_target = 30.0
    win_size = 5 # in seconds
    t_in = np.arange(acc_xyz.shape[0]) / fs_in_hz
    t_out = np.arange(0, win_size, 1.0/fs_target)

    acc_resampled = np.zeros((len(t_out), 3), dtype=np.float32)
    for axis in range(3):
        acc_resampled[:, axis] = np.interp(t_out, t_in, acc_xyz[:, axis])

    return acc_resampled, fs_target
#====================================

def embedding_features(dataset):
    feature_encoder = load_model()
    all_resampled_segments = [] #guardar todos os segmentos
    all_activities = [] #guardar todas as atividades
    all_subjects = []

    FS_IN_HZ = 51.5 #frequencia original (está no ficheiro embeddings_extractor.py)

    for key, data in dataset.items():

        match = re.search(r'part(\d+)', key)
        if match:
            subject_id = int(match.group(1))
        else:
            continue

        #segmentar os dados, apenas as colunas xyz do acc (o acc_segmentation ja faz isso)
        original_segments, activities = acc_segmentation(data)

        if not original_segments:
            continue    

        #reamostrar cada segmento para 30Hz e 5s
        for seg, act in zip(original_segments, activities):
            acc_resampled, fs_target = resample_to_30hz_5s(seg, FS_IN_HZ)
            all_resampled_segments.append(acc_resampled)
            all_activities.append(act)
            all_subjects.append(subject_id)
        
    #converter para array numpy
    x_all = np.array(all_resampled_segments)
    y_all = np.array(all_activities)
    s_all = np.array(all_subjects)

    print("[DEBUG]:", x_all.shape) #(N_SEGMENTOS (soma dos segmentos extraídos dos participantes), N_AMOSTRAS (5s x 30Hz), N_DIMENSOES (x,y,z pedidos do enunciado))
    print("[DEBUG]:", y_all.shape) #(N_SEGMENTOS,)

    #o modelo que vamos usar espera o input com shape [N_SEGMENTOS, N_DIMENSOES, N_AMOSTRAS], foi o que foi feito no embeddings_extractor.py
    x_all_transposed = np.transpose(x_all, (0, 2, 1)) 
    print("[DEBUG]:", x_all_transposed.shape) #(N_SEGMENTOS, N_DIMENSOES, N_AMOSTRAS)


    embeddings_list = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, x_all_transposed.shape[0], batch_size):
            xb = torch.from_numpy(x_all_transposed[i:i+batch_size]).float().to("cpu")
            eb = feature_encoder(xb) #shape = (batch_size, 64)
            embeddings_list.append(eb.cpu().numpy())
    
    embeddings_3d = np.concatenate(embeddings_list, axis=0)
    print("[DEBUG]:", embeddings_3d.shape) 

    #eles pedem para que o dataset final tenha o shape [N_SEGMENTS, N_EMBEDDINGS]
    embeddings_final = np.squeeze(embeddings_3d)
    print("[DEBUG]:", embeddings_final.shape) #(N_SEGMENTS, N_EMBEDD)

    y_all_reshaped = y_all.reshape(-1, 1) #(N_SEGMENTS, 1)
    s_all_reshaped = s_all.reshape(-1, 1) #(N_SEGMENTS, 1)

    EMBEDDINGS_DATASET = np.hstack((embeddings_final, y_all_reshaped, s_all_reshaped)) #(N_SEGMENTS, N_EMBEDDINGS + 1)
    print("[DEBUG]:", EMBEDDINGS_DATASET.shape)
    np.save('embeddings_dataset.npy', EMBEDDINGS_DATASET)

    return EMBEDDINGS_DATASET


def perform_splits(dataset, method):
    """
    Vamos dividir o dataset em Treino (60%), Validação (20%) e Teste (20%)
    """

    X = dataset[:, :-2]
    y = dataset[:, -2]
    groups = dataset[:, -1]

    if method == 'random':
        print("A aplicar Random Split (Stratified)...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    elif method == 'subject':
        print("A aplicar Subject Split...")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.6, random_state=5)
        train_indices, test_indices = next(splitter.split(X, y, groups))
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test, groups_test = X[test_indices], y[test_indices], groups[test_indices]

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=5)
        val_idx, test_idx = next(splitter.split(X_test, y_test, groups_test))

        X_val, Y_val = X_test[val_idx], y_test[val_idx]
        X_test, Y_test = X_test[test_idx], y_test[test_idx]


        return (X_train, y_train), (X_val, Y_val), (X_test, Y_test)
        
def perform_pca(features, n_components=0.75):

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)

    print(f"Explained variance ratio by PCA components: {pca.explained_variance_ratio_}")
    print(f"Total explained variance by selected components: {np.sum(pca.explained_variance_ratio_):.4f}")

    return features_pca, pca, scaler

def perform_reliefF(X, y, n_neighbors=100, n_features_to_select=15):
    fs = ReliefF(n_neighbors=n_neighbors, n_features_to_keep=n_features_to_select)
    X_train = fs.fit_transform(X, y)

    feature_scores = fs.feature_scores
    return X_train, feature_scores, fs
        
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
    
    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)
        


def main():

    features_dataset = None
    with open('features.pkl', 'rb') as f:
        features_dataset = pickle.load(f)

        if isinstance(features_dataset, dict):
            features_dataset = features_dataset['features']
            print("Dicionário carregado com sucesso.")

    dataset = loadData(None)

    embedding_dataset = None
    embedding_file = 'embeddings_dataset.npy'

    if os.path.exists(embedding_file):
        embedding_dataset = np.load(embedding_file)
    else:
        embedding_dataset = embedding_features(dataset)

    
    activity_counts, dataset = analyze_data(dataset)
    print(activity_counts)
    
    # Vamos aplicar o SMOTE para gerar e visualizar 3 novas samples
    # da atividade 4, do participante 3
    # Atenção, só devem ser utilizadas as samples do participante 3 para gerar as novas samples
    atividade_alvo = 4
    K = 3
    # Filtrar o dataset para incluir apenas samples do participante 3
    dataset_participante3 = {}
    # Como as keys estão no formato "partXdevY", podemos filtrar por "part3"

    for key, values in dataset.items():
        if "part3" in key:
            dataset_participante3[key] = values

    print("Aplicando SMOTE...")

    samples_sinteticas = smote_activity(dataset_participante3, atividade_alvo, K)

    print("Samples sintéticas geradas.")
    visualize_smote(dataset_participante3, samples_sinteticas)

    #2.
    embedding_features(dataset)
    
    #3. Vamos fazer splits nos dois sets, dentro do mesmo subject e entre subjects
    #3.1 Vamos começar pelo TVT de 60%/20%/20%
    print("Divisão de Treino/Validação/Teste em 60%/20%/20% do EMBEDDING FEATURE SET")
    (e_X_train, e_Y_train), (e_X_val, e_Y_val), (e_X_test, e_Y_test) = perform_splits(embedding_dataset, 'random')
    print(f'Treino: {e_X_train.shape[0]} amostras')
    print(f'Validação: {e_X_val.shape[0]} amostras')
    print(f'Teste: {e_X_test.shape[0]} amostras')

    print("Divisão de Treino/Validação/Teste em 60%/20%/20% do FEATURES SET")
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = perform_splits(features_dataset, 'random')
    print(f'Treino: {X_train.shape[0]} amostras')
    print(f'Validação: {X_val.shape[0]} amostras')
    print(f'Teste: {X_test.shape[0]} amostras')

    print("Divisão de Treino/Validação/Teste em 60%/20%/20% do EMBEDDING FEATURE SET com Subject Split")
    (e_X_train, e_Y_train), (e_X_val, e_Y_val), (e_X_test, e_Y_test) = perform_splits(embedding_dataset, 'subject')
    print(f"Treino: {e_X_train.shape[0]} amostras")
    print(f"Validação: {e_X_val.shape[0]} amostras")
    print(f"Teste: {e_X_test.shape[0]} amostras")

    print("Divisão de Treino/Validação/Teste em 60%/20%/20% do FEATURES SET com Subject Split")
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = perform_splits(features_dataset, 'subject')
    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Validação: {X_val.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")

    print("Treino e avaliacao do KNN")

    k_values = [3,5,7,10]

    for k in k_values:
        print(f"\nKNN com k={k}")

        knn = our_KNN_Classifier(k=k, distance_metric='euclidean')
        knn.fit(X_train, Y_train)

        y_pred = knn.predict(X_val)

        accuracy = knn.score(X_val, Y_val)
        print(f"Accuracy: {accuracy:.4f}")

    print("\nComparar com sklearn KNeighborsClassifier")
    from sklearn.neighbors import KNeighborsClassifier
    
    knn_sklearn = KNeighborsClassifier(n_neighbors=5)
    knn_sklearn.fit(X_train, Y_train)
    y_pred_sklearn = knn_sklearn.predict(X_val)

    accuracy_sklearn = np.mean(y_pred_sklearn == Y_val)
    print(f"Accuracy sklearn KNeighborsClassifier: {accuracy_sklearn:.4f}")



if __name__ == "__main__":
    main()