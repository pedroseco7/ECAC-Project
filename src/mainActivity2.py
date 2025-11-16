import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random
import torch

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
    2D scatter plot dos dados originais e sintéticos
    Apenas as duas primeiras características são usadas para visualização
    0: feature 0
    1: feature 1
    11: activity label
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

    # Plotar uma vez por atividade
    for activity, rows in activity_sets.items():
        if len(rows) == 0:
            continue
        np.array(rows)
        arr = np.vstack(rows)
        plt.scatter(arr[:, 0], arr[:, 1], c=colors[activity - 1], marker='o', s=8, alpha=0.5, label=f'Atividade {activity}')

    # Plotar pontos sintéticos por atividade
    synth = np.array(synthetic_samples)
    for activity in range(1, 8):
        rows = synth[synth[:, 11] == activity]
        if rows.shape[0] == 0:
            continue
        plt.scatter(rows[:, 0], rows[:, 1], c=highlighted_color_for_synthetic, marker='x', s=30, alpha=1, label=f'Sintética Atividade {activity}')

    plt.title('Visualização de Samples Originais e Sintéticas')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
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

    FS_IN_HZ = 51.5 #frequencia original (está no ficheiro embeddings_extractor.py)

    for key, data in dataset.items():

        #segmentar os dados, apenas as colunas xyz do acc (o acc_segmentation ja faz isso)
        original_segments, activities = acc_segmentation(data)

        if not original_segments:
            continue    

        #reamostrar cada segmento para 30Hz e 5s
        for seg, act in zip(original_segments, activities):
            acc_resampled, fs_target = resample_to_30hz_5s(seg, FS_IN_HZ)
            all_resampled_segments.append(acc_resampled)
            all_activities.append(act)
        
    #converter para array numpy
    x_all = np.array(all_resampled_segments)
    y_all = np.array(all_activities)

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

    EMBEDDINGS_DATASET = np.hstack((embeddings_final, y_all_reshaped)) #(N_SEGMENTS, N_EMBEDDINGS + 1)
    print("[DEBUG]:", EMBEDDINGS_DATASET.shape)
    np.save('embeddings_dataset.npy', EMBEDDINGS_DATASET)

def main():

    dataset = loadData(None)
    
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


if __name__ == "__main__":
    main()