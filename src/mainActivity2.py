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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import re

from ReliefF import ReliefF

import pandas as pd

from scipy.stats import wilcoxon

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

    return activity_counts
    

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

def perform_splits(dataset, method, current_seed):
    """
    Vamos dividir o dataset em Treino (60%), Validação (20%) e Teste (20%)
    """

    X = dataset[:, :-2]
    y = dataset[:, -2]
    groups = dataset[:, -1]

    if method == 'random':
        print("A aplicar Random Split (Within-Subject)...")
        
        # Listas para acumular os pedaços de cada participante
        X_train_list, y_train_list = [], []
        X_val_list, y_val_list = [], []
        X_test_list, y_test_list = [], []

        # Identificar todos os sujeitos únicos (ex: 1, 2, 3...)
        unique_subjects = np.unique(groups)

        for sub_id in unique_subjects:
            # 1. Isolar os dados APENAS deste sujeito
            mask = (groups == sub_id)
            X_sub = X[mask]
            y_sub = y[mask]

            # 2. Fazer o split 60/20/20 para ESTE sujeito específico
            try:
                # Passo A: Tirar 60% para Treino (sobram 40% temporários)
                # Usamos stratify=y_sub para manter a proporção de atividades DENTRO deste sujeito
                X_tr, X_temp, y_tr, y_temp = train_test_split(
                    X_sub, y_sub, 
                    test_size=0.4, 
                    random_state=current_seed, 
                    stratify=y_sub
                )

                # Passo B: Dividir o temporário (40%) em Validação (20%) e Teste (20%)
                X_val, X_te, y_val, y_te = train_test_split(
                    X_temp, y_temp, 
                    test_size=0.5, 
                    random_state=current_seed, 
                    stratify=y_temp
                )

                # 3. Guardar os pedaços nas listas
                X_train_list.append(X_tr)
                y_train_list.append(y_tr)
                X_val_list.append(X_val)
                y_val_list.append(y_val)
                X_test_list.append(X_te)
                y_test_list.append(y_te)

            except ValueError:
                # Fallback: Se o sujeito tiver tão poucas amostras de uma classe que o stratify falha
                # fazemos split sem stratify para não crashar o código
                # print(f"Aviso: Stratify falhou para Subject {sub_id}, a fazer aleatório simples.")
                X_tr, X_temp, y_tr, y_temp = train_test_split(X_sub, y_sub, test_size=0.4, random_state=current_seed)
                X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.5, random_state=current_seed)
                
                X_train_list.append(X_tr); y_train_list.append(y_tr)
                X_val_list.append(X_val); y_val_list.append(y_val)
                X_test_list.append(X_te); y_test_list.append(y_te)

        # 4. Consolidar todos os pedaços num único array gigante
        return (np.vstack(X_train_list), np.concatenate(y_train_list)), \
               (np.vstack(X_val_list), np.concatenate(y_val_list)), \
               (np.vstack(X_test_list), np.concatenate(y_test_list))
    
    elif method == 'subject':
        print("A aplicar Subject Split...")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.6, random_state=current_seed)
        train_indices, test_indices = next(splitter.split(X, y, groups))
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test, groups_test = X[test_indices], y[test_indices], groups[test_indices]

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=current_seed)
        val_idx, test_idx = next(splitter.split(X_test, y_test, groups_test))

        X_val, Y_val = X_test[val_idx], y_test[val_idx]
        X_test, Y_test = X_test[test_idx], y_test[test_idx]

        return (X_train, y_train), (X_val, Y_val), (X_test, Y_test)

def perform_pca(x_train, x_val, n_components=0.75):

    scaler = StandardScaler()
    features_x_scaled = scaler.fit_transform(x_train)
    features_val_scaled = scaler.transform(x_val)

    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_x_scaled)
    features_val_pca = pca.transform(features_val_scaled)

    print(f"Explained variance ratio by PCA components: {pca.explained_variance_ratio_}")
    print(f"Total explained variance by selected components: {np.sum(pca.explained_variance_ratio_):.4f}")

    return features_pca, features_val_pca

def perform_reliefF(x_train, y_train, x_val, y_val, n_neighbors=100, n_features_to_select=15):
    fs = ReliefF(n_neighbors=n_neighbors, n_features_to_keep=n_features_to_select)
    X_train = fs.fit_transform(x_train, y_train)
    X_val = fs.transform(x_val, y_val)

    feature_scores = fs.feature_scores
    return X_train, X_val, feature_scores
        
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
        
def calculate_classification_metrics(y_true, y_pred, activity_names=None):

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    error_rate = 1.0 - accuracy
    
    print(f"\n{'='*40}")
    print(f"RESUMO GLOBAL")
    print(f"{'='*40}")
    print(f"Accuracy Global:    {accuracy:.4f}")
    print(f"Error Rate Global:  {error_rate:.4f}")

    # 3. Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de Confusão:")
    print(cm)

    #Métricas por Classe
    #TP, TN, FP, FN para cada classe
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if activity_names is None:
        activity_names = [f"Class {c}" for c in classes]
    
    metrics_per_class = []

    #precisão, recall, f1 via sklearn para validação
    precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    # Calcular TP, TN, FP, FN
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Conversão para evitar divisão por zero
    with np.errstate(divide='ignore', invalid='ignore'):
        TPR = TP / (TP + FN) 
        TNR = TN / (TN + FP) 
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN) 
        Precision = TP / (TP + FP)
        
    # Limpeza de NaNs
    TPR = np.nan_to_num(TPR)
    FPR = np.nan_to_num(FPR)
    Precision = np.nan_to_num(Precision)
    
    # Construir DataFrame
    df_metrics = pd.DataFrame({
        'Activity': activity_names,
        'Precision': Precision,
        'Recall (TPR)': TPR,
        'F1-Score': f1s,
        'FPR (False Pos Rate)': FPR,
        'Support': supports
    })

    print(f"\n{'='*40}")
    print(f"DETALHE POR CLASSE")
    print(f"{'='*40}")
    print(df_metrics.round(4).to_string(index=False))

    #Médias Globais (Macro Average)
    # A média "Global" real de métricas por classe costuma ser a Macro Average (média simples das classes)
    # Ou Weighted Average (ponderada pelo suporte). 
    # Vou apresentar a Macro para veres o desempenho médio "por atividade".
    
    print(f"\n{'='*40}")
    print(f"MÉDIAS GLOBAIS (MACRO)")
    print(f"{'='*40}")
    print(f"Global Precision (Macro): {np.mean(Precision):.4f}")
    print(f"Global Recall/TPR (Macro):{np.mean(TPR):.4f}")
    print(f"Global FPR (Macro):       {np.mean(FPR):.4f}")
    print(f"Global F1-Score (Macro):  {np.mean(f1s):.4f}")
    
    return cm, df_metrics

# Exercício 5

def hyperparameter_tuning_and_eval(X_train, y_train, X_val, y_val, X_test, y_test, dataset_name):
    """
    Executa o exercício 5.1:
    1. Testa vários k usando Treino e Validação.
    2. Escolhe o melhor k.
    3. Junta Treino+Validação, retreina e avalia no Teste.
    """
    
    k_values = [3, 5, 7, 9, 11] # Valores a testar (CONFIRMAR SE SÃO APENAS ESTES)
    best_k = -1
    best_val_score = -1
    
    print(f"\n   >>> Tuning para: {dataset_name}")
    
    # --- FASE 1: Encontrar o melhor k (Train vs Val) ---
    for k in k_values:
        # Podes usar o 'our_KNN_Classifier' ou 'KNeighborsClassifier' do sklearn
        # Usar sklearn é geralmente mais rápido para loops de tuning
        knn = our_KNN_Classifier(k=k, distance_metric='euclidean')
        knn.fit(X_train, y_train)
        score = knn.score(X_val, y_val)
        
        if score > best_val_score:
            best_val_score = score
            best_k = k
            
    print(f"      Melhor k encontrado: {best_k} (Val Acc: {best_val_score:.4f})")
    
    # --- FASE 2: Retreino (Train + Val) e Avaliação Final (Test) ---
    
    # Concatenar Treino e Validação
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))
    
    # Treinar o modelo final com o melhor k
    final_knn = our_KNN_Classifier(k=best_k, distance_metric='euclidean')
    final_knn.fit(X_combined, y_combined)
    
    # Avaliar no Teste (O momento da verdade!)
    # Aqui vamos prever e usar a tua função de métricas detalhadas
    y_pred_test = final_knn.predict(X_test)
    
    # Retornar as previsões e o target real para usares no calculate_classification_metrics
    return y_test, y_pred_test, best_k, final_knn

def perform_hypothesis_testing(results_dict):
    """
    Compara o melhor modelo contra todos os outros usando Wilcoxon Signed-Rank Test.
    """
    print(f"\n{'#'*80}")
    print("   ANÁLISE ESTATÍSTICA (WILCOXON SIGNED-RANK TEST)")
    print(f"{'#'*80}")

    # 1. Calcular médias
    means = {k: np.mean(v) for k, v in results_dict.items()}
    
    # Ordenar por melhor performance
    sorted_models = sorted(means.items(), key=lambda x: x[1], reverse=True)
    best_model_name = sorted_models[0][0]
    best_model_scores = results_dict[best_model_name]
    
    print(f"MELHOR MODELO (Média): {best_model_name} (Acc: {means[best_model_name]:.4f})")
    print("-" * 100)
    print(f"{'Comparison Model':<40} | {'Mean Acc':<10} | {'p-value':<12} | {'Significant?':<10}")
    print("-" * 100)

    stats_results = []
    alpha = 0.05

    for model_name, acc_mean in sorted_models:
        if model_name == best_model_name:
            continue
        
        scores = results_dict[model_name]
        
        try:
            stat, p_value = wilcoxon(best_model_scores, scores, alternative='greater')
            is_significant = p_value < alpha
            sig_str = "YES" if is_significant else "NO "
            
            print(f"{model_name:<40} | {acc_mean:.4f}     | {p_value:.6f}     | {sig_str}")
            
            stats_results.append({
                'Model': model_name,
                'Mean Accuracy': acc_mean,
                'p-value': p_value,
                'Significant Difference': is_significant
            })
            
        except ValueError:
            print(f"{model_name:<40} | {acc_mean:.4f}     | N/A (Equal)  | NO")

    return best_model_name, pd.DataFrame(stats_results)

def main():

    activities_map = {
        1: 'STAND', 
        2: 'SIT', 
        3: 'SIT AND TALK', 
        4: 'WALK', 
        5: 'WALK AND TALK', 
        6: 'CLIMB STAIR (UP/DOWN)', 
        7: 'CLIMB STAIR (UP/DOWN) AND TALK'
    }

    # --- 1. CARREGAR DADOS ---
    print("--- 1. CARREGAR DADOS ---")
    features_dataset = None
    with open('features.pkl', 'rb') as f:
        data = pickle.load(f)

        if isinstance(data, dict):
            features_dataset = data['features']
            feature_names = data['names']
            print("Features e Nomes carregados com sucesso.")
        else:
            features_dataset = data
            # Não existe nomes, temos de gerar valores aleatórios
            # Gera nomes feat_0, feat_1... se não existirem
            feature_names = [f"feat_{i}" for i in range(features_dataset.shape[1])]

    dataset = loadData(None)
    activity_counts = analyze_data(dataset) 
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
    #============================================================
    if os.path.exists('embeddings_dataset.npy'):
        embedding_dataset = np.load('embeddings_dataset.npy')
    else:
        embedding_dataset = embedding_features(dataset)

    print(f"Shape original Features: {features_dataset.shape}")
    print(f"Shape original Embeddings: {embedding_dataset.shape}")

    # 1. Filtrar Features Manuais
    # Assumindo: [Features... | Label | SubjectID] -> Label é a coluna -2
    mask_f = features_dataset[:, -2] <= 7
    features_dataset = features_dataset[mask_f]

    # 2. Filtrar Embeddings (MUITO IMPORTANTE)
    # Tens de fazer o mesmo aqui, senão vais comparar 7 classes contra 11 classes!
    mask_e = embedding_dataset[:, -2] <= 7
    embedding_dataset = embedding_dataset[mask_e]

    print(f"\n[INFO] Dados filtrados (Apenas Classes 1-7)")
    print(f" -> Features novas: {features_dataset.shape}")
    print(f" -> Embeddings novos: {embedding_dataset.shape}")

    unique, counts = np.unique(features_dataset[:, -2], return_counts=True)
    print("\n[ANÁLISE DE CLASSES]")
    for cls, count in zip(unique, counts):
        print(f"   Actividade {int(cls)}: {count} amostras ({count/sum(counts)*100:.1f}%)")


    # --- ESTRUTURA PARA GUARDAR RESULTADOS DAS 5 RUNS ---
    results_storage = {} 
    NUM_RUNS = 5
    N_FEATS_RELIEF = 15

    global_best_info = {
        'accuracy': 0.0,
        'seed': None,
        'model_name': None,
        'best_k': None,
        'split_type': None
    }

    best_deploy_package = None
    global_best_acc = 0.0

    # --- 2. LOOP DE REPETIÇÃO (ROBUSTEZ ESTATÍSTICA) ---
    for run_idx in range(NUM_RUNS):
        current_seed = np.random.randint(0, 10000)
        print(f"\n{'='*80}")
        print(f"   >>> RUN {run_idx + 1}/{NUM_RUNS} (Seed: {current_seed}) <<<")
        print(f"{'='*80}")

        split_strategies = ['random', 'subject']
        
        for split_type in split_strategies:
            print(f"\n   --- ESTRATÉGIA: {split_type.upper()} ---")
            
            # 2.1 Splits
            (f_X_train, f_y_train), (f_X_val, f_y_val), (f_X_test, f_y_test) = perform_splits(features_dataset, split_type, current_seed)
            (e_X_train, e_y_train), (e_X_val, e_y_val), (e_X_test, e_y_test) = perform_splits(embedding_dataset, split_type, current_seed)

            print("Aplicando o SMOTE para equilibrar as classes de treino.")
            # Vamos aplicar o SMOTE às classes minoritárias
            from imblearn.over_sampling import SMOTE
            
            # k_neighbors=3 é seguro para a classe 7 que é pequena
            smote = SMOTE(random_state=current_seed, k_neighbors=3)
            
            # O SMOTE cria novas amostras sintéticas apenas para as classes minoritárias
            f_X_train, f_y_train = smote.fit_resample(f_X_train, f_y_train)

            # 2.2 Scaler
            scaler_f = StandardScaler()
            f_Xt_sc = scaler_f.fit_transform(f_X_train)
            f_Xv_sc = scaler_f.transform(f_X_val)
            f_Xtest_sc = scaler_f.transform(f_X_test)

            scaler_e = StandardScaler()
            e_Xt_sc = scaler_e.fit_transform(e_X_train)
            e_Xv_sc = scaler_e.transform(e_X_val)
            e_Xtest_sc = scaler_e.transform(e_X_test)

            # 2.3 Preparar Experiências
            experiments = {}
            # --- GRUPO A: MANUAL FEATURES ---
            experiments[f"[{split_type}] Manual: All"] = {
                'data': (f_Xt_sc, f_y_train, f_Xv_sc, f_y_val, f_Xtest_sc, f_y_test),
                'scaler': scaler_f,
                'reducer': None, # Não há redução
                'type': 'manual'
            }

            # 2. Manual PCA
            pca = PCA(n_components=0.90)
        
            f_Xt_pca = pca.fit_transform(f_Xt_sc)
            f_Xv_pca = pca.transform(f_Xv_sc)
            f_Xtest_pca = pca.transform(f_Xtest_sc)
            experiments[f"[{split_type}] Manual: PCA (90%)"] = {
                'data': (f_Xt_pca, f_y_train, f_Xv_pca, f_y_val, f_Xtest_pca, f_y_test),
                'scaler': scaler_f,
                'reducer': pca, # Guardamos o objeto PCA treinado
                'type': 'manual'
            }

            # 3. Manual ReliefF
            fs = ReliefF(n_neighbors=100, n_features_to_keep=N_FEATS_RELIEF)
            f_Xt_sel = fs.fit_transform(f_Xt_sc, f_y_train)
            cols = fs.top_features[:N_FEATS_RELIEF]
            f_Xv_sel = f_Xv_sc[:, cols]
            f_Xtest_sel = f_Xtest_sc[:, cols]

            experiments[f"[{split_type}] Manual: ReliefF ({N_FEATS_RELIEF})"] = {
                'data': (f_Xt_sel, f_y_train, f_Xv_sel, f_y_val, f_Xtest_sel, f_y_test),
                'scaler': scaler_f,
                'reducer': fs,
                'type': 'manual'
            }

            # --- GRUPO B: EMBEDDINGS ---
            # 1. All
            experiments[f"[{split_type}] Embed: All"] = {
                'data': (e_Xt_sc, e_y_train, e_Xv_sc, e_y_val, e_Xtest_sc, e_y_test),
                'scaler': scaler_e,
                'reducer': None,
                'type': 'embedding'
            }
            
            # 2. PCA (90%)
            pca_emb = PCA(n_components=0.90)
            e_Xt_pca = pca_emb.fit_transform(e_Xt_sc)
            e_Xv_pca = pca_emb.transform(e_Xv_sc)
            e_Xtest_pca = pca_emb.transform(e_Xtest_sc)
            experiments[f"[{split_type}] Embed: PCA (90%)"] = {
                'data': (e_Xt_pca, e_y_train, e_Xv_pca, e_y_val, e_Xtest_pca, e_y_test),
                'scaler': scaler_e,
                'reducer': pca_emb, # Guardamos o objeto PCA treinado
                'type': 'embedding'
            }

            # 3. ReliefF (Top 15)
            fs_emb = ReliefF(n_neighbors=100, n_features_to_keep=N_FEATS_RELIEF)
            e_Xt_sel = fs_emb.fit_transform(e_Xt_sc, e_y_train)

            cols_emb = fs_emb.top_features[:N_FEATS_RELIEF]
            
            e_Xv_sel = e_Xv_sc[:, cols_emb]
            e_Xtest_sel = e_Xtest_sc[:, cols_emb]
            
            experiments[f"[{split_type}] Embed: ReliefF ({N_FEATS_RELIEF})"] = {
                'data': (e_Xt_sel, e_y_train, e_Xv_sel, e_y_val, e_Xtest_sel, e_y_test),
                'scaler': scaler_e,
                'reducer': fs_emb,
                'type': 'embedding'
            }

            # --- 3. EXECUTAR TUNING E AVALIAÇÃO ---
            for exp_name, exp_info in experiments.items():
                X_tr, y_tr, X_v, y_v, X_te, y_te = exp_info['data']
                
                # Tuning e Retreino
                y_true_final, y_pred_final, best_k, trained_model = hyperparameter_tuning_and_eval(
                    X_tr, y_tr, X_v, y_v, X_te, y_te, exp_name
                )
                
                # APENAS NA PRIMEIRA RUN: Imprimir Relatório Detalhado
                if run_idx == 0:
                    print(f"\n>>> [DETALHES RUN 1] {exp_name} (Melhor k={best_k})")
                    unique_labels = np.unique(np.concatenate([y_true_final, y_pred_final]))
                    class_names_list = [activities_map.get(int(c), f"Class {int(c)}") for c in unique_labels]
                    
                    calculate_classification_metrics(y_true_final, y_pred_final, activity_names=class_names_list)
                    # Sem mapa de atividades: Ele usa os IDs numéricos (Class 1, Class 2...)
                    calculate_classification_metrics(y_true_final, y_pred_final, activity_names=None)
                
                # Calcular Accuracy Simples para estatística
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(y_true_final, y_pred_final)
                
                # Guardar no dicionário de resultados
                if exp_name not in results_storage:
                    results_storage[exp_name] = []
                results_storage[exp_name].append(acc)
                
                # Print curto para acompanhar progresso
                print(f"      -> {exp_name}: Acc={acc:.4f} (k={best_k})")

                # --- NOVO: VERIFICAR SE É O MELHOR RESULTADO GLOBAL ---
                if acc > global_best_acc:

                    if exp_info['type'] == 'manual':
                        # Se carregaste de um dict no início, os nomes devem estar acessíveis
                        # Assumindo que tens uma variável 'feature_names' global ou carregada
                        current_feat_names = feature_names # <--- TENS DE TER ISTO DISPONÍVEL
                    else:
                        # Para embeddings os nomes não importam tanto (é posicional), mas criamos para consistência
                        current_feat_names = [f"emb_{i}" for i in range(X_tr.shape[1])]

                    global_best_acc = acc
                    
                    # Empacotar tudo o que é preciso para o futuro
                    best_deploy_package = {
                        'model_name': exp_name,
                        'accuracy': acc,
                        'scaler': exp_info['scaler'],   # O scaler treinado nesta run
                        'reducer': exp_info['reducer'], # O PCA/ReliefF treinado nesta run
                        'knn_model': trained_model,     # O KNN treinado nesta run
                        'data_type': exp_info['type'],  # 'manual' ou 'embedding'
                        'best_k': best_k,
                        'feature_names': current_feat_names
                    }
                    print(f"      [NOVO RECORDE] {exp_name} -> {acc:.4f}")

    # --- 4. RESUMO FINAL E ESTATÍSTICA ---
    print("\n\n" + "="*100)
    print(f"RESUMO FINAL APÓS {NUM_RUNS} RUNS (Ordenado por Média)")
    print("="*100)
    
    summary_data = []
    for model_name, acc_list in results_storage.items():
        summary_data.append({
            'Modelo': model_name,
            'Mean Accuracy': np.mean(acc_list),
            'Std Dev': np.std(acc_list),
            'Max Acc': np.max(acc_list),
            'Min Acc': np.min(acc_list)
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values(by='Mean Accuracy', ascending=False)
    
    # Configuração para mostrar tudo
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    
    print(df_summary)
    print("="*100)
    
    # Guardar Tabela de Médias
    df_summary.to_csv("resultados_finais_medias.csv", index=False)
    print("Tabela de médias guardada em 'resultados_finais_medias.csv'")

    # --- 5. TESTE DE HIPÓTESES (WILCOXON) ---
    best_model, df_stats = perform_hypothesis_testing(results_storage)

    df_stats.to_csv("resultados_estatisticos.csv", index=False)
    print("\nResultados estatísticos guardados em 'resultados_estatisticos.csv'")


    # --- Dar display dos MELHORES PARÂMETROS GLOBAIS ---

    print("\n\n" + "X"*100)
    print("    A GUARDAR O MELHOR MODELO (DEPLOYMENT) ")
    print("X"*100)
    print(f"Melhor Modelo: {best_deploy_package['model_name']}")
    print(f"Accuracy:      {best_deploy_package['accuracy']:.4f}")

    with open('best_model_pipeline.pkl', 'wb') as f:
        pickle.dump(best_deploy_package, f)
    
    print("Pipeline guardada em 'best_model_pipeline.pkl'.")    


if __name__ == "__main__":
    main()