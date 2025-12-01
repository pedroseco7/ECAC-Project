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
        
        # print(f"      k={k}: Val Acc = {score:.4f}") # (Opcional: print detalhado)
        
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
    return y_test, y_pred_test, best_k

def perform_hypothesis_testing(results_dict):
    """
    Compara o melhor modelo contra todos os outros usando Wilcoxon Signed-Rank Test.
    results_dict: { 'ModelName': [acc_run1, acc_run2, ..., acc_runN] }

    Escolhi este teste estatistico porque assume que os dados nao seguem uma distribuicao normal, ao contrario do t-test. Usa dados emparelhados (mesmas runs para cada modelo) e testa se as medianas das diferencas sao significativamente diferentes de zero.
    """
    print("   ANÁLISE ESTATÍSTICA (WILCOXON SIGNED-RANK TEST)")

    # 1. Calcular médias para encontrar o "Melhor Modelo"
    means = {k: np.mean(v) for k, v in results_dict.items()}
    best_model_name = max(means, key=means.get)
    best_model_scores = results_dict[best_model_name]
    
    print(f"MELHOR MODELO (Média): {best_model_name} (Acc: {means[best_model_name]:.4f})")
    print(f"{'Comparison Model':<40} | {'p-value':<12} | {'Significant?':<10}")

    stats_results = []

    # 2. Comparar o melhor contra os restantes
    alpha = 0.05
    for model_name, scores in results_dict.items():
        if model_name == best_model_name:
            continue
        
        # Teste de Wilcoxon (alternative='greater' testa se o Best é > Other)
        # Nota: Se as accuracies forem idênticas em todas as runs, o teste falha (zero diff).
        try:
            stat, p_value = wilcoxon(best_model_scores, scores, alternative='greater')
            
            is_significant = p_value < alpha
            sig_str = "YES" if is_significant else "NO"
            
            print(f"{model_name:<40} | {p_value:.6f}     | {sig_str}")
            
            stats_results.append({
                'Model': model_name,
                'p-value': p_value,
                'Significant Difference': is_significant
            })
            
        except ValueError:
            # Acontece se todos os valores forem exatamente iguais
            print(f"{model_name:<40} | N/A (Equal)  | NO")

    return best_model_name, pd.DataFrame(stats_results)

def main():
    # --- 1. CARREGAR DADOS ---
    print("--- 1. CARREGAR DADOS ---")
    features_dataset = None
    # Tenta carregar o pickle das features manuais
    with open('features.pkl', 'rb') as f:
        data = pickle.load(f)
        features_dataset = data['features'] if isinstance(data, dict) else data

    # Tenta carregar ou gerar os embeddings
    dataset_raw = loadData(None)
    if os.path.exists('embeddings_dataset.npy'):
        embedding_dataset = np.load('embeddings_dataset.npy')
    else:
        embedding_dataset = embedding_features(dataset_raw)
    dataset_raw = None # Limpar memória

    # Lista para guardar o resumo final (para a tabela bonita no fim)
    final_results = []

    # --- 2. LOOP PRINCIPAL (ITERAR POR ESTRATÉGIA DE SPLIT) ---
    split_strategies = ['random', 'subject']
    
    for split_type in split_strategies:
        print(f"\n{'#'*80}")
        print(f"   ESTRATÉGIA DE SPLIT: {split_type.upper()}")
        print(f"{'#'*80}")
        
        # 2.1 Realizar os Splits (Treino, Validação, Teste)
        (f_X_train, f_y_train), (f_X_val, f_y_val), (f_X_test, f_y_test) = perform_splits(features_dataset, split_type)
        (e_X_train, e_y_train), (e_X_val, e_y_val), (e_X_test, e_y_test) = perform_splits(embedding_dataset, split_type)

        # 2.2 Pré-Processamento Base: STANDARD SCALER (Obrigatório para KNN/PCA/ReliefF)
        scaler_f = StandardScaler()
        f_Xt_sc = scaler_f.fit_transform(f_X_train)
        f_Xv_sc = scaler_f.transform(f_X_val)
        f_Xtest_sc = scaler_f.transform(f_X_test)

        scaler_e = StandardScaler()
        e_Xt_sc = scaler_e.fit_transform(e_X_train)
        e_Xv_sc = scaler_e.transform(e_X_val)
        e_Xtest_sc = scaler_e.transform(e_X_test)

        # 2.3 Preparar as Variantes de Dados (Experiments)
        experiments = {}

        # ==========================================
        # GRUPO A: MANUAL FEATURES
        # ==========================================
        # 1. Todas as Features (Normal)
        experiments[f"[{split_type.upper()}] Manual: All"] = (f_Xt_sc, f_y_train, f_Xv_sc, f_y_val, f_Xtest_sc, f_y_test)

        # 2. PCA (90% Variância)
        pca = PCA(n_components=0.90)
        f_Xt_pca = pca.fit_transform(f_Xt_sc)
        f_Xv_pca = pca.transform(f_Xv_sc)
        f_Xtest_pca = pca.transform(f_Xtest_sc)
        experiments[f"[{split_type.upper()}] Manual: PCA (90%)"] = (f_Xt_pca, f_y_train, f_Xv_pca, f_y_val, f_Xtest_pca, f_y_test)

        # 3. ReliefF (Top 15 Features)
        print(f"   > Calculando ReliefF (Manual Features)...")
        n_feats = 15
        fs = ReliefF(n_neighbors=100, n_features_to_keep=n_feats)
        
        # Fit Transform no Treino
        f_Xt_sel = fs.fit_transform(f_Xt_sc, f_y_train)
        
        # Aplicar a Validação e Teste (com fallback se .transform falhar)
        try:
            f_Xv_sel = fs.transform(f_Xv_sc)
            f_Xtest_sel = fs.transform(f_Xtest_sc)
        except AttributeError:
            if hasattr(fs, 'top_features'):
                cols = fs.top_features[:n_feats]
                f_Xv_sel = f_Xv_sc[:, cols]
                f_Xtest_sel = f_Xtest_sc[:, cols]
            else:
                f_Xv_sel = f_Xv_sc[:, :n_feats]
                f_Xtest_sel = f_Xtest_sc[:, :n_feats]

        experiments[f"[{split_type.upper()}] Manual: ReliefF ({n_feats})"] = (f_Xt_sel, f_y_train, f_Xv_sel, f_y_val, f_Xtest_sel, f_y_test)


        # ==========================================
        # GRUPO B: EMBEDDINGS
        # ==========================================
        # 1. Todos os Embeddings (Normal)
        experiments[f"[{split_type.upper()}] Embed: All"] = (e_Xt_sc, e_y_train, e_Xv_sc, e_y_val, e_Xtest_sc, e_y_test)
        
        # 2. PCA Embeddings (90%)
        pca_emb = PCA(n_components=0.90)
        e_Xt_pca = pca_emb.fit_transform(e_Xt_sc)
        e_Xv_pca = pca_emb.transform(e_Xv_sc)
        e_Xtest_pca = pca_emb.transform(e_Xtest_sc)
        experiments[f"[{split_type.upper()}] Embed: PCA (90%)"] = (e_Xt_pca, e_y_train, e_Xv_pca, e_y_val, e_Xtest_pca, e_y_test)

        # 3. ReliefF Embeddings (Top 15)
        print(f"   > Calculando ReliefF (Embeddings)...")
        fs_emb = ReliefF(n_neighbors=100, n_features_to_keep=n_feats)
        e_Xt_sel = fs_emb.fit_transform(e_Xt_sc, e_y_train)
        
        try:
            e_Xv_sel = fs_emb.transform(e_Xv_sc)
            e_Xtest_sel = fs_emb.transform(e_Xtest_sc)
        except AttributeError:
            if hasattr(fs_emb, 'top_features'):
                cols = fs_emb.top_features[:n_feats]
                e_Xv_sel = e_Xv_sc[:, cols]
                e_Xtest_sel = e_Xtest_sc[:, cols]
            else:
                e_Xv_sel = e_Xv_sc[:, :n_feats]
                e_Xtest_sel = e_Xtest_sc[:, :n_feats]

        experiments[f"[{split_type.upper()}] Embed: ReliefF ({n_feats})"] = (e_Xt_sel, e_y_train, e_Xv_sel, e_y_val, e_Xtest_sel, e_y_test)


        # --- 3. EXECUÇÃO DOS EXPERIMENTOS ---
        for exp_name, data_pack in experiments.items():
            X_tr, y_tr, X_v, y_v, X_te, y_te = data_pack
            
            # 3.1 Tuning (Encontrar k usando Train+Val e prever no Test)
            y_true_final, y_pred_final, best_k = hyperparameter_tuning_and_eval(
                X_tr, y_tr, X_v, y_v, X_te, y_te, exp_name
            )
            
            # 3.2 Relatório Detalhado (Matriz de Confusão, Precision, Recall, etc.)
            print(f"\n>>> RELATÓRIO DETALHADO: {exp_name} (k={best_k})")
            
            # Chamada direta sem mapa de atividades (usa IDs genéricos)
            calculate_classification_metrics(y_true_final, y_pred_final)
            
            # 3.3 Guardar dados para o Resumo Final
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(y_true_final, y_pred_final)
            f1 = f1_score(y_true_final, y_pred_final, average='macro')
            
            final_results.append({
                'Experiência': exp_name,
                'Melhor k': best_k,
                'Accuracy': acc,
                'Macro F1': f1,
                'Num Features': X_tr.shape[1]
            })

    # --- 4. RESUMO FINAL COMPARATIVO ---
    print("\n\n" + "="*100)
    print("RESUMO FINAL COMPARATIVO (Ordenado por Accuracy)")
    print("="*100)
    
    df_results = pd.DataFrame(final_results)
    df_results = df_results.sort_values(by='Accuracy', ascending=False)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(df_results)
    print("="*100)
    
    # Opcional: Guardar em CSV
    df_results.to_csv("resultados_finais_knn.csv", index=False)
    print("Tabela guardada em 'resultados_finais_knn.csv'")
    

if __name__ == "__main__":
    main()