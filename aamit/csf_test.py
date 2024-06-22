import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from math import sqrt
from sklearn.model_selection import cross_val_score
from sklearn import svm
import time

# Carregar o conjunto de dados
df = pd.read_csv('./data/bfd_data.csv')
print(df.head(3))

# Nome da coluna de rótulos
label = 'Filename'

# Lista com os nomes das características
features = df.columns.tolist()
features.remove(label)

# Transformar os rótulos para binários
df[label] = np.where(df[label].str.endswith('.png'), 1, 0)

def getMerit(subset, label):
    k = len(subset)

    # Correlação média entre característica-classe
    rcf_all = []
    for feature in subset:
        coeff = pointbiserialr(df[label], df[feature])
        rcf_all.append(abs(coeff.correlation))
    rcf = np.mean(rcf_all)

    # Correlação média entre características
    corr = df[subset].corr().values
    rff = np.nanmean(np.abs(corr[np.triu_indices_from(corr, k=1)]))

    return (k * rcf) / sqrt(k + k * (k - 1) * rff)

best_value = -1
best_feature = ''
for feature in features:
    coeff = pointbiserialr(df[label], df[feature])
    abs_coeff = abs(coeff.correlation)
    if abs_coeff > best_value:
        best_value = abs_coeff
        best_feature = feature

print("Feature %s with merit %.4f" % (best_feature, best_value))

class PriorityQueue:
    def __init__(self):
        self.queue = []

    def isEmpty(self):
        return len(self.queue) == 0

    def push(self, item, priority):
        for index, (i, p) in enumerate(self.queue):
            if set(i) == set(item):
                if p >= priority:
                    break
                del self.queue[index]
                self.queue.append((item, priority))
                break
        else:
            self.queue.append((item, priority))

    def pop(self):
        max_idx = 0
        for index, (i, p) in enumerate(self.queue):
            if self.queue[max_idx][1] < p:
                max_idx = index
        (item, priority) = self.queue[max_idx]
        del self.queue[max_idx]
        return (item, priority)

# Inicializa a fila
queue = PriorityQueue()

# Adiciona o primeiro tuplo (subconjunto, mérito)
queue.push([best_feature], best_value)

# Lista de nós visitados
visited = []

# Contador de retrocessos
n_backtrack = 0

# Limite de retrocessos
max_backtrack = 5

while not queue.isEmpty():
    subset, priority = queue.pop()

    if priority < best_value:
        n_backtrack += 1
    else:
        best_value = priority
        best_subset = subset

    if n_backtrack == max_backtrack:
        break

    for feature in features:
        temp_subset = subset + [feature]

        for node in visited:
            if set(node) == set(temp_subset):
                break
        else:
            visited.append(temp_subset)
            merit = getMerit(temp_subset, label)
            queue.push(temp_subset, merit)

# Avaliação com SVM
# Preditores
X = df[features].to_numpy()
# Alvo
Y = df[label].to_numpy()

# Cronometragem
t0 = time.time()

# Executa SVM com validação cruzada de 10 vezes
svc = svm.SVC(kernel='rbf', C=100, gamma=0.01, random_state=42)
scores = cross_val_score(svc, X, Y, cv=10)
best_score = np.mean(scores)

print("Score: %.2f%% (Time: %.4f s)" % (best_score * 100, time.time() - t0))

# Avaliação com o Melhor Subconjunto
# Preditores
X = df[best_subset].to_numpy()

# Cronometragem
t0 = time.time()

# Executa SVM com validação cruzada de 10 vezes
svc = svm.SVC(kernel='rbf', C=100, gamma=0.01, random_state=42)
scores_subset = cross_val_score(svc, X, Y, cv=10)
best_score = np.mean(scores_subset)

print("Score: %.2f%% (Time: %.4f s)" % (best_score * 100, time.time() - t0))

num_selected_features = len(best_subset)
print("Número de atributos selecionados: %d" % num_selected_features)

reducao_atributos = len(features) - num_selected_features
print("Redução de atributos: %d" % reducao_atributos)

