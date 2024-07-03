import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from math import sqrt

# Carregar o conjunto de dados
print("Carregando o conjunto de dados...")
df = pd.read_csv('bfd_output.csv')

# Embaralhar os dados
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Extrair o rótulo dos grupos a partir da coluna Filename
labels = df['Filename']
df_features = df.drop(columns=['Filename'])

def getMerit(subset, df, labels):
    k = len(subset)
    if k == 0:
        return 0
    
    # average feature-class correlation
    rcf_all = []
    for feature in subset:
        coeff, _ = spearmanr(labels, df[feature])
        rcf_all.append(abs(coeff))
    rcf = np.mean(rcf_all)

    # average feature-feature correlation
    corr = df[subset].corr()
    corr.values[np.tril_indices_from(corr.values)] = np.nan
    corr = abs(corr)
    rff = corr.unstack().mean()

    return (k * rcf) / sqrt(k + k * (k-1) * rff)

def initialize_population(num_individuals, num_features):
    return np.random.randint(2, size=(num_individuals, num_features))

def evaluate_population(population, df, labels):
    fitness_values = []
    for ind in population:
        subset = [df.columns[i] for i in range(len(ind)) if ind[i] == 1]
        fitness = getMerit(subset, df, labels)
        fitness_values.append(fitness)
    return fitness_values

def selection(population, fitness_values, num_parents):
    selected_indices = np.argsort(fitness_values)[-num_parents:]
    return population[selected_indices]

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1] / 2)
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover):
    for idx in range(offspring_crossover.shape[0]):
        rand_gene = np.random.randint(0, offspring_crossover.shape[1])
        offspring_crossover[idx, rand_gene] = 1 - offspring_crossover[idx, rand_gene]
    return offspring_crossover

# Parâmetros do algoritmo genético
num_generations = 100
num_individuals = 50
num_parents = 15
num_features = 256

# Inicialização da população
population = initialize_population(num_individuals, num_features)
print("População inicial:")
print(population)

# Evolução da população
for generation in range(num_generations):
    fitness_values = evaluate_population(population, df_features, labels)
    parents = selection(population, fitness_values, num_parents)
    offspring_crossover = crossover(parents, offspring_size=(num_individuals - parents.shape[0], num_features))
    offspring_mutated = mutation(offspring_crossover)
    population[0:parents.shape[0], :] = parents
    population[parents.shape[0]:, :] = offspring_mutated

    # Print dos melhores resultados por geração
    if generation % 10 == 0:
        best_fitness = np.max(fitness_values)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

# Selecionando o melhor indivíduo da última geração
fitness_values = evaluate_population(population, df_features, labels)  # Reavaliar a última população para garantir que fitness_values esteja atualizado
best_individual = population[np.argmax(fitness_values)]
selected_features = [df_features.columns[i] for i in range(len(best_individual)) if best_individual[i] == 1]
reduced_df = df[selected_features + ['Filename']]  # Reintroduzir a coluna Filename ao dataframe final

# Salvar o novo dataset reduzido em um arquivo CSV
reduced_df.to_csv('reduced_dataset_new.csv', index=False)

print("\nSelected Features:")
print(selected_features, len(selected_features))
print("\nReduced DataFrame:")
print(reduced_df.head())
