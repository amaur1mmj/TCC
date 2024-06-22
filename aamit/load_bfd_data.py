import pandas as pd

# Carregar o conjunto de dados
df = pd.read_csv('./data/reduced_dataset.csv')

# Mostrar os primeiros 5 registros
print(df.head(5))

# Contar a distribuição dos valores na coluna 'Filename'
class_counts = df['Filename'].value_counts()

# Contar a quantidade de valores não nulos por linha
count_per_row = df.count(axis=1)

# Exibir a contagem de cada valor na coluna 'Filename'
print("Contagem de cada valor na coluna 'Filename':")
print(class_counts)


# Exibir a contagem de valores não nulos por linha
print("Quantidade de valores não nulos por linha:")
print(count_per_row)