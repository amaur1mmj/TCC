import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import os

# Carregar o conjunto de dados reduzido
df = pd.read_csv('./data/reduced_dataset.csv')
#df = pd.read_csv('./data/bfd_data_new.csv')

# Embaralhar os dados
df = df.sample(frac=1, random_state=64).reset_index(drop=True)

# Separar os recursos e o rótulo
X = df.drop(columns=['Filename']).values  # Excluir a coluna de rótulo
y = df['Filename'].values

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir a arquitetura da rede neural
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),  # Adicionar uma camada densa adicional
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compilar o modelo com uma taxa de aprendizagem reduzida
#optimizer = tf.keras.optimizers.rmsprop(learning_rate=0.001)  # Definir uma taxa de aprendizagem menor
model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#* nadam 55% 
#* rmsprop 54%
#* sgd 53%
#* adam
#* adamax 55%
# Treinar o modelo
history = model.fit(X_train, y_train, epochs=500, validation_split=0.3, batch_size=32)

# Avaliar o modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# Criar o diretório para salvar os gráficos se não existir
os.makedirs('./models', exist_ok=True)

# Plotar e salvar a acurácia durante o treinamento
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('./models/training_validation_accuracy.png')

# Fazer previsões
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Mostrar algumas previsões
print("True labels: ", y_test[:10])
print("Predicted labels: ", predicted_classes[:10])

# Calcular e exibir a matriz de confusão
conf_matrix = confusion_matrix(y_test, predicted_classes)
print('Matriz de Confusão:')
print(conf_matrix)

