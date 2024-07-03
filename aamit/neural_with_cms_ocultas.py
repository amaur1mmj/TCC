import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import joblib
import matplotlib.pyplot as plt

# Carregar o dataset reduzido
df = pd.read_csv('reduced_dataset.csv')

# Embaralhar os dados
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Separar as características e os rótulos
X = df.drop(columns=['Filename']).values
y = df['Filename'].values

# Dividir o dataset em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Salvar o scaler ajustado
joblib.dump(scaler, './models/scaler_camadaOculta.pkl')

# Definir a arquitetura da MLP com regularização e Dropout
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation='softmax'))  # 4 classes (0, 1, 2, 3)

model.compile(optimizer='adamax',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo e capturar o histórico
history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_split=0.2)

# Gráfico da acurácia
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('./img/neural_grafics/accuracy_plot_camadaOculta.png')
plt.show()

# Gráfico da perda
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('./img/neural_grafics/loss_plot_camadaOculta.png')
plt.show()

# Avaliar o modelo nos dados de teste
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')
model.save('./models/MLP_treinado_camadaOculta.h5')
