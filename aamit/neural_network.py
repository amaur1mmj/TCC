import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# Carregar o conjunto de dados
print("Carregando o conjunto de dados...")
df = pd.read_csv('./data/reduced_dataset.csv')

# Separar os dados em features (X) e alvo (y)
print("Separando os dados em features e alvo...")
X = df.drop(columns=['Filename'])
y = df['Filename']

# Embaralhar os dados
print("Embaralhando os dados...")
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
X = df_shuffled.drop(columns=['Filename'])
y = df_shuffled['Filename']

# Normalizar os dados para evitar valores muito altos ou baixos
print("Normalizando os dados para evitar valores extremos...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dividir o conjunto de dados em treino, validação e teste
print("Dividindo o conjunto de dados em treino, validação e teste...")
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.1, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.9, random_state=42, stratify=y_temp)

# Treinar o modelo XGBoost
print("Treinando o modelo XGBoost...")
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4, random_state=42)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)

# Obter as previsões do XGBoost
print("Obtendo as previsões do XGBoost...")
xgb_train_preds = xgb_model.predict_proba(X_train)
xgb_val_preds = xgb_model.predict_proba(X_val)
xgb_test_preds = xgb_model.predict_proba(X_test)

# Combinar as previsões do XGBoost com as features originais
print("Combinando as previsões do XGBoost com as features originais...")
X_train_combined = np.hstack((X_train, xgb_train_preds))
X_val_combined = np.hstack((X_val, xgb_val_preds))
X_test_combined = np.hstack((X_test, xgb_test_preds))

# Preparar os rótulos para a rede neural
print("Preparando os rótulos para a rede neural...")
y_train_categorical = to_categorical(y_train)
y_val_categorical = to_categorical(y_val)
y_test_categorical = to_categorical(y_test)

# Definir e treinar o modelo MLP
print("Definindo e treinando o modelo MLP...")
mlp_model = Sequential()
mlp_model.add(Dense(256, input_dim=X_train_combined.shape[1], activation='relu'))
mlp_model.add(Dropout(0.2))
mlp_model.add(Dense(128, activation='tanh'))
mlp_model.add(Dense(64, activation='tanh'))
mlp_model.add(Dropout(0.2))
mlp_model.add(Dense(4, activation='softmax'))

mlp_model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history = mlp_model.fit(X_train_combined, y_train_categorical, epochs=500, batch_size=32, validation_data=(X_val_combined, y_val_categorical))

# Avaliar o modelo nos dados de teste
print("Avaliando o modelo nos dados de teste...")
y_test_pred_mlp = mlp_model.predict(X_test_combined)
print("Shape das previsões do MLP:", y_test_pred_mlp.shape)  # Adicionada linha para verificar a forma das previsões
y_test_pred_classes = np.argmax(y_test_pred_mlp, axis=1)
test_accuracy = accuracy_score(y_test, y_test_pred_classes)
print(f'Acurácia no teste: {test_accuracy}')
print(classification_report(y_test, y_test_pred_classes))

# Salvar o modelo treinado
print("Salvando o modelo treinado...")
mlp_model.save('./models/mlp_model_combined.h5')

# Salvar o scaler usado
print("Salvando o scaler usado.. <3")
import joblib
joblib.dump(scaler, "./models/scaler_combined.pkl")

# Plotar e salvar os gráficos de acurácia e perda
print("Plotando e salvando os gráficos de acurácia e perda...")
plt.figure(figsize=(12, 6))

# Gráfico de acurácia
plt.subplot(1, 2, 1)
plt.plot(np.array(history.history['accuracy']) * 100, label='Acurácia de Treinamento')
plt.plot(np.array(history.history['val_accuracy']) * 100, label='Acurácia de Validação')
plt.title('Acurácia durante o Treinamento e Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia (%)')
plt.legend()
plt.grid(True)

# Gráfico de perda
plt.subplot(1, 2, 2)
plt.plot(np.array(history.history['loss']) * 100, label='Perda de Treinamento')
plt.plot(np.array(history.history['val_loss']) * 100, label='Perda de Validação')
plt.title('Perda durante o Treinamento e Validação')
plt.xlabel('Época')
plt.ylabel('Perda (%)')
plt.legend()
plt.grid(True)

plt.savefig('./models/training_plots.png')
print("Gráficos salvos com sucesso.")
