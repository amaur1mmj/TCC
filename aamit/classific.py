import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Criar pasta para salvar a imagem, se não existir
output_dir = './img'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Carregar o modelo treinado
model = load_model('./models/MLP_treinado.h5')

# Carregar o CSV com os vetores BFD das novas imagens
new_bfd_df = pd.read_csv('bfd_png_md_jpg.csv')

# Índices das características selecionadas pelo GA (exemplo)
selected_features_indices = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18, 22, 23, 33, 40, 47, 62, 81, 84, 135, 158, 189, 192, 200, 202, 209, 213, 215, 219, 225, 229, 232, 235, 236, 237, 239, 241, 242, 243, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]

# Selecionar apenas as características relevantes
new_bfd_selected = new_bfd_df.iloc[:, selected_features_indices]

# Carregar o scaler ajustado nos dados de treinamento
scaler = joblib.load('./models/scaler.pkl')

# Normalizar os novos dados
new_bfd_scaled = scaler.transform(new_bfd_selected)

# Fazer predições nas novas imagens
predictions = model.predict(new_bfd_scaled)
predicted_classes = np.argmax(predictions, axis=1)

# Mapeamento das classes para os tipos de arquivo
class_mapping = {0: 'PNG', 1: 'GIF', 2: 'JPG', 3: 'TIFF'}

# Adicionar uma coluna com as previsões ao dataframe
new_bfd_df['Predicted_Class'] = predicted_classes

# Exibir as predições
for i, row in new_bfd_df.iterrows():
    print(f"Row {i}, Predicted Class: {class_mapping[predicted_classes[i]]}")

# Supondo que você tenha uma coluna 'Filename' com os rótulos reais (0 para PNG)
true_labels = new_bfd_df['Filename'].values

# Gerar a matriz de confusão
conf_matrix = confusion_matrix(true_labels, predicted_classes, labels=[0, 1, 2, 3])
print("Confusion Matrix:")
print(conf_matrix)

# Calcular a porcentagem de acertos
accuracy = accuracy_score(true_labels, predicted_classes)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plotar a matriz de confusão
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_mapping.values(), yticklabels=class_mapping.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Salvar a matriz de confusão como imagem
conf_matrix_image_path = os.path.join(output_dir, 'confusion_matrix_PNG_mod_JPG.png')
plt.savefig(conf_matrix_image_path)
plt.close()

# Exibir o relatório de classificação
class_report = classification_report(true_labels, predicted_classes, labels=[0, 1, 2, 3], target_names=class_mapping.values(), zero_division=0)
print("Classification Report:")
print(class_report)

# Salvar o relatório de classificação em um arquivo de texto
class_report_path = os.path.join(output_dir, 'classification_report_PNG_mod_JPG.txt')
with open(class_report_path, 'w') as f:
    f.write("Classification Report:\n")
    f.write(class_report)
    f.write(f"\nAccuracy: {accuracy * 100:.2f}%\n")
