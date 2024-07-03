import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Exemplo de classes reais e previsões do modelo
y_true = np.array([1, 0, 1, 2, 1, 0, 2, 1, 2, 0])
y_pred = np.array([1, 0, 1, 2, 1, 0, 1, 1, 2, 0])

# Calcular a matriz de confusão
cm = confusion_matrix(y_true, y_pred)

# Definir rótulos das classes
classes = ['Class 0', 'Class 1', 'Class 2']

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')

# Salvar a imagem
plt.savefig('confusion_matrix.png')
plt.close()
