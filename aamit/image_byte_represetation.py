import os
import numpy as np

# Caminho para a imagem original
file_path = './data/jpg/imagem_4624.jpg'

# Verificando o tamanho do arquivo
file_size = os.path.getsize(file_path)
print(f"Tamanho do arquivo: {file_size} bytes")

# Abrindo o arquivo em modo binário
with open(file_path, 'rb') as file:
    byte_data = file.read()

# Exibindo os primeiros 100 bytes da imagem de forma visualmente melhorada
print("Primeiros 100 bytes da imagem:")
for i in range(0, 100, 10):
    line = byte_data[i:i+10]
    hex_string = ' '.join(f"{byte:02X}" for byte in line)
    print(f"{i:3}: {hex_string}")

# Convertendo os bytes em uma matriz de bytes (8 bits cada)
byte_matrix = np.frombuffer(byte_data[:100], dtype=np.uint8).reshape((10, 10))

# Exibindo a matriz de bytes
print("\nMatriz de bytes (10x10):")
print(byte_matrix)
