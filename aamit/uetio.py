import os
import numpy as np
import pandas as pd
from collections import Counter

def calculate_bfd(file_path):
    """
    Calcula a distribuição de frequência de bytes para o arquivo especificado.
    Retorna contagens de cada byte de 0 a 255.
    """
    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        byte_counts = Counter(file_bytes)
        bfd = [byte_counts.get(byte, 0) for byte in range(256)]
        return bfd
    except Exception as e:
        print(f"Error calculating BFD for file '{file_path}': {str(e)}")
        return None

def process_files(directory, output_path):
    """
    Processa todos os arquivos em um diretório e calcula/salva a BFD de cada um em um arquivo CSV.
    Ignora arquivos com extensão '.Identifier'.
    Substitui o nome completo do arquivo por um número inteiro com base na extensão do arquivo.
    Mantém a ordem dos arquivos como na pasta original.
    """
    bfd_data = []

    # Mapeamento de extensões para números
    extension_mapping = {
        '.png': 0,
        '.gif': 1,
        '.jpg': 2,
        '.tiff': 3
    }

    # Percorrer as extensões na ordem desejada
    for extension, label in extension_mapping.items():
        files = sorted([filename for filename in os.listdir(directory) if filename.lower().endswith(extension)])
        for filename in files:
            if filename.lower().endswith('.identifier'):
                continue  # Ignorar arquivos com extensão '.Identifier'
            file_path = os.path.join(directory, filename)
            try:
                bfd = calculate_bfd(file_path)
                if bfd is None:
                    continue
                # Substituir o nome completo do arquivo pelo número correspondente à extensão
                bfd_data.append([label] + bfd)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

    # Salvar os dados em um arquivo CSV
    column_names = ['Filename'] + [f'Byte {i}' for i in range(256)]
    df = pd.DataFrame(bfd_data, columns=column_names)
    df.to_csv(output_path, index=False)

# Configurações
directory = './data/png_modificado_tiff'
output_path = './data/bfd_output_test.csv'

# Processar arquivos e salvar BFD em um CSV
process_files(directory, output_path)
