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
    Substitui o nome completo do arquivo pelo número inteiro 3 na coluna 'Filename'.
    """
    bfd_data = []

    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.identifier'):
                continue  # Ignorar arquivos com extensão '.Identifier'
            file_path = os.path.join(subdir, filename)
            try:
                bfd = calculate_bfd(file_path)
                if bfd is None:
                    continue
                # Substituir o nome do arquivo pelo número inteiro 3
                bfd_data.append([0] + bfd)  # Aqui adicionamos diretamente o número inteiro 3
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

    # Salvar os dados em um arquivo CSV
    column_names = ['Filename'] + [f'Byte {i}' for i in range(256)]
    df = pd.DataFrame(bfd_data, columns=column_names)
    df.to_csv(output_path, index=False)




main_directory = '/home/a1/tcc/aamit/data/png_modificado_jpg'
output_file = 'bfd_png_md_jpg.csv'
process_files(main_directory, output_file)
