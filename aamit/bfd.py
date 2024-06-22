import os
from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd
from collections import Counter

def load_image(image_path):
    """
    Carrega uma imagem e a converte para uma sequência de bytes.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')  # Converter para RGB para uniformidade
            img_bytes = np.array(img).tobytes()
        return img_bytes
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file '{image_path}'")
        return None

def calculate_bfd(image_bytes):
    """
    Calcula a distribuição de frequência de bytes para a sequência de bytes fornecida.
    """
    byte_counts = Counter(image_bytes)
    total_bytes = len(image_bytes)
    bfd = [byte_counts.get(byte, 0) / total_bytes for byte in range(256)]
    return bfd

def process_images(directory, output_path):
    """
    Processa imagens em subdiretórios (png, jpg, gif, tiff) e calcula/salva a BFD de cada uma em um arquivo CSV.
    """
    subdirs = ['png', 'jpg', 'gif', 'tiff']
    extension_to_num = {'.png': 0, '.jpg': 1, '.jpeg': 1, '.gif': 2, '.tiff': 3, '.tif': 3}
    bfd_data = []

    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        if not os.path.exists(subdir_path):
            print(f"Subdiretório {subdir_path} não encontrado.")
            continue

        image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(tuple(extension_to_num.keys()))]
        for filename in image_files:
            image_path = os.path.join(subdir_path, filename)
            print(f"Processing {filename} in {subdir} folder...")
            image_bytes = load_image(image_path)
            if image_bytes is None:
                continue
            bfd = calculate_bfd(image_bytes)
            # Substituir o nome do arquivo pela extensão correspondente
            file_ext = os.path.splitext(filename)[1].lower()
            file_num = extension_to_num.get(file_ext, -1)
            bfd_data.append([file_num] + bfd)

    # Salvar os dados em um arquivo CSV
    column_names = ['Filename'] + [f'Byte {i}' for i in range(256)]
    df = pd.DataFrame(bfd_data, columns=column_names)
    df.to_csv(output_path, index=False)

# Exemplo de uso
main_directory = '/home/a1/tcc/aamit/data'
output_file = '/home/a1/tcc/aamit/data/bfd_data_new.csv'
process_images(main_directory, output_file)
