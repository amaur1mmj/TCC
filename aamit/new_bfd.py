import os
from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd
from collections import Counter
import fitz  # PyMuPDF

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

def load_pdf(pdf_path):
    """
    Carrega um PDF e converte suas páginas para uma sequência de bytes.
    """
    try:
        pdf_document = fitz.open(pdf_path)
        all_bytes = bytearray()
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_bytes = np.array(img).tobytes()
            all_bytes.extend(img_bytes)
        return bytes(all_bytes)
    except Exception as e:
        print(f"Error processing PDF file '{pdf_path}': {e}")
        return None

def calculate_bfd(byte_sequence):
    """
    Calcula a distribuição de frequência de bytes para a sequência de bytes fornecida.
    """
    byte_counts = Counter(byte_sequence)
    total_bytes = len(byte_sequence)
    bfd = [byte_counts.get(byte, 0) / total_bytes for byte in range(256)]
    return bfd

def process_images_and_pdfs(directory, output_path):
    """
    Processa imagens e PDFs em subdiretórios (png, jpg, gif, pdf) e calcula/salva a BFD de cada um em um arquivo CSV.
    """
    subdirs = ['png', 'jpg', 'gif', 'pdf']
    bfd_data = []
    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        if not os.path.exists(subdir_path):
            print(f"Subdiretório {subdir_path} não encontrado.")
            continue

        if subdir == 'pdf':
            file_extension = '.pdf'
        else:
            file_extension = ('.png', '.jpg', '.jpeg', '.gif')

        image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(file_extension)]
        if not image_files:
            print(f"Nenhum arquivo encontrado em {subdir_path}.")
            continue

        for filename in image_files:
            file_path = os.path.join(subdir_path, filename)
            print(f"Processing {filename} in {subdir} folder...")
            
            if subdir == 'pdf':
                byte_sequence = load_pdf(file_path)
            else:
                byte_sequence = load_image(file_path)
                
            if byte_sequence is None:
                continue

            bfd = calculate_bfd(byte_sequence)
            bfd_data.append([filename] + bfd)

    if bfd_data:
        # Salvar os dados em um arquivo CSV
        column_names = ['Filename'] + [f'Byte {i}' for i in range(256)]
        df = pd.DataFrame(bfd_data, columns=column_names)
        df.to_csv(output_path, index=False)
        print(f"BFD data saved to {output_path}.")
    else:
        print("No valid files processed. No BFD data to save.")

# Exemplo de uso
main_directory = '/home/a1/tcc/aamit/data'
output_file = '/home/a1/tcc/aamit/data/bfd_data.csv'
process_images_and_pdfs(main_directory, output_file)
