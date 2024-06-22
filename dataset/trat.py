from PIL import Image
import os
import glob

# Diretório onde as imagens estão localizadas
diretorio_dataset = '/home/a1/tcc/dataset/caltech-101/101_ObjectCategories'

# Obter a lista de todas as imagens
lista_imagens = glob.glob(diretorio_dataset + '/**/*.jpg', recursive=True)

# Obter o número total de imagens
num_total_imagens = len(lista_imagens)
print(f'O número total de imagens é {num_total_imagens}')

# Calcular o número de imagens para cada formato
num_imagens_por_formato = num_total_imagens // 4
print(f'Número de imagens por formato: {num_imagens_por_formato}')

# Diretórios onde as novas imagens serão salvas
data_png = '/home/a1/tcc/dataset/data/png'
data_gif = '/home/a1/tcc/dataset/data/gif'
data_jpg = '/home/a1/tcc/dataset/data/jpg'
data_tiff = '/home/a1/tcc/dataset/data/tiff'

# Certifique-se de que os diretórios existem
os.makedirs(data_png, exist_ok=True)
os.makedirs(data_gif, exist_ok=True)
os.makedirs(data_jpg, exist_ok=True)
os.makedirs(data_tiff, exist_ok=True)

# Loop através de todas as imagens e distribuí-las entre os formatos
for i, caminho_imagem in enumerate(lista_imagens):
    # Carregar a imagem
    imagem = Image.open(caminho_imagem)
    
    # Determinar o formato de acordo com o índice
    if i < num_imagens_por_formato:
        nome_base = os.path.join(data_png, f'imagem_{i}')
        imagem.save(nome_base + '.png')
    elif i < 2 * num_imagens_por_formato:
        nome_base = os.path.join(data_gif, f'imagem_{i}')
        imagem.save(nome_base + '.gif')
    elif i < 3 * num_imagens_por_formato:
        nome_base = os.path.join(data_jpg, f'imagem_{i}')
        imagem.save(nome_base + '.jpg')
    else:
        nome_base = os.path.join(data_tiff, f'imagem_{i}')
        imagem.save(nome_base + '.tiff')

print('Todas as imagens foram processadas.')
