from PIL import Image
#import numpy as np
import os
import glob

# Diretório onde as imagens estão localizadas
diretorio_dataset = '/home/a1/tcc/dataset/caltech-101/101_ObjectCategories'

# Obter a lista de todas as imagens
lista_imagens = glob.glob(diretorio_dataset + '/**/*.jpg', recursive=True)

# Obter o número total de imagens
num_total_imagens = len(lista_imagens)
print(f'O número total de imagens é {num_total_imagens}')

# Diretórios onde as novas imagens serão salvas
data_png = '/home/a1/tcc/dataset/data/png'
data_gif = '/home/a1/tcc/dataset/data/gif'
data_jpg = '/home/a1/tcc/dataset/data/jpg'

# Certifique-se de que os diretórios existem
os.makedirs(data_png, exist_ok=True)
os.makedirs(data_gif, exist_ok=True)
os.makedirs(data_jpg, exist_ok=True)

# Loop através de todas as imagens
for i, caminho_imagem in enumerate(lista_imagens):
    # Carregar a imagem
    imagem = Image.open(caminho_imagem)

    # Salvar a imagem nos formatos PNG, GIF e JPEG
    if i % 3 == 0:
        nome_base = os.path.join(data_png, f'imagem_{i}')
        imagem.save(nome_base + '.png')
    elif i % 3 == 1:
        nome_base = os.path.join(data_gif, f'imagem_{i}')
        imagem.save(nome_base + '.gif')
    else:
        nome_base = os.path.join(data_jpg, f'imagem_{i}')
        imagem.save(nome_base + '.jpg')

print('Todas as imagens foram processadas.')
