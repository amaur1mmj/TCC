import glob

data = '/home/a1/tcc/dataset/data'

# Obter a lista de todas as imagens
lista_imagens = glob.glob(data + '/**/*', recursive=True)

# Obter o número total de imagens
num_total_imagens = len(lista_imagens)
print(f'O número total de imagens é {num_total_imagens}')