import imghdr
import magic

def verificar_tipo_arquivo(filename):
    tipo = imghdr.what(filename)
    if tipo is not None:
        print(f'O tipo do arquivo {filename} é {tipo}.')
        # Abre o arquivo em modo de leitura binária
        with open(filename, 'rb') as arquivo:
            # Cria um objeto Magic para identificação do tipo de arquivo
            m = magic.Magic()
            # Lê os primeiros bytes do arquivo para identificar o número mágico
            numero_magico = m.from_buffer(arquivo.read(2048))
            print(f'O número mágico do arquivo {filename} é {numero_magico}.')
    else:
        print(f'O arquivo {filename} não é uma imagem válida.')

# Substitua 'nome_da_imagem.jpg' pelo caminho do arquivo de imagem que você quer verificar
verificar_tipo_arquivo('./data/tiff/imagem_6867.tiff')
