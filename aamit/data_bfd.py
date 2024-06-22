import pandas as pd

def check_bfd_data(file_path):
    """
    Verifica se todas as linhas no arquivo CSV de BFD têm o número correto de valores.
    """
    df = pd.read_csv(file_path)
    num_columns_expected = 257  # 256 valores de byte + 1 coluna para o nome do arquivo
    
    num_rows_with_wrong_columns = sum(df.apply(lambda row: len(row) != num_columns_expected, axis=1))
    
    if num_rows_with_wrong_columns == 0:
        print("Todos os dados no arquivo BFD estão corretos.")
    else:
        print(f"Foram encontradas {num_rows_with_wrong_columns} linhas com o número errado de valores.")

# Exemplo de uso
file_path = '/home/a1/tcc/aamit/data/bfd_data.csv'
check_bfd_data(file_path)
