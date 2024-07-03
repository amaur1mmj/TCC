from PIL import Image
import os

def change_extension_and_magic(directory, old_extension, new_extension, old_magic, new_magic):
    for filename in os.listdir(directory):
        if filename.lower().endswith(old_extension):
            base = os.path.splitext(filename)[0]
            old_name = os.path.join(directory, filename)
            new_name = os.path.join(directory, f"{base}{new_extension}")
            os.rename(old_name, new_name)
            fix_magic(new_name, old_magic, new_magic)

def fix_magic(file_path, old_magic, new_magic):
    with open(file_path, 'r+b') as file:
        file.seek(0)
        magic = file.read(len(old_magic))
        if magic != old_magic:
            raise ValueError(f"Magic number mismatch. Expected {old_magic}, found {magic}")
        file.seek(0)
        file.write(new_magic)

# Configurações
directory = '/home/a1/tcc/aamit/data/png_modificado_jpg'
old_extension = '.png'   # Extensão atual dos arquivos
new_extension = '.jpg'   # Nova extensão desejada ('.jpg', '.png', '.tiff', etc.)

# Números mágicos para cada formato de imagem
magic_numbers = {
    '.jpg': b'\xFF\xD8\xFF\xE0',   # JPEG
    '.png': b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A',   # PNG
    '.tiff': b'\x49\x49\x2A\x00',   # TIFF (little-endian)
    '.gif': b'\x47\x49\x46\x38\x37\x61'  # GIF
}

# Alterar extensões e ajustar números mágicos
old_magic = magic_numbers.get(old_extension)
new_magic = magic_numbers.get(new_extension)

if old_magic and new_magic:
    change_extension_and_magic(directory, old_extension, new_extension, old_magic, new_magic)
else:
    print(f"Números mágicos não definidos para {old_extension} ou {new_extension}. Verifique as configurações.")
