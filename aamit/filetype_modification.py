import os
import random

def change_magic_to_random(directory, magic_numbers):
    for filename in os.listdir(directory):
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension in magic_numbers:
            file_path = os.path.join(directory, filename)
            old_magic = magic_numbers[file_extension]
            change_magic(file_path, old_magic)

def change_magic(file_path, old_magic):
    with open(file_path, 'r+b') as file:
        file.seek(0)
        magic = file.read(len(old_magic))
        if magic != old_magic:
            raise ValueError(f"Magic number mismatch in {file_path}. Expected {old_magic}, found {magic}")
        file.seek(0)
        new_magic = generate_random_magic(len(old_magic))
        file.write(new_magic)

def generate_random_magic(length):
    return bytes(random.getrandbits(8) for _ in range(length))

# Configurações
directory = './data/kk'

# Números mágicos para cada formato de imagem
magic_numbers = {
    '.jpg': b'\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46\x00\x01',   # JPEG
    '.png': b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A',   # PNG
    '.tiff': b'\x49\x49\x2A\x00',  # TIFF (little-endian)
    '.gif': b'\x47\x49\x46\x38\x37\x61'  # GIF
}

# Alterar números mágicos para valores aleatórios
change_magic_to_random(directory, magic_numbers)
