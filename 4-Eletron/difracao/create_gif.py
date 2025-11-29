"""
create_gif.py

Cria um GIF animado com todas as imagens da pasta raw_images.
"""

from PIL import Image
from pathlib import Path
import re

# Diretório das imagens
raw_images_dir = Path('Data/raw_images')

# Buscar todas as imagens JPEG
image_files = sorted(raw_images_dir.glob('*.jpeg'))

# Filtrar apenas imagens numeradas (ignorar reference_size.jpeg)
numbered_images = []
for img_file in image_files:
    if img_file.stem.isdigit():
        numbered_images.append((int(img_file.stem), img_file))

# Ordenar por número
numbered_images.sort(key=lambda x: x[0])
image_paths = [img[1] for img in numbered_images]

print("="*70)
print("CRIAÇÃO DE GIF - DIFRAÇÃO DE ELÉTRONS")
print("="*70)
print(f"\n📸 Total de imagens encontradas: {len(image_paths)}")
print(f"   Range: {numbered_images[0][0]} - {numbered_images[-1][0]}")

# Carregar todas as imagens
frames = []
for img_path in image_paths:
    img = Image.open(img_path)
    frames.append(img)
    print(f"   Carregada: {img_path.name} ({img.size[0]}x{img.size[1]})")

# Calcular duração de cada frame (5 segundos total)
total_duration_ms = 5000
duration_per_frame = total_duration_ms / len(frames)

print(f"\n🎬 Parâmetros do GIF:")
print(f"   Duração total: {total_duration_ms/1000:.1f} s")
print(f"   Duração por frame: {duration_per_frame:.1f} ms")
print(f"   FPS: {1000/duration_per_frame:.1f}")

# Criar GIF
output_file = Path('difracao_animation.gif')
frames[0].save(
    output_file,
    save_all=True,
    append_images=frames[1:],
    duration=duration_per_frame,
    loop=0  # Loop infinito
)

print(f"\n✅ GIF criado com sucesso!")
print(f"   Arquivo: {output_file}")
print(f"   Tamanho: {output_file.stat().st_size / 1024:.1f} KB")
print("="*70)
