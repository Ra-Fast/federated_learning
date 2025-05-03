import os
import shutil
from sklearn.model_selection import train_test_split
import random

# Ruta de origen
#source_dir = r"C:\Users\rafae\OneDrive\Doctorado\Datasets\patología\patología\citologias\citologias\citologias_revisadas_LMG\REVISADOS\2021_RepoMED\CLASIFICACION"
source_dir = r"C:\Users\rafae\OneDrive\Doctorado\Datasets\patología\patología\citologias\citologias\citologias_revisadas_LMG\REVISADOS\2019_bhs\CLASIFICACIÓN"
# Ruta de destino
dest_dir = r"C:\Users\rafae\OneDrive\Doctorado\Datasets\train_test\RepoMED\CLASIFICACION"

# Crear directorios de destino
os.makedirs(os.path.join(dest_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(dest_dir, "test"), exist_ok=True)

# Definir proporción de datos de prueba
test_size = 0.2

# Procesar cada categoría
categories = ["ASC-US", "Data_H-Sil", "Data_L-Sil", "Data_Normal", "ASC-H", "Carcinoma"]
for category in categories:
    # Nombre normalizado de la categoría para los directorios de destino
    dest_category = category.replace("Data_", "")
    
    # Crear directorios de destino para esta categoría
    os.makedirs(os.path.join(dest_dir, "train", dest_category), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "test", dest_category), exist_ok=True)
    
    # Listar todas las imágenes de esta categoría
    source_category_dir = os.path.join(source_dir, category)
    images = [f for f in os.listdir(source_category_dir) if os.path.isfile(os.path.join(source_category_dir, f))]
    
    # Dividir en entrenamiento y prueba
    train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
    
    # Copiar imágenes de entrenamiento
    for img in train_images:
        shutil.copy2(
            os.path.join(source_category_dir, img),
            os.path.join(dest_dir, "train", dest_category, img)
        )
    
    # Copiar imágenes de prueba
    for img in test_images:
        shutil.copy2(
            os.path.join(source_category_dir, img),
            os.path.join(dest_dir, "test", dest_category, img)
        )
    
    print(f"Categoría {category}: {len(train_images)} imágenes de entrenamiento, {len(test_images)} imágenes de prueba")
