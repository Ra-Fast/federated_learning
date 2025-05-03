from datasets import load_dataset

# Cargar el dataset
dataset = load_dataset("RaFast/RepoMED")

# Acceder a los conjuntos de entrenamiento y prueba
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Ver algunas estadísticas
print(f"Tamaño del conjunto de entrenamiento: {len(train_dataset)}")
print(f"Tamaño del conjunto de prueba: {len(test_dataset)}")
print(f"Categorías: {train_dataset.features['label'].names}")

# Ver una muestra
sample = train_dataset[0]
print(f"Etiqueta: {sample['label']}")
print(f"Nombre de la etiqueta: {train_dataset.features['label'].int2str(sample['label'])}")
sample['image'].show()  # Muestra la imagen si estás en un entorno que lo permite
