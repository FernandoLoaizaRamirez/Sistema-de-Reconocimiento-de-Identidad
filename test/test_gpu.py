from insightface.app import FaceAnalysis
import numpy as np
import time

print("Inicializando modelo...")
app = FaceAnalysis(providers=['CUDAExecutionProvider'])

print("Preparando...")
app.prepare(ctx_id=0)

print("Procesando imágenes...")

# imagen falsa
img = np.random.randint(0,255,(640,640,3),dtype=np.uint8)

# Ejecutar varias veces para forzar uso de GPU
for i in range(100):
    faces = app.get(img)

print("Esperando para ver GPU...")
time.sleep(10)

print("Listo 🚀")