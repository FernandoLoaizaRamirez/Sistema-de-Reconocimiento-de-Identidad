import cv2
import time

# --- CONFIGURACIÓN ---
NOMBRE_ARCHIVO = "video_prueba_benchmark.mp4"
RESOLUCION = (1280, 720) # Resolución estándar HD
FPS = 20.0               # Cuadros por segundo

# Iniciar la webcam
cap = cv2.VideoCapture(0) # El 0 es la webcam integrada

# Configurar el codec y el objeto de escritura
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(NOMBRE_ARCHIVO, fourcc, FPS, RESOLUCION)

print(f"🔴 Iniciando grabación: {NOMBRE_ARCHIVO}")
print("Presiona 'q' para detener la grabación.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Asegurar que el frame tenga el tamaño correcto
    frame = cv2.resize(frame, RESOLUCION)

    # Escribir el frame en el archivo
    out.write(frame)

    # Mostrar visualización con un indicador de grabación
    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
    cv2.putText(frame, "GRABANDO...", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Grabando Video de Prueba', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar todo
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Video guardado exitosamente como: {NOMBRE_ARCHIVO}")