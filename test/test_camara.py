import cv2
import time
import os

# Configuración
URL = "rtsp://LoaizaStream:YfPvHbQX68zBCRJ@192.168.30.110:554/Streaming/Channels/101"
INTERVALO = 60  # Un intento cada 60 segundos
CARPETA_LOGS = "pruebas_estabilidad"

if not os.path.exists(CARPETA_LOGS):
    os.makedirs(CARPETA_LOGS)

# Contadores
intentos_totales = 0
exitos = 0
fallos = 0
inicio_prueba = time.time()

print(f"--- Iniciando Monitoreo de Estabilidad ---")
print(f"Objetivo: Verificar respuesta de imagen cada {INTERVALO}s")

try:
    while True:
        intentos_totales += 1
        timestamp = time.strftime("%H:%M:%S")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        
        cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                exitos += 1
                # Guardamos una imagen minúscula para evidencia técnica
                frame_min = cv2.resize(frame, (160, 90))
                cv2.imwrite(f"{CARPETA_LOGS}/last_check.jpg", frame_min, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
                status = "OK"
            else:
                fallos += 1
                status = "FALLO (Sin Imagen)"
        else:
            fallos += 1
            status = "FALLO (No conectó)"
        
        # Calcular porcentaje de estabilidad
        porcentaje = (exitos / intentos_totales) * 100
        tiempo_transcurrido = (time.time() - inicio_prueba) / 60
        
        print(f"[{timestamp}] Intento #{intentos_totales} | {status} | Estabilidad: {porcentaje:.2f}% | Tiempo: {tiempo_transcurrido:.1f} min")
        
        cap.release()
        time.sleep(INTERVALO)

except KeyboardInterrupt:
    # Esto ocurre cuando presionas Ctrl+C para detenerlo
    print("\n--- RESUMEN FINAL DE LA PRUEBA ---")
    print(f"Tiempo total: {((time.time() - inicio_prueba) / 60):.2f} minutos")
    print(f"Intentos realizados: {intentos_totales}")
    print(f"Éxitos: {exitos}")
    print(f"Fallos detectados: {fallos}")
    print(f"Disponibilidad real: {(exitos / intentos_totales * 100):.2f}%")