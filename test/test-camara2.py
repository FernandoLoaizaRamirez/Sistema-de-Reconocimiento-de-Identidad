import cv2
import time
import os
import subprocess
import re
import statistics
from datetime import datetime

# --- CONFIGURACIÓN ---
CAMARAS = [
    {"nombre": "Camara_110", "ip": "192.168.30.110", "url": "rtsp://LoaizaStream:YfPvHbQX68zBCRJ@192.168.30.110:554/Streaming/Channels/101"},
    {"nombre": "Camara_111", "ip": "192.168.30.111", "url": "rtsp://LoaizaStream:YfPvHbQX68zBCRJ@192.168.30.111:554/Streaming/Channels/101"}
]

INTERVALO = 60 
CARPETA_RAIZ = "diagnostico_estabilidad"
ARCHIVO_LOG = f"log_fallos_{datetime.now().strftime('%Y%m%d')}.txt"

# Inicializar métricas
for cam in CAMARAS:
    cam.update({
        "intentos": 0, "exitos_img": 0, 
        "latencias": [], "jitter_max": 0, "packet_loss_total": 0
    })
    os.makedirs(os.path.join(CARPETA_RAIZ, cam["nombre"]), exist_ok=True)

def escribir_log(mensaje):
    with open(ARCHIVO_LOG, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {mensaje}\n")

def get_network_stats(ip):
    import platform
    param = "-n" if platform.system().lower() == "windows" else "-c"
    comando = f"ping {param} 5 {ip}"
    try:
        output = subprocess.check_output(comando, shell=True).decode('latin-1')
        tiempos = re.findall(r"(?:time|tiempo)[=<](\d+\.?\d*)ms", output)
        tiempos = [float(t) for t in tiempos]
        if tiempos:
            latencia_media = sum(tiempos) / len(tiempos)
            jitter = statistics.stdev(tiempos) if len(tiempos) > 1 else 0
            return latencia_media, jitter, 0
        return 0, 0, 100
    except:
        return 0, 0, 100

inicio_prueba = time.time()
print(f"--- Monitoreo de Estabilidad Iniciado: {datetime.now().strftime('%H:%M:%S')} ---")
escribir_log("INICIO DE MONITOREO DE ESTABILIDAD")

try:
    while True:
        print(f"\n>>> Ciclo de prueba: {datetime.now().strftime('%H:%M:%S')}")
        for cam in CAMARAS:
            cam["intentos"] += 1
            
            # 1. Medición de Red (Latencia, Jitter, Pérdida)
            lat, jit, loss = get_network_stats(cam["ip"])
            cam["latencias"].append(lat)
            if jit > cam["jitter_max"]: cam["jitter_max"] = jit
            if loss > 0: cam["packet_loss_total"] += 1

            # 2. Medición de Imagen (Video)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(cam["url"], cv2.CAP_FFMPEG)
            
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cam["exitos_img"] += 1
                    status_img = "OK"
                else:
                    status_img = "FALLO_VIDEO"
                    escribir_log(f"FALLO: {cam['nombre']} - Sin flujo de video.")
            else:
                status_img = "DESCONECTADO"
                escribir_log(f"CRÍTICO: {cam['nombre']} - Error de conexión RTSP.")
            
            cap.release()
            print(f"  [{cam['nombre']}] Intento #{cam['intentos']} | Lat: {lat:.1f}ms | Jitter: {jit:.1f}ms | Video: {status_img}")

        time.sleep(INTERVALO)

except KeyboardInterrupt:
    fin_prueba = time.time()
    duracion_segundos = fin_prueba - inicio_prueba
    horas = int(duracion_segundos // 3600)
    minutos = int((duracion_segundos % 3600) // 60)
    
    print("\n" + "="*95)
    print(f"RESUMEN DE ESTABILIDAD - TIEMPO TOTAL: {horas}h {minutos}min")
    print("="*95)
    
    # Tabla con columna de Intentos reintegrada
    header = "{:<12} | {:<10} | {:<10} | {:<12} | {:<10} | {:<10} | {:<10}"
    print(header.format("Cámara", "Intentos", "Disp. Vid", "Latencia Prom", "Jitter Max", "Pkt Loss %", "Estado"))
    print("-" * 95)

    for cam in CAMARAS:
        disp = (cam["exitos_img"] / cam["intentos"]) * 100 if cam["intentos"] > 0 else 0
        lat_avg = sum(cam["latencias"]) / len(cam["latencias"]) if cam["latencias"] else 0
        loss_pct = (cam["packet_loss_total"] / cam["intentos"]) * 100 if cam["intentos"] > 0 else 0
        
        # Clasificación de Estado
        if disp > 98 and loss_pct < 2: estado = "EXCELENTE"
        elif disp < 90 or loss_pct > 5: estado = "DEFICIENTE"
        else: estado = "ESTABLE"

        print(header.format(
            cam["nombre"], 
            cam["intentos"],
            f"{disp:.1f}%", 
            f"{lat_avg:.1f}ms", 
            f"{cam['jitter_max']:.1f}ms", 
            f"{loss_pct:.1f}%", 
            estado
        ))
    
    escribir_log(f"MONITOREO FINALIZADO. Tiempo: {horas}h {minutos}min")
    print(f"\nReporte de fallos guardado en: {ARCHIVO_LOG}")