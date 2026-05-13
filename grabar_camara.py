import cv2
import os
import threading
import time
from datetime import datetime

# --- CONFIGURACIÓN ---
CAMARAS = [
    {"nombre": "Camara_110", "url": "rtsp://LoaizaStream:YfPvHbQX68zBCRJ@192.168.30.110:554/Streaming/Channels/101"},
    {"nombre": "Camara_111", "url": "rtsp://LoaizaStream:YfPvHbQX68zBCRJ@192.168.30.111:554/Streaming/Channels/101"}
]

CARPETA_RAIZ = "videos_camaras"
DURACION_SEGMENTO = 300  # 5 minutos (5 * 60 segundos)

class GrabadorSegmentado:
    def __init__(self, nombre, url):
        self.nombre = nombre
        self.url = url
        self.cap = None
        self.out = None
        self.grabando = True
        self.conectado = False
        self.inicio_segmento = 0
        
        # Crear subcarpeta para cada cámara
        self.ruta_carpeta = os.path.join(CARPETA_RAIZ, self.nombre)
        os.makedirs(self.ruta_carpeta, exist_ok=True)
        
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    def conectar(self):
        if self.cap: self.cap.release()
        if self.out: self.out.release()
        
        print(f"[{self.nombre}] Intentando conectar...")
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        
        if self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 60: fps = 25
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archivo = os.path.join(self.ruta_carpeta, f"{self.nombre}_{timestamp}.avi")
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(archivo, fourcc, fps, (w, h))
            
            self.inicio_segmento = time.time()
            self.conectado = True
            print(f"[{self.nombre}] Nuevo segmento iniciado: {archivo}")
            return True
        return False

    def ejecutar(self):
        while self.grabando:
            if not self.conectado:
                if not self.conectar():
                    time.sleep(5)
                    continue

            ret, frame = self.cap.read()
            
            if ret:
                self.out.write(frame)
                
                # Verificar si pasaron los 5 minutos para cortar el video
                if time.time() - self.inicio_segmento >= DURACION_SEGMENTO:
                    print(f"[{self.nombre}] Corte de 5 min alcanzado. Rotando archivo...")
                    self.conectado = False # Esto forzará la reconexión y nuevo archivo
            else:
                print(f"[{self.nombre}] Error de lectura. Reconectando...")
                self.conectado = False
                time.sleep(1)

        self.limpiar()

    def limpiar(self):
        if self.cap: self.cap.release()
        if self.out: self.out.release()
        print(f"[{self.nombre}] Recursos liberados.")

# --- INICIO DEL PROGRAMA ---
if __name__ == "__main__":
    hilos = []
    instancias = []

    print(f"--- Iniciando Grabación Dual Segmentada (Cortes de 5 min) ---")

    for cam in CAMARAS:
        obj = GrabadorSegmentado(cam["nombre"], cam["url"])
        instancias.append(obj)
        t = threading.Thread(target=obj.ejecutar)
        t.start()
        hilos.append(t)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDeteniendo grabaciones...")
        for obj in instancias:
            obj.grabando = False
        for t in hilos:
            t.join()
        print("Sistema detenido correctamente.")