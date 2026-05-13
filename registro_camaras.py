import cv2
import os
import numpy as np
import threading
import time
from src.face_engine import FaceEngine

class CentinelaPro:
    def __init__(self):
        self.engine = FaceEngine()
        self.cap = cv2.VideoCapture("rtsp://LoaizaStream:YfPvHbQX68zBCRJ@192.168.30.111:554/Streaming/Channels/101", cv2.CAP_FFMPEG)
        self.db_path = 'data/embeddings'
        self.raw_path = 'data/raw'
        self.frame = None
        self.running = True
        
        # Variables para la ráfaga
        self.fotos_rafaga = []
        self.registrando_id = None
        
        # Crear carpetas si no existen
        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(self.raw_path, exist_ok=True)
        
        # Iniciar hilo de captura
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret: self.frame = frame

    def obtener_siguiente_id(self):
        archivos = [f for f in os.listdir(self.db_path) if f.startswith("trabajador_")]
        if not archivos: return "trabajador_1"
        numeros = [int(f.split("_")[1].split(".")[0]) for f in archivos]
        return f"trabajador_{max(numeros) + 1}"

    def run(self):
        print("🛡️ Sistema Calibrado: Iniciando...")
        while self.running:
            if self.frame is None: continue
            
            h, w = self.frame.shape[:2]
            frame_viz = self.frame.copy()
            
            # Dibujar cuadro de zona basado en porcentajes (Zona más amplia)
            x_m, x_M = int(w * 0.30), int(w * 0.70)
            y_m, y_M = int(h * 0.05), int(h * 0.95)
            cv2.rectangle(frame_viz, (x_m, y_m), (x_M, y_M), (0, 255, 255), 2)

            rostros = self.engine.procesar_frame(self.frame)
            db_actual = {f.replace(".npy", ""): np.load(os.path.join(self.db_path, f)) 
                        for f in os.listdir(self.db_path) if f.endswith(".npy")}

            for r in rostros:
                x1, y1, x2, y2 = r['bbox']
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                if x_m < cx < x_M and y_m < cy < y_M:
                    nombre, score = self.engine.comparar_rostros(r['embedding'], db_actual)
                    
                    if score > 0.50:
                        color, label = (0, 255, 0), f"{nombre} (OK)"
                    else:
                        # --- SOLUCIÓN AL KEYERROR ---
                        # Usamos .get() para evitar que el programa se cierre si no hay 'medidas'
                        medidas = r.get('medidas', {})
                        yaw = abs(medidas.get('yaw', 0)) # Si no hay yaw, asumimos 0 (frente)

                        if yaw < 35:
                            # LÓGICA DE RÁFAGA (5 FOTOS)
                            if self.registrando_id is None:
                                self.registrando_id = self.obtener_siguiente_id()
                                self.fotos_rafaga = []

                            if len(self.fotos_rafaga) < 5:
                                self.fotos_rafaga.append({
                                    'emb': r['embedding'],
                                    'img': self.frame.copy(),
                                    'score': r.get('det_score', 0)
                                })
                                color, label = (0, 0, 255), f"REGISTRANDO {len(self.fotos_rafaga)}/5..."
                            else:
                                # Elegir la foto con mejor puntaje de detección y guardar
                                mejor = max(self.fotos_rafaga, key=lambda x: x['score'])
                                np.save(os.path.join(self.db_path, f"{self.registrando_id}.npy"), mejor['emb'])
                                cv2.imwrite(os.path.join(self.raw_path, f"{self.registrando_id}.jpg"), mejor['img'])
                                
                                print(f"✅ ¡Éxito! Registrado: {self.registrando_id}")
                                self.registrando_id = None # Reset para el siguiente
                                self.fotos_rafaga = []
                        else:
                            color, label = (255, 255, 255), "Gira un poco a la camara"
                else:
                    color, label = (150, 150, 150), "Fuera de zona"

                cv2.rectangle(frame_viz, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_viz, label, (x1, y1-10), 0, 0.6, color, 2)

            cv2.imshow("Calibracion Final", cv2.resize(frame_viz, (1280, 720)))
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    CentinelaPro().run()