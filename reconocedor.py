"""
reconocedor.py — Sistema de Reconocimiento Facial
==================================================
Modelo : buffalo_l (InsightFace)
DB     : PKLs cargados en RAM al arrancar — búsqueda vectorizada
Display: verde = reconocido, rojo = desconocido

Uso:
  python reconocedor.py
"""

import cv2
import numpy as np
import os
import sys
import pickle
import time
import threading
import argparse

sys.path.insert(0, os.path.dirname(__file__))
try:
    from src.face_engine import FaceEngine
except ModuleNotFoundError:
    from face_engine import FaceEngine

from insightface.app import FaceAnalysis
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================================
# AJUSTA AQUÍ SI NECESITAS
# ==========================================================
DB_PATH               = "data/embeddings"
UMBRAL_RECONOCIMIENTO = 1.05   # L2: menor = más estricto
TAMANO_MIN_PX         = 50     # ignorar rostros más pequeños que esto
COOLDOWN_LABEL_SEG    = 2.0    # segundos que permanece el nombre en pantalla

RTSP_CAM1 = "rtsp://LoaizaStream:YfPvHbQX68zBCRJ@192.168.30.110:554/Streaming/Channels/101"
RTSP_CAM2 = "rtsp://LoaizaStream:YfPvHbQX68zBCRJ@192.168.30.111:554/Streaming/Channels/101"

COLOR_VERDE = (0, 210, 0)
COLOR_ROJO  = (0, 0, 210)
FONT        = cv2.FONT_HERSHEY_SIMPLEX


# ==========================================================
# GALERÍA EN RAM
# ==========================================================

class GaleriaRAM:
    """
    Carga todos los PKL en un diccionario:
      { 'Jose_Fernando': np.array shape (N, 512) }   N = cantidad de embeddings del sujeto

    Al buscar, construye una matriz global (M_total, 512) y hace
    una sola operación np.linalg.norm() — sin loops por persona.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        # { subject_id: matrix (N, 512) }
        self._galeria: dict[str, np.ndarray] = {}
        # Para búsqueda vectorizada: matriz plana + mapa de índice → subject_id
        self._matrix_global: np.ndarray | None = None
        self._id_map: list[str] = []   # posición i → subject_id
        self.cargar()

    # ------------------------------------------------------------------

    def cargar(self):
        """Lee todos los PKL del directorio y los sube a RAM."""
        self._galeria = {}

        if not os.path.exists(self.db_path):
            print(f"  AVISO: no existe {self.db_path}")
            self._reconstruir_index()
            return

        cargados, errores = 0, 0
        for fname in sorted(os.listdir(self.db_path)):
            if not fname.endswith("_embedding.pkl"):
                continue
            ruta = os.path.join(self.db_path, fname)
            try:
                with open(ruta, "rb") as f:
                    data = pickle.load(f)
                embeddings = self._extraer_embeddings(data)
                if not embeddings:
                    continue
                # subject_id = nombre del archivo sin sufijo
                sid = fname.replace("_embedding.pkl", "")
                self._galeria[sid] = np.array(embeddings, dtype=np.float32)
                cargados += 1
            except Exception as e:
                print(f"  Error leyendo {fname}: {e}")
                errores += 1

        self._reconstruir_index()

        total_embs = sum(m.shape[0] for m in self._galeria.values())
        print(f"\n  DB cargada: {cargados} personas | "
              f"{total_embs} embeddings totales en RAM"
              + (f" | {errores} errores" if errores else ""))
        for sid, mat in self._galeria.items():
            print(f"    {sid:<40}  {mat.shape[0]} embedding/s")

    def _extraer_embeddings(self, data: dict) -> list:
        """
        Compatible con el único formato que queda (v4):
          data['gallery'] = [ {'embedding': array, ...}, ... ]
        Por seguridad también acepta el v1 legacy por si quedara alguno.
        """
        embs = []
        if data.get("version") == 2 and "gallery" in data:
            # Formato v4 — galería multi-ángulo
            for entry in data["gallery"]:
                emb  = entry["embedding"].astype(np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    embs.append(emb / norm)
        elif "embedding" in data:
            # Formato v1 legacy — por si acaso
            emb  = data["embedding"].astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                embs.append(emb / norm)
        return embs

    def _reconstruir_index(self):
        """Aplana la galería en una sola matriz para búsqueda vectorizada."""
        filas, ids = [], []
        for sid, matrix in self._galeria.items():
            for emb in matrix:
                filas.append(emb)
                ids.append(sid)
        if filas:
            self._matrix_global = np.array(filas, dtype=np.float32)
        else:
            self._matrix_global = None
        self._id_map = ids

    # ------------------------------------------------------------------

    def buscar(self, query: np.ndarray, umbral: float) -> tuple:
        """
        Búsqueda vectorizada L2 contra toda la DB en una sola operación.

        Retorna (encontrado: bool, subject_id: str | None, distancia: float)
        """
        if self._matrix_global is None:
            return False, None, 2.0

        norm = np.linalg.norm(query)
        q    = query / norm if norm > 0 else query

        # (M_total,) — una distancia por embedding
        dists  = np.linalg.norm(self._matrix_global - q, axis=1)
        idx    = int(np.argmin(dists))
        dist   = float(dists[idx])
        sid    = self._id_map[idx]

        return (dist < umbral), sid, dist

    def __len__(self):
        return len(self._galeria)


# ==========================================================
# MOTOR BUFFALO_L CONFIGURABLE
# ==========================================================

class MotorReconocimiento:
    """
    Envuelve FaceAnalysis con det_size configurable.
    det_size=(1280,1280) detecta caras desde ~40px pero es más lento.
    det_size=(640,640)   detecta desde ~80px, más rápido.
    """

    def __init__(self, det_size: int = 640):
        size = (det_size, det_size)
        print(f"\n  Cargando buffalo_l  det_size={size} ...")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=size)
        for m in self.app.models:
            print(f"    sub-modelo: {m}")

    def procesar_frame(self, frame: np.ndarray) -> list:
        if frame is None or frame.size == 0:
            return []
        try:
            faces = self.app.get(frame)
        except Exception:
            return []
        resultados = []
        for face in faces:
            bbox = face.bbox.astype(int)
            ancho = int(bbox[2] - bbox[0])
            alto  = int(bbox[3] - bbox[1])
            try:
                p, y, r = face.pose
            except Exception:
                p, y, r = 0.0, 0.0, 0.0
            emb = face.normed_embedding
            if emb is None:
                continue
            resultados.append({
                "bbox"     : bbox,
                "embedding": emb,
                "res"      : (ancho, alto),
                "pose"     : (p, y, r),
            })
        return resultados


# ==========================================================
# STREAM RTSP SIN LATENCIA ACUMULADA
# ==========================================================

class CameraStream:
    def __init__(self, source: str):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


# ==========================================================
# DISPLAY
# ==========================================================

def _nombre_display(sid: str) -> str:
    """Convierte 'Jose_Fernando_Loaiza_Ramirez' → 'Jose Fernando' para el label."""
    partes = sid.replace("_embedding", "").split("_")
    # Tomar solo nombre + primer apellido para que quepa en pantalla
    return " ".join(partes[:2]) if len(partes) >= 2 else partes[0]


def dibujar_rostro(frame, bbox, nombre_display, dist, reconocido):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = COLOR_VERDE if reconocido else COLOR_ROJO
    label = nombre_display if reconocido else "Desconocido"

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Fondo del texto para legibilidad
    txt = f"{label}  {dist:.2f}"
    (tw, th), bl = cv2.getTextSize(txt, FONT, 0.5, 1)
    y_bg = max(y1 - 6, th + 6)
    cv2.rectangle(frame,
                  (x1, y_bg - th - 4),
                  (x1 + tw + 6, y_bg + bl),
                  color, cv2.FILLED)
    cv2.putText(frame, txt,
                (x1 + 3, y_bg - 2),
                FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def dibujar_hud(frame, galeria, fps, det_size):
    """Esquina superior izquierda: métricas de sesión."""
    info = [
        f"FPS: {fps:.1f}",
        f"Personas DB: {len(galeria)}",
        f"Umbral L2: {UMBRAL_RECONOCIMIENTO}",
        f"det_size: {det_size}x{det_size}",
        "Q=salir  R=recargar DB",
    ]
    for i, txt in enumerate(info):
        cv2.putText(frame, txt, (8, 20 + i * 18),
                    FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


# ==========================================================
# LOOP PRINCIPAL
# ==========================================================

def procesar_fuente(fuente: str, motor: MotorReconocimiento,
                    galeria: GaleriaRAM, det_size: int,
                    usar_hilos: bool = False):

    if usar_hilos:
        print(f"  Conectando a {fuente} ...")
        cam = CameraStream(fuente).start()
        time.sleep(1.5)  # warm-up
    else:
        cam = cv2.VideoCapture(fuente)
        if not cam.isOpened():
            print(f"  ERROR: no se pudo abrir {fuente}")
            return

    fps      = 0.0
    t_fps    = time.time()
    f_cnt    = 0

    print("  Procesando... (Q = salir  |  R = recargar DB)\n")

    while True:
        ret, frame = cam.read() if usar_hilos else cam.read()
        if not ret or frame is None:
            break

        display = frame.copy()
        rostros = motor.procesar_frame(frame)

        for r in rostros:
            ancho = r["res"][0]
            if ancho < TAMANO_MIN_PX:
                continue

            reconocido, sid, dist = galeria.buscar(
                r["embedding"], UMBRAL_RECONOCIMIENTO
            )

            nombre_lbl = _nombre_display(sid) if reconocido else "Desconocido"
            dibujar_rostro(display, r["bbox"], nombre_lbl, dist, reconocido)

        # FPS rolling cada 30 frames
        f_cnt += 1
        if f_cnt % 30 == 0:
            fps   = 30.0 / max(time.time() - t_fps, 1e-6)
            t_fps = time.time()

        dibujar_hud(display, galeria, fps, det_size)

        # Redimensionar para mostrar (no altera el frame procesado)
        h, w = display.shape[:2]
        escala = min(1280 / w, 720 / h)
        if escala < 1.0:
            vis = cv2.resize(display,
                             (int(w * escala), int(h * escala)),
                             interpolation=cv2.INTER_LINEAR)
        else:
            vis = display

        cv2.imshow("Reconocimiento Facial — buffalo_l", vis)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        elif tecla == ord("r"):
            print("\n  Recargando DB ...")
            galeria.cargar()

    if usar_hilos:
        cam.stop()
    else:
        cam.release()
    cv2.destroyAllWindows()


# ==========================================================
# MENÚ
# ==========================================================

def menu():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--det_size", type=int, default=640,
                        help="Resolucion del detector: 640 (rapido) o 1280 (detecta caras pequenas)")
    args, _ = parser.parse_known_args()
    det_size = args.det_size

    print("\n" + "="*52)
    print("  SISTEMA DE RECONOCIMIENTO FACIAL")
    print("  Modelo: buffalo_l")
    print("="*52)

    # Cargar galería en RAM una sola vez
    galeria = GaleriaRAM(DB_PATH)

    if len(galeria) == 0:
        print("\n  AVISO: DB vacía. Registra personas primero.")

    # Cargar motor
    motor = MotorReconocimiento(det_size=det_size)

    while True:
        print(f"\n{'─'*52}")
        print(f"  Personas en RAM : {len(galeria)}")
        print(f"  Umbral L2       : {UMBRAL_RECONOCIMIENTO}")
        print(f"  det_size        : {det_size}x{det_size}")
        print(f"{'─'*52}")
        print("  1. Cámara 1  (IP .110)")
        print("  2. Cámara 2  (IP .111)")
        print("  3. Video local")
        print("  r. Recargar DB desde disco")
        print("  q. Salir")
        print(f"{'─'*52}")

        opc = input("  Opción: ").strip().lower()

        if opc == "1":
            procesar_fuente(RTSP_CAM1, motor, galeria,
                            det_size, usar_hilos=True)

        elif opc == "2":
            procesar_fuente(RTSP_CAM2, motor, galeria,
                            det_size, usar_hilos=True)

        elif opc == "3":
            ruta = input("  Ruta del video: ").strip().strip('"').strip("'")
            if os.path.exists(ruta):
                procesar_fuente(ruta, motor, galeria,
                                det_size, usar_hilos=False)
            else:
                print(f"  ERROR: no encontrado → {ruta}")

        elif opc == "r":
            galeria.cargar()

        elif opc == "q":
            print("  Saliendo.")
            break

        else:
            print("  Opción no válida.")


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    menu()