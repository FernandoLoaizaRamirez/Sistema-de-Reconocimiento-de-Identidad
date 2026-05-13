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
from src.face_engine import FaceEngine
from insightface.app import FaceAnalysis
import warnings

sys.path.insert(0, os.path.dirname(__file__))





warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================================
# AJUSTA AQUÍ SI NECESITAS
# ==========================================================
DB_PATH               = "data/embeddings"
UMBRAL_CONFIRMAR  = 1.00   # L2 para ENTRAR en estado reconocido (estricto)
UMBRAL_PERDER     = 1.40   # L2 para SALIR  de estado reconocido (permisivo)
# La brecha entre ambos (0.25) es el margen que elimina el parpadeo.
# Un frame bueno confirma; hace falta alejarse bastante para perder el nombre.

TAMANO_MIN_PX     = 20     # ignorar rostros más pequeños que esto
STICKY_SEGUNDOS   = 2.5    # tiempo que se mantiene el nombre tras perder detección

RTSP_CAM1 = "rtsp://LoaizaStream:YfPvHbQX68zBCRJ@192.168.30.110:554/Streaming/Channels/101"
RTSP_CAM2 = "rtsp://LoaizaStream:YfPvHbQX68zBCRJ@192.168.30.111:554/Streaming/Channels/101"

COLOR_VERDE = (0, 210, 0)
COLOR_ROJO  = (0, 0, 210)
FONT        = cv2.FONT_HERSHEY_SIMPLEX


# ==========================================================
# GALERÍA EN RAM
# ==========================================================

# [OPERATIVO]: Sistema de gestión de memoria para búsquedas biométricas de alta velocidad.
# [TÉCNICO]: Implementa una arquitectura de búsqueda vectorizada eliminando bucles de Python durante la comparación.
class GaleriaRAM:
    """
    Carga todos los PKL en un diccionario:
      { 'Jose_Fernando': np.array shape (N, 512) }   N = cantidad de embeddings del sujeto

    Al buscar, construye una matriz global (M_total, 512) y hace
    una sola operación np.linalg.norm() — sin loops por persona.
    """

    # [OPERATIVO]: Inicializa los contenedores de datos y lanza la carga automática de la base de datos.
    # [TÉCNICO]: Define el diccionario de galería, la matriz global de NumPy y el mapa de índices para trazabilidad.
    def __init__(self, db_path: str):
        self.db_path = db_path
        # { subject_id: matrix (N, 512) }
        self._galeria: dict[str, np.ndarray] = {}
        # Para búsqueda vectorizada: matriz plana + mapa de índice → subject_id
        self._matrix_global: np.ndarray | None = None
        self._id_map: list[str] = []   # posición i → subject_id
        self.cargar()

    # ------------------------------------------------------------------

    # [OPERATIVO]: Escanea el disco y sube todos los perfiles biométricos a la memoria RAM.
    # [TÉCNICO]: Itera sobre archivos .pkl, extrae vectores y reconstruye el índice matricial para operaciones vectorizadas.
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
                
                # [OPERATIVO]: Asocia el ID del sujeto basándose en el nombre del archivo sanitizado.
                sid = fname.replace("_embedding.pkl", "")
                self._galeria[sid] = np.array(embeddings, dtype=np.float32)
                cargados += 1
            except Exception as e:
                print(f"  Error leyendo {fname}: {e}")
                errores += 1

        # [TÉCNICO]: Genera la matriz global tras finalizar la lectura de archivos individuales.
        self._reconstruir_index()

        total_embs = sum(m.shape[0] for m in self._galeria.values())
        print(f"\n  DB cargada: {cargados} personas | "
              f"{total_embs} embeddings totales en RAM"
              + (f" | {errores} errores" if errores else ""))
        for sid, mat in self._galeria.items():
            print(f"    {sid:<40}  {mat.shape[0]} embedding/s")

    # [OPERATIVO]: Extrae y normaliza los vectores matemáticos de los archivos guardados (soporta versiones v1 y v2).
    # [TÉCNICO]: Normaliza vectores a magnitud 1 (Norma L2) para permitir comparaciones de distancia euclidiana o coseno coherentes.
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

    # [OPERATIVO]: Transforma la estructura de diccionario en una sola matriz gigante para cálculos masivos.
    # [TÉCNICO]: Crea _matrix_global donde M es el total de embeddings de todos los sujetos.
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

    # [OPERATIVO]: Busca la identidad de un rostro nuevo comparándolo contra toda la base de datos simultáneamente.
    # [TÉCNICO]: Input: embedding de consulta. Output: tuple(bool, str, float). Utiliza broadcast de NumPy para calcular distancias L2.
    def buscar(self, query: np.ndarray, umbral: float) -> tuple:
        """
        Búsqueda vectorizada L2 contra toda la DB en una sola operación.
        Retorna (encontrado: bool, subject_id: str | None, distancia: float)
        """
        if self._matrix_global is None:
            return False, None, 2.0

        # [TÉCNICO]: Normalización del query para asegurar consistencia con la matriz global.
        norm = np.linalg.norm(query)
        q    = query / norm if norm > 0 else query

        # [TÉCNICO]: Cálculo de distancias Euclidianas masivo: $$\sqrt{\sum (M_{global} - q)^2}$$.
        dists  = np.linalg.norm(self._matrix_global - q, axis=1)
        idx    = int(np.argmin(dists))
        dist   = float(dists[idx])
        sid    = self._id_map[idx]

        # [OPERATIVO]: Aplica el umbral de decisión para confirmar si la identidad es válida.
        return (dist < UMBRAL_CONFIRMAR), sid, dist

    # [OPERATIVO]: Realiza la búsqueda pero delega la decisión de identidad a un módulo superior (TrackZona).
    # [TÉCNICO]: Retorna la menor distancia encontrada sin aplicar filtros lógicos de confianza.
    def buscar_raw(self, query: np.ndarray) -> tuple:
        """Igual que buscar() pero sin aplicar umbral — lo gestiona TrackZona."""
        if self._matrix_global is None:
            return False, None, 2.0
        norm = np.linalg.norm(query)
        q    = query / norm if norm > 0 else query
        dists = np.linalg.norm(self._matrix_global - q, axis=1)
        idx   = int(np.argmin(dists))
        dist  = float(dists[idx])
        sid   = self._id_map[idx]
        return True, sid, dist

    # [OPERATIVO]: Retorna la cantidad de personas únicas registradas en la galería.
    def __len__(self):
        return len(self._galeria)

# [OPERATIVO]: Filtro de estabilidad para evitar el parpadeo de etiquetas en el stream de video.
# [TÉCNICO]: Implementa una lógica de Histéresis de dos umbrales (Schmitt Trigger) para la gestión de estados de identidad.
class TrackZona:
    """
    Mantiene el estado de identidad de UNA zona del frame (una cara).

    Máquina de estados:
      DESCONOCIDO ──(dist < UMBRAL_CONFIRMAR)──► CONFIRMADO
      CONFIRMADO  ──(dist > UMBRAL_PERDER    )──► DESCONOCIDO
      CONFIRMADO  ──(cara desaparece > STICKY)──► DESCONOCIDO
    """

    # [OPERATIVO]: Inicializa los valores de seguimiento para un objeto facial detectado.
    # [TÉCNICO]: t_ultimo registra el timestamp de la última detección para manejar la persistencia (stickiness).
    def __init__(self):
        self.sid         = None     # subject_id actual (ID del sujeto)
        self.confirmado  = False
        self.dist         = 2.0      # Distancia L2 inicial (infinito práctico)
        self.t_ultimo    = 0.0      # último frame donde se vio esta cara

    # [OPERATIVO]: Procesa la nueva distancia y decide si mantiene, cambia o descarta la identidad.
    # [TÉCNICO]: Input: id candidato, distancia euclidiana, tiempo actual. Aplica lógica de zona muerta (0.90 - 1.15).
    def actualizar(self, sid_candidato: str, dist: float, t_ahora: float):
        self.t_ultimo = t_ahora

        if not self.confirmado:
            # [OPERATIVO]: Estado DESCONOCIDO -> Solo otorga identidad si la confianza es muy alta.
            # [TÉCNICO]: Requiere superar el filtro estricto UMBRAL_CONFIRMAR (ej. 0.90).
            if dist < UMBRAL_CONFIRMAR:
                self.sid         = sid_candidato
                self.confirmado = True
                self.dist       = dist
            else:
                self.sid         = None
                self.dist       = dist
        else:
            # [OPERATIVO]: Estado CONFIRMADO -> Mantiene la identidad aunque la calidad baje ligeramente.
            # [TÉCNICO]: Solo pierde el estado de confirmación si la distancia excede el UMBRAL_PERDER (ej. 1.15).
            if dist > UMBRAL_PERDER:
                self.sid         = None
                self.confirmado = False
                self.dist       = dist
            else:
                # [OPERATIVO]: Zona gris (Histéresis) -> Mantiene el nombre actual para dar estabilidad visual.
                self.dist = dist
                # [TÉCNICO]: Si aparece un candidato con excelente distancia, permite actualizar el puntero de ID.
                if dist < UMBRAL_CONFIRMAR:
                    self.sid = sid_candidato

    # [OPERATIVO]: Determina si el rastro de la cara debe eliminarse por falta de actividad.
    # [TÉCNICO]: Compara el delta de tiempo contra STICKY_SEGUNDOS para liberar memoria de tracking.
    def expirado(self, t_ahora: float) -> bool:
        """True si la cara no apareció en los últimos STICKY_SEGUNDOS."""
        return (t_ahora - self.t_ultimo) > STICKY_SEGUNDOS

    # [OPERATIVO]: Prepara los datos finales para ser renderizados en el HUD de la interfaz.
    # [TÉCNICO]: Propiedad que formatea el nombre y estado de reconocimiento para las funciones de dibujo (OpenCV).
    @property
    def label(self) -> tuple:
        """Retorna (nombre_display, dist, reconocido) listo para dibujar."""
        if self.confirmado and self.sid:
            # [TÉCNICO]: Llama a la utilidad de limpieza de strings para remover guiones bajos.
            return _nombre_display(self.sid), self.dist, True
        return "Desconocido", self.dist, False


# [OPERATIVO]: Gestiona la persistencia de identidad vinculando detecciones físicas con lógica de Tracking.
# [TÉCNICO]: Implementa un seguimiento por proximidad euclidiana (Centroides) sin la carga computacional de SORT o Kalman.
class GestorTracks:
    """
    Asocia cada detección del frame actual con una TrackZona existente
    usando distancia entre centros de bounding box.
    """

    MAX_DIST_PX = 120   # [OPERATIVO]: Radio de búsqueda (si la cara se mueve más de esto entre frames, se considera zona nueva).

    def __init__(self):
        self._tracks: list[tuple] = [] # Almacena (centro_xy, instancia_TrackZona)

    # [TÉCNICO]: Calcula el punto central geométrico de un Bounding Box.
    def _centro(self, bbox) -> tuple:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    # [TÉCNICO]: Teorema de Pitágoras para medir qué tan lejos se movió una cara desde el frame anterior.
    def _dist_centros(self, c1, c2) -> float:
        return float(np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2))

    # [OPERATIVO]: Función principal que coordina el flujo: Detección -> Tracking -> Identificación.
    def procesar(self, rostros: list, galeria: "GaleriaRAM") -> list:
        t_ahora = time.time()
        usados  = set()
        output  = []

        for r in rostros:
            # [OPERATIVO]: Filtro de calidad para ignorar caras demasiado pequeñas/lejanas.
            if r["res"][0] < TAMANO_MIN_PX:
                continue

            centro = self._centro(r["bbox"])

            # [TÉCNICO]: Algoritmo de "Vecino más cercano" para asociar la cara detectada con un track previo.
            mejor_idx, mejor_d = None, self.MAX_DIST_PX + 1
            for i, (c, _track) in enumerate(self._tracks):
                if i in usados: continue
                d = self._dist_centros(centro, c)
                if d < mejor_d:
                    mejor_d, mejor_idx = d, i

            # [OPERATIVO]: Lanza la consulta masiva a la base de datos RAM para obtener el candidato más probable.
            _enc, sid_cand, dist = galeria.buscar_raw(r["embedding"])

            if mejor_idx is not None:
                # [OPERATIVO]: La cara ya existía; actualizamos su posición y refinamos su identidad con la histéresis.
                c_old, track = self._tracks[mejor_idx]
                self._tracks[mejor_idx] = (centro, track)
                track.actualizar(sid_cand, dist, t_ahora)
                usados.add(mejor_idx)
            else:
                # [OPERATIVO]: Es una persona nueva entrando a la escena; creamos un seguidor fresco.
                track = TrackZona()
                track.actualizar(sid_cand, dist, t_ahora)
                self._tracks.append((centro, track))
                usados.add(len(self._tracks) - 1)

            # [TÉCNICO]: Extrae el estado actual (Nombre, Confianza) procesado por el filtro de parpadeo.
            lbl, d_show, rec = track.label
            output.append((r["bbox"], lbl, d_show, rec))

        # [TÉCNICO]: Garbage Collector: Limpia la memoria de caras que salieron de cámara hace X segundos.
        self._tracks = [(c, t) for c, t in self._tracks if not t.expirado(t_ahora)]

        return output

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
    info = [
        f"FPS: {fps:.1f}",
        f"Personas DB: {len(galeria)}",
        f"Confirmar: <{UMBRAL_CONFIRMAR}  Perder: >{UMBRAL_PERDER}",
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
    gestor   = GestorTracks()   # histéresis por zona

    print("  Procesando... (Q = salir  |  R = recargar DB)\n")

    while True:
        ret, frame = cam.read() if usar_hilos else cam.read()
        if not ret or frame is None:
            break

        display = frame.copy()
        rostros = motor.procesar_frame(frame)

        # GestorTracks aplica histéresis y retorna labels estables
        resultados = gestor.procesar(rostros, galeria)
        for bbox, lbl, dist, reconocido in resultados:
            dibujar_rostro(display, bbox, lbl, dist, reconocido)

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
        print(f"  Umbral L2       : {UMBRAL_CONFIRMAR}")
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