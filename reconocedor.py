# Documented version
"""
reconocedor.py — Sistema de Reconocimiento Facial en Tiempo Real
================================================================
Modelo   : buffalo_l  (InsightFace)  vía FaceEngine
Galería  : PKLs cargados en RAM al arrancar — búsqueda vectorizada NumPy
Hardware : GPU (CUDA) con fallback automático a CPU
Display  : verde = reconocido  |  rojo = desconocido

Formato PKL esperado (producido por registro_por_video.py):
    {
        "subject_id"   : str,
        "gallery"      : [ { "embedding": ndarray(512,), ... }, ... ],
        "registered_at": float,
    }

Uso:
    python reconocedor.py
    python reconocedor.py --det_size 1280   # para cámaras lejanas
"""

import argparse
import os
import pickle
import sys
import threading
import time
import warnings

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.face_engine import FaceEngine

warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================================
# CONFIGURACIÓN GLOBAL
# ==========================================================
DB_PATH = "data/embeddings"

# Umbrales de distancia L2 para decisión de identidad.
# Reducir UMBRAL_CONFIRMAR si hay muchos falsos positivos.
# Aumentarlo si personas conocidas no son reconocidas.
UMBRAL_CONFIRMAR = 1.15  # dist < este valor → reconocido

TAMANO_MIN_PX = 20  # Ignorar rostros más pequeños (px de ancho)

RTSP_CAM1 = (
    "rtsp://LoaizaStream:YfPvHbQX68zBCRJ@192.168.30.110:554/Streaming/Channels/101"
)
RTSP_CAM2 = (
    "rtsp://LoaizaStream:YfPvHbQX68zBCRJ@192.168.30.111:554/Streaming/Channels/101"
)

COLOR_VERDE = (0, 210, 0)
COLOR_ROJO = (0, 0, 210)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ==========================================================
# GALERÍA EN RAM — búsqueda vectorizada
# ==========================================================


class GaleriaRAM:
    """Carga y gestiona los embeddings faciales en memoria RAM para búsqueda rápida.

    La búsqueda se realiza con una sola operación matricial de NumPy
    (sin bucles Python), calculando distancias L2 contra todos los
    embeddings de todos los sujetos simultáneamente.

    Attributes:
        db_path (str): Ruta al directorio que contiene los archivos PKL.
        _galeria (dict): Diccionario que mapea subject_id a su matriz de embeddings (N, 512).
        _matrix_global (np.ndarray | None): Matriz única que concatena todos los embeddings.
        _id_map (list): Mapeo de índices de la matriz global a subject_ids.
    """

    def __init__(self, db_path: str):
        """Inicializa la galería y carga los datos desde el disco.

        Args:
            db_path (str): Ruta al directorio con archivos .pkl de embeddings.
        """
        self.db_path = db_path
        self._galeria: dict[str, np.ndarray] = {}  # { sid: matrix (N, 512) }
        self._matrix_global: np.ndarray | None = None
        self._id_map: list[str] = []  # índice i → subject_id
        self.cargar()

    # ------------------------------------------------------------------

    def cargar(self):
        """Lee todos los PKL del directorio y los sube a RAM.

        Busca archivos que terminen en '_embedding.pkl', los carga y extrae
        los embeddings normalizados. Al finalizar, reconstruye el índice global.
        """
        self._galeria = {}

        if not os.path.exists(self.db_path):
            print(f"  AVISO: directorio no encontrado — {self.db_path}")
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
                embeddings = self._leer_gallery(data)
                if not embeddings:
                    print(f"  AVISO: {fname} sin embeddings válidos — omitido")
                    continue
                sid = fname.replace("_embedding.pkl", "")
                self._galeria[sid] = np.array(embeddings, dtype=np.float32)
                cargados += 1
            except Exception as e:
                print(f"  Error leyendo {fname}: {e}")
                errores += 1

        self._reconstruir_index()

        total_embs = sum(m.shape[0] for m in self._galeria.values())
        print(
            f"\n  DB cargada: {cargados} personas | {total_embs} embeddings en RAM"
            + (f" | {errores} errores" if errores else "")
        )
        for sid, mat in self._galeria.items():
            print(f"    {sid:<40}  {mat.shape[0]} embedding/s")

    def _leer_gallery(self, data: dict) -> list[np.ndarray]:
        """Extrae y normaliza los embeddings de un diccionario de datos.

        Args:
            data (dict): Diccionario cargado desde un archivo PKL.

        Returns:
            list[np.ndarray]: Lista de embeddings normalizados L2.
        """
        if "gallery" not in data:
            return []
        embs = []
        for entry in data["gallery"]:
            emb = np.array(entry["embedding"], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                embs.append(emb / norm)
        return embs

    def _reconstruir_index(self):
        """Aplana la galería en una sola matriz para búsqueda vectorizada.

        Crea una matriz global NumPy y un mapa de IDs para permitir
        la búsqueda eficiente mediante operaciones matriciales.
        """
        filas, ids = [], []
        for sid, matrix in self._galeria.items():
            for emb in matrix:
                filas.append(emb)
                ids.append(sid)
        self._matrix_global = np.array(filas, dtype=np.float32) if filas else None
        self._id_map = ids

    # ------------------------------------------------------------------

    def buscar(self, query: np.ndarray) -> tuple[bool, str | None, float]:
        """Realiza una búsqueda vectorizada L2 contra toda la base de datos.

        Args:
            query (np.ndarray): Embedding del rostro a buscar.

        Returns:
            tuple[bool, str | None, float]: Una tupla conteniendo:
                - reconocido (bool): True si la distancia es menor al umbral.
                - subject_id (str | None): ID del sujeto más cercano, o None si no hay datos.
                - distancia_minima (float): La menor distancia L2 encontrada.
        """
        if self._matrix_global is None:
            return False, None, 2.0

        norm = np.linalg.norm(query)
        q = query / norm if norm > 0 else query

        dists = np.linalg.norm(self._matrix_global - q, axis=1)
        idx = int(np.argmin(dists))
        dist = float(dists[idx])
        sid = self._id_map[idx]

        reconocido = dist < UMBRAL_CONFIRMAR
        return reconocido, sid, dist

    def __len__(self) -> int:
        """Retorna el número de personas en la galería.

        Returns:
            int: Cantidad de sujetos únicos cargados.
        """
        return len(self._galeria)


# ==========================================================
# STREAM RTSP SIN LATENCIA ACUMULADA
# ==========================================================


class CameraStream:
    """Captura frames en un hilo secundario para minimizar latencia.

    Asegura que el hilo principal siempre procese el frame más reciente,
    evitando la acumulación en el buffer interno de OpenCV.

    Attributes:
        cap (cv2.VideoCapture): Objeto de captura de video.
        ret (bool): Indica si la última lectura fue exitosa.
        frame (np.ndarray): El frame más reciente capturado.
        stopped (bool): Bandera para detener el hilo de captura.
    """

    def __init__(self, source: str):
        """Inicializa la captura de video.

        Args:
            source (str): Fuente de video (URL RTSP, ruta de archivo o índice de cámara).
        """
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self) -> "CameraStream":
        """Inicia el hilo de lectura de frames.

        Returns:
            CameraStream: La instancia actual de la clase.
        """
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        """Hilo interno que actualiza constantemente el frame más reciente."""
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame

    def read(self) -> tuple[bool, np.ndarray]:
        """Retorna el frame más reciente almacenado.

        Returns:
            tuple[bool, np.ndarray]: Éxito de la lectura y el frame (imagen).
        """
        return self.ret, self.frame

    def stop(self):
        """Detiene la captura y libera los recursos del video."""
        self.stopped = True
        self.cap.release()


# ==========================================================
# DISPLAY
# ==========================================================


def _nombre_display(sid: str) -> str:
    """Formatea el ID del sujeto para mostrar en pantalla.

    Convierte 'Jose_Fernando_Loaiza_Ramirez' → 'Jose Fernando'.

    Args:
        sid (str): ID del sujeto (generalmente el nombre del archivo PKL).

    Returns:
        str: Nombre formateado para la interfaz.
    """
    partes = sid.replace("_embedding", "").split("_")
    return " ".join(partes[:2]) if len(partes) >= 2 else partes[0]


def dibujar_rostro(frame: np.ndarray, bbox, nombre: str, dist: float, reconocido: bool):
    """Dibuja el cuadro delimitador y la etiqueta de identidad.

    Args:
        frame (np.ndarray): Imagen sobre la que se dibujará.
        bbox (list|tuple): Coordenadas del rostro (x1, y1, x2, y2).
        nombre (str): Nombre a mostrar en la etiqueta.
        dist (float): Distancia L2 del reconocimiento.
        reconocido (bool): Si el rostro fue identificado positivamente.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    color = COLOR_VERDE if reconocido else COLOR_ROJO

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    txt = f"{nombre}  {dist:.2f}"
    (tw, th), bl = cv2.getTextSize(txt, FONT, 0.5, 1)
    y_bg = max(y1 - 6, th + 6)
    cv2.rectangle(
        frame, (x1, y_bg - th - 4), (x1 + tw + 6, y_bg + bl), color, cv2.FILLED
    )
    cv2.putText(frame, txt, (x1 + 3, y_bg - 2), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def dibujar_hud(frame: np.ndarray, galeria: GaleriaRAM, fps: float):
    """Dibuja información de estado en el frame.

    Args:
        frame (np.ndarray): Imagen sobre la que se dibujará.
        galeria (GaleriaRAM): Instancia de la galería para obtener estadísticas.
        fps (float): Cuadros por segundo actuales.
    """
    info = [
        f"FPS: {fps:.1f}",
        f"Personas DB: {len(galeria)}",
        f"Confirmar < {UMBRAL_CONFIRMAR}",
        "Q = salir",
    ]
    for i, txt in enumerate(info):
        cv2.putText(
            frame, txt, (8, 20 + i * 18), FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA
        )


# ==========================================================
# LOOP PRINCIPAL DE RECONOCIMIENTO
# ==========================================================


def procesar_fuente(
    fuente: str, engine: FaceEngine, galeria: GaleriaRAM, usar_hilos: bool = False
):
    """Ejecuta el loop de reconocimiento facial sobre una fuente de video.

    Args:
        fuente (str): URL RTSP o ruta de archivo de video.
        engine (FaceEngine): Instancia del motor de reconocimiento.
        galeria (GaleriaRAM): Instancia de la galería con embeddings cargados.
        usar_hilos (bool): True para fuentes RTSP para eliminar latencia.
    """
    if usar_hilos:
        print(f"  Conectando a {fuente} ...")
        cam = CameraStream(fuente).start()
        time.sleep(1.5)
    else:
        cam = cv2.VideoCapture(fuente)
        if not cam.isOpened():
            print(f"  ERROR: no se pudo abrir {fuente}")
            return

    fps = 0.0
    t_fps = time.time()
    f_cnt = 0

    print("  Procesando...  (Q = salir)\n")

    while True:
        ret, frame = cam.read() if usar_hilos else cam.read()
        if not ret or frame is None:
            break

        display = frame.copy()
        rostros = engine.procesar_frame(frame)

        for r in rostros:
            if r["res"][0] < TAMANO_MIN_PX:
                continue

            reconocido, sid, dist = galeria.buscar(r["embedding"])
            nombre = _nombre_display(sid) if reconocido and sid else "Desconocido"
            dibujar_rostro(display, r["bbox"], nombre, dist, reconocido)

        # FPS rolling cada 30 frames
        f_cnt += 1
        if f_cnt % 30 == 0:
            fps = 30.0 / max(time.time() - t_fps, 1e-6)
            t_fps = time.time()

        dibujar_hud(display, galeria, fps)

        # Redimensionar solo para visualización (no afecta el procesamiento)
        h, w = display.shape[:2]
        escala = min(1280 / w, 720 / h)
        vis = (
            cv2.resize(
                display,
                (int(w * escala), int(h * escala)),
                interpolation=cv2.INTER_LINEAR,
            )
            if escala < 1.0
            else display
        )

        cv2.imshow("Reconocimiento Facial — buffalo_l", vis)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break

    if usar_hilos:
        cam.stop()
    else:
        cam.release()
    cv2.destroyAllWindows()


# ==========================================================
# MENÚ INTERACTIVO
# ==========================================================


def menu():
    """Muestra el menú principal y gestiona la interacción del usuario.

    Carga la base de datos, inicializa el motor y permite al usuario elegir
    la fuente de video (Cámaras IP o video local).
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--det_size",
        type=int,
        default=640,
        help="Resolución del detector: 640 (rápido) | 1280 (cámaras lejanas)",
    )
    args, _ = parser.parse_known_args()

    print("\n" + "=" * 52)
    print("  SISTEMA DE RECONOCIMIENTO FACIAL")
    print("  Modelo  : buffalo_l")
    print("  Hardware: GPU (CUDA) + fallback CPU")
    print("=" * 52)

    galeria = GaleriaRAM(DB_PATH)

    if len(galeria) == 0:
        print(
            "\n  AVISO: DB vacía — registra personas primero con registro_por_video.py"
        )

    engine = FaceEngine(det_size=args.det_size)

    while True:
        print(f"\n{'─' * 52}")
        print(f"  Personas en RAM : {len(galeria)}")
        print(f"  Umbral L2       : confirmar < {UMBRAL_CONFIRMAR}")
        print(f"  det_size        : {args.det_size}x{args.det_size}")
        print(f"{'─' * 52}")
        print("  1. Cámara 1  (IP .110)")
        print("  2. Cámara 2  (IP .111)")
        print("  3. Video local")
        print("  q. Salir")
        print(f"{'─' * 52}")

        opc = input("  Opción: ").strip().lower()

        if opc == "1":
            procesar_fuente(RTSP_CAM1, engine, galeria, usar_hilos=True)
        elif opc == "2":
            procesar_fuente(RTSP_CAM2, engine, galeria, usar_hilos=True)
        elif opc == "3":
            ruta = input("  Ruta del video: ").strip().strip('"').strip("'")
            if os.path.exists(ruta):
                procesar_fuente(ruta, engine, galeria, usar_hilos=False)
            else:
                print(f"  ERROR: no encontrado → {ruta}")
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
