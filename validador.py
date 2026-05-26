"""
validador_gt.py — Validación contra Ground Truth (formato MOT)
==============================================================
Compara las detecciones del sistema de reconocimiento facial con un archivo
de Ground Truth etiquetado manualmente en formato MOT Challenge.

Responde exactamente a las preguntas del proyecto:
  1. ¿Reconoce siempre a la misma persona cuando ve su rostro?  → ID Accuracy
  2. ¿Aparecen cajas de más (fantasmas)?                        → False Alarm Rate
  3. ¿Cuántas veces no detecta a alguien que sí está?          → Miss Rate
  4. ¿Confunde identidades (error de ID)?                       → ID Switch Rate

Particularidades del GT
───────────────────────
  El GT tiene anotaciones SIEMPRE, incluso cuando la persona está de espaldas.
  El sistema solo puede detectar caras visibles. Por eso diferenciamos:
    - Miss "posible espalda" : GT presente + InsightFace no detecta cara
    - Miss "real"            : GT presente + InsightFace SÍ detecta cara + error de identidad

  Los bbox del GT son de CABEZA (ratio h/w ~ 1.2), por lo que usamos IoU >= 0.15
  como umbral de matching. Si la cara detectada solapa con el bbox del GT,
  son el mismo evento.

Estructura del GT (formato MOT):
  frame_id, track_id, x, y, w, h, 1, 1, 1.0
  Solo se usan: frame_id, track_id, x, y, w, h

Mapeo track_id → nombre (configurable por video):
  Se define en GT_CONFIGS (al final del script) o via --gt_map en CLI.

Uso
───
  python validador_gt.py --video ./cam47.mp4 --gt ./gt_mot_47.txt --video_id 47
  python validador_gt.py --video ./cam48.mp4 --gt ./gt_mot_48.txt --video_id 48
  python validador_gt.py --video ./cam47.mp4 --gt ./gt_mot_47.txt --video_id 47
         --db data/embeddings --output data/validaciones --det_size 640
         --skip 2 --iou_min 0.15

Salida por ejecución:
  data/validaciones/<nombre_video>/
    validacion_frame_<nombre>.csv   ← una fila por evento (GT + predicción + veredicto)
    metricas_<nombre>.csv           ← una fila por frame con conteos agregados
    reporte_validacion_<nombre>.txt ← resumen ejecutivo con KPIs y tabla por persona
"""

import argparse
import csv
import os
import pickle
import queue
import sys
import threading
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
from insightface.app import FaceAnalysis

warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE GT POR VIDEO
# Añade aquí los mapeos de cada video.
# track_ids NO presentes en el dict → tratados como DESCONOCIDO.
# ══════════════════════════════════════════════════════════════════════════════
GT_CONFIGS: dict[str, dict[int, str]] = {
    "47": {
        1: "mitzi_ramirez",
        2: "cesar_angeles",
        3: "rafael_alcantar",
        # Track 4 y 5 no listados → DESCONOCIDO
    },
    "48": {
        1: "jessica_urrea",
        2: "cesar_angeles",
        3: "rafael_alcantar",
        14: "cesar_angeles",
        # Nota: hay un track adicional de cesar_angeles (puede ser 10 u 11).
        # Si lo identificas, agrégalo aquí: ej. 10: "cesar_angeles"
        # El resto → DESCONOCIDO
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS GLOBALES
# ══════════════════════════════════════════════════════════════════════════════
UMBRAL_CONFIRMAR = 1.25  # igual que reconocedor 2.0
TAMANO_MIN_PX = 20
IOU_MATCH_MIN = (
    0.15  # IoU mínimo para considerar que GT y detección son el mismo evento
)
# Valor bajo porque los bbox de GT son cabeza y los del sistema
# pueden estar ligeramente desplazados.
COLA_MAXSIZE = 32
DESCONOCIDO = "DESCONOCIDO"

SEP = "=" * 66
SEPS = "─" * 66


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DEL GROUND TRUTH
# ══════════════════════════════════════════════════════════════════════════════
def cargar_gt(ruta_gt: str, track_map: dict[int, str]) -> dict[int, list]:
    """Lee el archivo Ground Truth (GT) y lo carga en un diccionario indexado por frame.

    El archivo GT debe seguir el formato MOT Challenge:
    frame_id, track_id, x, y, w, h, 1, 1, 1.0

    Args:
        ruta_gt (str): Ruta al archivo de texto del Ground Truth.
        track_map (dict[int, str]): Mapeo de track_id a nombre del sujeto.

    Returns:
        dict[int, list]: Diccionario donde la clave es el frame_id y el valor es una lista
            de tuplas: (track_id, nombre_gt, x1, y1, x2, y2).
    """
    gt: dict[int, list] = defaultdict(list)
    lineas_ok = lineas_err = 0

    with open(ruta_gt, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            try:
                partes = linea.split(",")
                fid = int(partes[0])
                tid = int(partes[1])
                x = float(partes[2])
                y = float(partes[3])
                w = float(partes[4])
                h = float(partes[5])
                nombre_gt = track_map.get(tid, DESCONOCIDO)
                # Convertir a x1,y1,x2,y2
                gt[fid].append((tid, nombre_gt, x, y, x + w, y + h))
                lineas_ok += 1
            except Exception:
                lineas_err += 1

    print(
        f"  GT cargado: {len(gt)} frames | {lineas_ok} anotaciones"
        + (f" | {lineas_err} errores" if lineas_err else "")
    )

    # Resumen de cobertura por track
    por_track: dict[int, int] = defaultdict(int)
    for anots in gt.values():
        for tid, nombre, *_ in anots:
            por_track[tid] += 1
    for tid in sorted(por_track):
        nombre = track_map.get(tid, DESCONOCIDO)
        print(f"    Track {tid:2d} → {nombre:<30} {por_track[tid]:5d} frames")
    print()

    return dict(gt)


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR DE INFERENCIA
# ══════════════════════════════════════════════════════════════════════════════
class MotorValidador:
    """Motor de inferencia para detección y extracción de rostros usando InsightFace.

    Attributes:
        app (FaceAnalysis): Instancia del modelo buffalo_l de InsightFace.
    """

    def __init__(self, det_size: int = 640):
        """Inicializa el modelo de detección y reconocimiento.

        Args:
            det_size (int): Resolución a la que se redimensiona la imagen para detección.
        """
        size = (det_size, det_size)
        print(f"  Cargando buffalo_l  det_size={size} ...")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=size)
        for m in self.app.models:
            print(f"    {m}")
        print()

    def procesar_frame(self, frame: np.ndarray) -> list:
        """Detecta rostros y extrae sus embeddings de un frame.

        Args:
            frame (np.ndarray): Imagen en formato BGR.

        Returns:
            list: Lista de diccionarios, cada uno con 'bbox', 'embedding', 'res' y 'pose'.
        """
        if frame is None or frame.size == 0:
            return []
        try:
            faces = self.app.get(frame)
        except Exception as exc:
            print(f"\n  [WARN] {exc}")
            return []
        out = []
        for face in faces:
            bbox = face.bbox.astype(int)
            ancho = int(bbox[2] - bbox[0])
            if ancho < TAMANO_MIN_PX:
                continue
            emb = face.normed_embedding
            if emb is None:
                continue
            try:
                p, y, r = face.pose
            except Exception:
                p, y, r = 0.0, 0.0, 0.0
            out.append(
                {
                    "bbox": bbox,
                    "embedding": emb,
                    "res": (ancho, int(bbox[3] - bbox[1])),
                    "pose": (float(p), float(y), float(r)),
                }
            )
        return out


# ══════════════════════════════════════════════════════════════════════════════
# GALERÍA EN RAM
# ══════════════════════════════════════════════════════════════════════════════
class GaleriaValidador:
    """Galería de embeddings para el validador con búsqueda vectorizada.

    Attributes:
        db_path (str): Ruta al directorio de embeddings.
        _galeria (dict): Mapeo de sujetos a sus matrices de embeddings.
        _matrix_global (np.ndarray | None): Matriz única de todos los embeddings.
        _id_map (list): Mapa de índices a identidades.
    """

    def __init__(self, db_path: str):
        """Inicializa la galería.

        Args:
            db_path (str): Directorio donde se encuentran los archivos .pkl.
        """
        self.db_path = db_path
        self._galeria: dict[str, np.ndarray] = {}
        self._matrix_global: np.ndarray | None = None
        self._id_map: list[str] = []
        self.cargar()

    def cargar(self):
        """Carga todos los embeddings de la base de datos en memoria RAM."""
        self._galeria = {}
        if not os.path.exists(self.db_path):
            print(f"  AVISO: {self.db_path} no existe")
            self._rebuild()
            return
        ok = err = 0
        for fname in sorted(os.listdir(self.db_path)):
            if not fname.endswith("_embedding.pkl"):
                continue
            try:
                with open(os.path.join(self.db_path, fname), "rb") as f:
                    data = pickle.load(f)
                embs = self._extraer(data)
                if embs:
                    sid = fname.replace("_embedding.pkl", "")
                    self._galeria[sid] = np.array(embs, dtype=np.float32)
                    ok += 1
            except Exception as exc:
                print(f"  [ERR] {fname}: {exc}")
                err += 1
        self._rebuild()
        total = sum(m.shape[0] for m in self._galeria.values())
        print(
            f"  DB: {ok} personas | {total} embeddings"
            + (f" | {err} errores" if err else "")
        )
        for sid, mat in self._galeria.items():
            print(f"    {sid:<40} {mat.shape[0]} emb/s")
        print()

    def _extraer(self, data):
        """Extrae embeddings de un diccionario cargado.

        Args:
            data (dict): Datos del archivo PKL.

        Returns:
            list: Lista de embeddings normalizados.
        """
        embs = []
        for e in data.get("gallery", []):
            v = e["embedding"].astype(np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                embs.append(v / n)
        if not embs and "embedding" in data:
            v = data["embedding"].astype(np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                embs.append(v / n)
        return embs

    def _rebuild(self):
        """Reconstruye el índice global para búsqueda matricial."""
        filas, ids = [], []
        for sid, mat in self._galeria.items():
            for emb in mat:
                filas.append(emb)
                ids.append(sid)
        self._matrix_global = np.array(filas, dtype=np.float32) if filas else None
        self._id_map = ids

    def buscar(self, query: np.ndarray) -> tuple:
        """Busca el sujeto más cercano en la galería.

        Args:
            query (np.ndarray): Embedding a consultar.

        Returns:
            tuple: (reconocido, sid, dist_l2, similitud)
        """
        if self._matrix_global is None:
            return False, None, 2.0, 0.0
        n = np.linalg.norm(query)
        q = query / n if n > 0 else query
        dists = np.linalg.norm(self._matrix_global - q, axis=1)
        idx = int(np.argmin(dists))
        dist = float(dists[idx])
        sid = self._id_map[idx]
        sim = max(0.0, 1.0 - dist / 2.0)
        return (dist < UMBRAL_CONFIRMAR), sid, dist, sim

    def __len__(self):
        """Retorna el número de personas en la galería."""
        return len(self._galeria)


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE MATCHING  (IoU + centro)
# ══════════════════════════════════════════════════════════════════════════════
def _iou(b1, b2) -> float:
    """Calcula el Intersection over Union (IoU) entre dos cajas delimitadoras.

    Args:
        b1 (tuple): Caja 1 (x1, y1, x2, y2).
        b2 (tuple): Caja 2 (x1, y1, x2, y2).

    Returns:
        float: Valor IoU entre 0 y 1.
    """
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centro_en_bbox(centro_xy: tuple, bbox: tuple) -> bool:
    """Determina si un punto central está dentro de una caja delimitadora.

    Args:
        centro_xy (tuple): Coordenadas (cx, cy).
        bbox (tuple): Caja (x1, y1, x2, y2).

    Returns:
        bool: True si el centro está dentro del bbox.
    """
    cx, cy = centro_xy
    x1, y1, x2, y2 = bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _mejor_match_gt(det_bbox, gt_anots: list, iou_min: float) -> tuple | None:
    """Busca la anotación GT que mejor coincide con una detección.

    Criterio: IoU >= iou_min O que el centro de la detección esté dentro del GT bbox.

    Args:
        det_bbox (tuple): Caja de la detección (x1, y1, x2, y2).
        gt_anots (list): Lista de anotaciones GT para el frame.
        iou_min (float): Umbral mínimo de IoU para considerar match.

    Returns:
        tuple | None: (track_id, nombre_gt, gt_bbox, iou_val) o None si no hay match.
    """
    db = det_bbox
    cx = (db[0] + db[2]) / 2
    cy = (db[1] + db[3]) / 2

    mejor_iou = -1.0
    mejor = None

    for tid, nombre_gt, gx1, gy1, gx2, gy2 in gt_anots:
        gb = (gx1, gy1, gx2, gy2)
        iou_val = _iou(db, gb)
        centro_dentro = _centro_en_bbox((cx, cy), gb)
        score = iou_val if not centro_dentro else max(iou_val, iou_min)
        if (iou_val >= iou_min or centro_dentro) and score > mejor_iou:
            mejor_iou = score
            mejor = (tid, nombre_gt, gb, iou_val)

    return mejor


# ══════════════════════════════════════════════════════════════════════════════
# HILO LECTOR
# ══════════════════════════════════════════════════════════════════════════════
_SENTINEL = object()


def _hilo_lector(cap, frame_q, skip, stop_evt):
    """Hilo secundario para lectura de frames de video.

    Args:
        cap (cv2.VideoCapture): Objeto de captura de video.
        frame_q (queue.Queue): Cola donde se depositan los frames leídos.
        skip (int): Número de frames a saltar.
        stop_evt (threading.Event): Evento para detener el hilo.
    """
    idx = 0
    while not stop_evt.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if skip > 1 and (idx % skip != 0):
            continue
        while not stop_evt.is_set():
            try:
                frame_q.put((idx, frame), timeout=0.05)
                break
            except queue.Full:
                continue
    frame_q.put(_SENTINEL)


# ══════════════════════════════════════════════════════════════════════════════
# ACUMULADORES POR PERSONA
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class StatsPersona:
    """Acumulador de estadísticas de validación para un sujeto específico.

    Attributes:
        nombre (str): Nombre del sujeto.
        tp (int): True Positives.
        fp_id (int): False Positives de identidad.
        fn_miss (int): Misses reales (cara detectada, ID incorrecta).
        fn_no_det (int): Misses por no detección (posible espalda).
        id_switches (int): Cantidad de cambios de identidad detectados.
        frames_gt (int): Cantidad de frames donde el sujeto aparece en el GT.
        _prev_pred (str | None): Identidad predicha en el frame anterior.
    """

    nombre: str
    tp: int = 0
    fp_id: int = 0
    fn_miss: int = 0
    fn_no_det: int = 0
    id_switches: int = 0
    frames_gt: int = 0
    _prev_pred: str | None = field(default=None, repr=False)

    def registrar_pred(self, pred: str | None):
        """Detecta cambio de identidad predicha en frames consecutivos.

        Args:
            pred (str | None): Identidad predicha en el frame actual.
        """
        p = pred if pred else DESCONOCIDO
        if self._prev_pred is not None and self._prev_pred != p:
            self.id_switches += 1
        self._prev_pred = p


# ══════════════════════════════════════════════════════════════════════════════
# PROCESADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
def validar_video(
    video_path: str,
    gt_path: str,
    track_map: dict[int, str],
    motor: MotorValidador,
    galeria: GaleriaValidador,
    output_dir: str,
    skip: int,
    iou_min: float,
) -> dict | None:
    """Ejecuta el proceso completo de validación de un video contra el Ground Truth.

    Analiza el video frame a frame, realiza el matching entre detecciones y anotaciones GT,
    calcula métricas de precisión, recall, FAR, etc., y genera reportes detallados.

    Args:
        video_path (str): Ruta al archivo de video.
        gt_path (str): Ruta al archivo de Ground Truth.
        track_map (dict[int, str]): Mapeo de track_id a identidad.
        motor (MotorValidador): Instancia del motor de inferencia.
        galeria (GaleriaValidador): Instancia de la galería de embeddings.
        output_dir (str): Directorio donde se guardarán los resultados.
        skip (int): Cantidad de frames a saltar entre procesamientos.
        iou_min (float): Umbral de IoU para considerar un match exitoso.

    Returns:
        dict | None: Diccionario con los KPIs globales calculados, o None si hubo error.
    """

    nombre = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n{'=' * 66}")
    print(f"  VALIDACIÓN : {nombre}")
    print(f"{'=' * 66}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] No se pudo abrir: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(
        f"  Video: {total_frames} frames | {fps_video:.1f} FPS | "
        f"Skip: {'OFF' if skip <= 1 else skip} | IoU min: {iou_min}\n"
    )

    # Cargar GT
    gt_data = cargar_gt(gt_path, track_map)

    # Identificar personas en el GT y sus categorías
    nombres_conocidos = set(track_map.values()) - {DESCONOCIDO}
    stats_por_nombre: dict[str, StatsPersona] = {}
    for nombre_g in nombres_conocidos:
        stats_por_nombre[nombre_g] = StatsPersona(nombre=nombre_g)
    stats_por_nombre[DESCONOCIDO] = StatsPersona(nombre=DESCONOCIDO)

    # Carpeta de salida
    carpeta_out = os.path.join(output_dir, nombre)
    os.makedirs(carpeta_out, exist_ok=True)
    csv_ev_path = os.path.join(carpeta_out, f"validacion_frame_{nombre}.csv")
    csv_met_path = os.path.join(carpeta_out, f"metricas_{nombre}.csv")
    txt_path = os.path.join(carpeta_out, f"reporte_validacion_{nombre}.txt")

    # Contadores globales
    total_tp = 0  # detección correcta con identidad correcta
    total_fp_id = 0  # identidad incorrecta (confusión entre personas)
    total_fp_ghost = 0  # detección sin GT (persona extra)
    total_fn_miss = 0  # GT presente + cara detectada + identidad incorrecta/desconocida
    total_fn_nodet = 0  # GT presente + no se detectó cara
    total_det = 0  # total de caras detectadas en el video

    # Hilo lector
    frame_q = queue.Queue(maxsize=COLA_MAXSIZE)
    stop_evt = threading.Event()
    t_lector = threading.Thread(
        target=_hilo_lector,
        args=(cap, frame_q, skip, stop_evt),
        daemon=True,
        name="Lector",
    )
    t_lector.start()

    t_inicio = time.time()
    frames_proc = 0

    # Columnas CSV de eventos por frame
    col_ev = [
        "Frame",
        "GT_Track_ID",
        "GT_Nombre",
        "GT_x1",
        "GT_y1",
        "GT_x2",
        "GT_y2",
        "Det_Pred",
        "Det_Sim",
        "Det_x1",
        "Det_y1",
        "Det_x2",
        "Det_y2",
        "IoU",
        "Veredicto",
    ]
    col_met = [
        "Frame",
        "GT_Count",
        "Det_Count",
        "TP",
        "FP_ID",
        "FP_Ghost",
        "FN_Miss",
        "FN_NoDet",
        "Yaw_Max",
        "Pitch_Max",
    ]

    f_ev = open(csv_ev_path, "w", newline="", encoding="utf-8-sig")
    f_met = open(csv_met_path, "w", newline="", encoding="utf-8-sig")
    wr_ev = csv.DictWriter(f_ev, fieldnames=col_ev, delimiter=",")
    wr_met = csv.DictWriter(f_met, fieldnames=col_met, delimiter=",")
    wr_ev.writeheader()
    wr_met.writeheader()

    try:
        while True:
            try:
                item = frame_q.get(timeout=30)
            except queue.Empty:
                print("\n  [WARN] Timeout 30s")
                break
            if item is _SENTINEL:
                break

            frame_idx, frame = item
            frames_proc += 1

            # Detecciones del frame
            rostros = motor.procesar_frame(frame)
            total_det += len(rostros)

            # Anotaciones GT del frame (puede no existir si el frame no tiene GT)
            gt_anots = gt_data.get(frame_idx, [])

            # Conteos del frame
            f_tp = f_fp_id = f_fp_ghost = f_fn_miss = f_fn_nodet = 0
            yaw_max = pitch_max = 0.0
            det_usadas = set()  # índices de detecciones ya asignadas

            # ── Para cada anotación GT, buscar la detección más cercana ──
            for tid, nombre_gt, gx1, gy1, gx2, gy2 in gt_anots:
                if nombre_gt in stats_por_nombre:
                    stats_por_nombre[nombre_gt].frames_gt += 1

                gt_bbox = (gx1, gy1, gx2, gy2)
                mejor_det_idx = None
                mejor_det_iou = -1.0
                mejor_det_info = None

                for i, r in enumerate(rostros):
                    if i in det_usadas:
                        continue
                    db = (
                        float(r["bbox"][0]),
                        float(r["bbox"][1]),
                        float(r["bbox"][2]),
                        float(r["bbox"][3]),
                    )
                    iou_val = _iou(db, gt_bbox)
                    cx = (db[0] + db[2]) / 2
                    cy = (db[1] + db[3]) / 2
                    centro_ok = _centro_en_bbox((cx, cy), gt_bbox)
                    score = iou_val if not centro_ok else max(iou_val, iou_min)
                    if (iou_val >= iou_min or centro_ok) and score > mejor_det_iou:
                        mejor_det_iou = score
                        mejor_det_idx = i
                        mejor_det_info = (db, iou_val)

                if mejor_det_idx is None:
                    # No se detectó cara en esta anotación GT
                    f_fn_nodet += 1
                    total_fn_nodet += 1
                    if nombre_gt in stats_por_nombre:
                        stats_por_nombre[nombre_gt].fn_no_det += 1
                        stats_por_nombre[nombre_gt].registrar_pred(None)
                    wr_ev.writerow(
                        {
                            "Frame": frame_idx,
                            "GT_Track_ID": tid,
                            "GT_Nombre": nombre_gt,
                            "GT_x1": round(gx1, 1),
                            "GT_y1": round(gy1, 1),
                            "GT_x2": round(gx2, 1),
                            "GT_y2": round(gy2, 1),
                            "Det_Pred": "",
                            "Det_Sim": "",
                            "Det_x1": "",
                            "Det_y1": "",
                            "Det_x2": "",
                            "Det_y2": "",
                            "IoU": "",
                            "Veredicto": "FN_NoDet",
                        }
                    )
                    continue

                # Hay matching de detección con este GT
                det_usadas.add(mejor_det_idx)
                r = rostros[mejor_det_idx]
                db = mejor_det_info[0]
                iou_val = mejor_det_info[1]

                rec, sid, dist, sim = galeria.buscar(r["embedding"])
                pred_nombre = sid if rec else DESCONOCIDO

                yaw_max = max(yaw_max, abs(r["pose"][1]))
                pitch_max = max(pitch_max, abs(r["pose"][0]))

                if nombre_gt in stats_por_nombre:
                    stats_por_nombre[nombre_gt].registrar_pred(pred_nombre)

                # Veredicto
                if nombre_gt == DESCONOCIDO:
                    if pred_nombre == DESCONOCIDO:
                        veredicto = "TN"  # correcto rechazo
                    else:
                        veredicto = "FP_ID"  # identificó a un desconocido como empleado
                        f_fp_id += 1
                        total_fp_id += 1
                        if pred_nombre in stats_por_nombre:
                            stats_por_nombre[pred_nombre].fp_id += 1
                else:
                    # GT es un empleado conocido
                    if pred_nombre == nombre_gt:
                        veredicto = "TP"
                        f_tp += 1
                        total_tp += 1
                        stats_por_nombre[nombre_gt].tp += 1
                    else:
                        veredicto = "FN_Miss"  # lo detectó pero no lo reconoció bien
                        f_fn_miss += 1
                        total_fn_miss += 1
                        stats_por_nombre[nombre_gt].fn_miss += 1

                wr_ev.writerow(
                    {
                        "Frame": frame_idx,
                        "GT_Track_ID": tid,
                        "GT_Nombre": nombre_gt,
                        "GT_x1": round(gx1, 1),
                        "GT_y1": round(gy1, 1),
                        "GT_x2": round(gx2, 1),
                        "GT_y2": round(gy2, 1),
                        "Det_Pred": pred_nombre,
                        "Det_Sim": round(sim, 4),
                        "Det_x1": db[0],
                        "Det_y1": db[1],
                        "Det_x2": db[2],
                        "Det_y2": db[3],
                        "IoU": round(iou_val, 4),
                        "Veredicto": veredicto,
                    }
                )

            # ── Detecciones sin GT correspondiente (fantasmas) ──
            for i, r in enumerate(rostros):
                if i in det_usadas:
                    continue
                rec, sid, dist, sim = galeria.buscar(r["embedding"])
                pred_nombre = sid if rec else DESCONOCIDO
                f_fp_ghost += 1
                total_fp_ghost += 1
                db = r["bbox"]
                wr_ev.writerow(
                    {
                        "Frame": frame_idx,
                        "GT_Track_ID": "",
                        "GT_Nombre": "SIN_GT",
                        "GT_x1": "",
                        "GT_y1": "",
                        "GT_x2": "",
                        "GT_y2": "",
                        "Det_Pred": pred_nombre,
                        "Det_Sim": round(sim, 4),
                        "Det_x1": db[0],
                        "Det_y1": db[1],
                        "Det_x2": db[2],
                        "Det_y2": db[3],
                        "IoU": 0.0,
                        "Veredicto": "FP_Ghost",
                    }
                )

            wr_met.writerow(
                {
                    "Frame": frame_idx,
                    "GT_Count": len(gt_anots),
                    "Det_Count": len(rostros),
                    "TP": f_tp,
                    "FP_ID": f_fp_id,
                    "FP_Ghost": f_fp_ghost,
                    "FN_Miss": f_fn_miss,
                    "FN_NoDet": f_fn_nodet,
                    "Yaw_Max": round(yaw_max, 1),
                    "Pitch_Max": round(pitch_max, 1),
                }
            )

            if frames_proc % 200 == 0 or frames_proc == 1:
                _progreso(frame_idx, total_frames, t_inicio)

    finally:
        stop_evt.set()
        t_lector.join(timeout=5)
        cap.release()
        f_ev.close()
        f_met.close()

    _progreso(total_frames, total_frames, t_inicio)
    print()

    # ── KPIs globales ──────────────────────────────────────────────────
    # Total de eventos donde el GT decía que había un empleado Y el sistema
    # detectó una cara (TP + FN_Miss). Los FN_NoDet quedan separados.
    total_detectable = total_tp + total_fn_miss
    precision = total_tp / max(total_tp + total_fp_id, 1)
    recall = total_tp / max(total_detectable, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    far = total_fp_ghost / max(total_det, 1)  # False Alarm Rate
    id_acc = total_tp / max(total_det, 1)  # Identidad correcta / total det

    reporte = _generar_reporte(
        nombre=nombre,
        stats_por_nombre=stats_por_nombre,
        total_tp=total_tp,
        total_fp_id=total_fp_id,
        total_fp_ghost=total_fp_ghost,
        total_fn_miss=total_fn_miss,
        total_fn_nodet=total_fn_nodet,
        total_det=total_det,
        precision=precision,
        recall=recall,
        f1=f1,
        far=far,
        id_acc=id_acc,
        frames_analizados=frames_proc,
        nombres_conocidos=nombres_conocidos,
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(reporte)
    print(reporte)

    return {
        "nombre": nombre,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "far": round(far, 4),
        "id_acc": round(id_acc, 4),
        "total_tp": total_tp,
        "total_fp_id": total_fp_id,
        "total_fp_ghost": total_fp_ghost,
        "total_fn_miss": total_fn_miss,
        "total_fn_nodet": total_fn_nodet,
        "total_det": total_det,
        "stats_por_nombre": {n: vars(s) for n, s in stats_por_nombre.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# GENERADORES DE TEXTO
# ══════════════════════════════════════════════════════════════════════════════
def _nombre_corto(sid: str) -> str:
    """Acorta el subject_id para visualización en tablas.

    Args:
        sid (str): ID del sujeto.

    Returns:
        str: Nombre acortado.
    """
    return " ".join(sid.split("_")[:3])


def _generar_reporte(
    nombre,
    stats_por_nombre,
    total_tp,
    total_fp_id,
    total_fp_ghost,
    total_fn_miss,
    total_fn_nodet,
    total_det,
    precision,
    recall,
    f1,
    far,
    id_acc,
    frames_analizados,
    nombres_conocidos,
) -> str:
    """Genera el contenido de texto del reporte final de validación.

    Args:
        nombre (str): Nombre del video.
        stats_por_nombre (dict): Estadísticas por cada sujeto.
        total_tp (int): Total True Positives.
        total_fp_id (int): Total False Positives de ID.
        total_fp_ghost (int): Total Cajas Fantasma.
        total_fn_miss (int): Total Misses Reales.
        total_fn_nodet (int): Total Misses No Detección.
        total_det (int): Total de detecciones.
        precision (float): Precisión calculada.
        recall (float): Recall calculado.
        f1 (float): F1 Score calculado.
        far (float): False Alarm Rate.
        id_acc (float): Identity Accuracy.
        frames_analizados (int): Frames procesados.
        nombres_conocidos (set): Conjunto de identidades esperadas.

    Returns:
        str: Contenido formateado del reporte.
    """

    total_eventos_conocidos = total_tp + total_fn_miss

    lineas = [
        SEP,
        f"REPORTE DE VALIDACIÓN GT: {nombre}",
        SEP,
        "",
        "GLOSARIO DE VEREDICTOS",
        SEPS,
        "  TP        : GT=empleadoX, sistema dice empleadoX  (correcto)",
        "  FN_Miss   : GT=empleadoX, cara detectada, sistema dice otro/desconocido",
        "  FN_NoDet  : GT=empleadoX, InsightFace NO detectó cara (espalda probable)",
        "  FP_ID     : GT=desconocido, sistema identificó como empleado (falsa alarma de ID)",
        "  FP_Ghost  : sistema detectó cara sin GT correspondiente",
        "  TN        : GT=desconocido, sistema dice desconocido  (correcto rechazo)",
        "",
        "RESUMEN DE EVENTOS",
        SEPS,
        f"  Frames analizados                : {frames_analizados}",
        f"  Caras detectadas (total)         : {total_det}",
        f"  TP   (ID correcta)               : {total_tp}",
        f"  FN   (ID incorrecta, cara visible): {total_fn_miss}",
        f"  FN   (no detectó cara / espalda) : {total_fn_nodet}",
        f"  FP   (confusión de identidad)    : {total_fp_id}",
        f"  FP   (cajas fantasma)            : {total_fp_ghost}",
        "",
        "KPIs GLOBALES",
        SEPS,
        f"  {'KPI':<40} {'VALOR':>8}  INTERPRETACIÓN",
        f"  {'─' * 40} {'─' * 8}  {'─' * 20}",
        f"  {'Precisión  (TP / TP+FP_ID)':<40} {precision * 100:>7.1f}%  "
        f"{'OK' if precision >= 0.85 else 'REVISAR' if precision >= 0.70 else 'CRITICO'}",
        f"  {'Recall     (TP / TP+FN_Miss)':<40} {recall * 100:>7.1f}%  "
        f"{'OK' if recall >= 0.70 else 'REVISAR' if recall >= 0.50 else 'CRITICO'}",
        f"  {'F1 Score':<40} {f1 * 100:>7.1f}%  "
        f"{'OK' if f1 >= 0.75 else 'REVISAR' if f1 >= 0.55 else 'CRITICO'}",
        f"  {'False Alarm Rate  (FP_Ghost/total_det)':<40} {far * 100:>7.1f}%  "
        f"{'OK' if far <= 0.10 else 'REVISAR' if far <= 0.20 else 'CRITICO'}",
        f"  {'ID Accuracy  (TP/total_det)':<40} {id_acc * 100:>7.1f}%  "
        f"{'OK' if id_acc >= 0.60 else 'REVISAR' if id_acc >= 0.40 else 'CRITICO'}",
        "",
        "DIAGNÓSTICO AUTOMÁTICO",
        SEPS,
    ]

    if precision < 0.70:
        lineas.append(
            f"  [PRECISION CRITICA] El sistema confunde identidades frecuentemente.\n"
            f"  -> Reduce UMBRAL_CONFIRMAR (prueba 1.10) o mejora las galerías PKL."
        )
    if recall < 0.50:
        lineas.append(
            f"  [RECALL CRITICO] Muchos rostros visibles no son reconocidos.\n"
            f"  -> El umbral puede ser muy estricto, o faltan ángulos en el PKL.\n"
            f"  -> Verifica también que la cámara tenga buen ángulo y resolución."
        )
    if far > 0.20:
        lineas.append(
            f"  [FAR CRITICO] Demasiadas cajas sin GT (personas no anotadas o ruido).\n"
            f"  -> Revisa si hay personas reales no incluidas en el GT.\n"
            f"  -> Aumenta TAMANO_MIN_PX para ignorar caras muy pequeñas."
        )
    if total_fn_nodet > total_fn_miss:
        lineas.append(
            f"  [AVISO ESPALDA] Los FN_NoDet ({total_fn_nodet}) superan a FN_Miss ({total_fn_miss}).\n"
            f"  -> Muchas personas visibles en GT no tienen cara detectable.\n"
            f"  -> Considera mejorar el ángulo de la cámara para ver más frentes."
        )
    if not any([precision < 0.70, recall < 0.50, far > 0.20]):
        lineas.append("  Todos los KPIs están en rango aceptable.")

    lineas += [
        "",
        "DETALLE POR PERSONA",
        SEPS,
        f"  {'PERSONA':<25} {'GT_F':>6} {'TP':>6} {'FN_M':>6} "
        f"{'FN_ND':>6} {'FP_ID':>6} {'ID_SW':>6} {'Prec':>6} {'Rec':>6}",
        f"  {'─' * 25} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 6}",
    ]
    for nombre_p in sorted(stats_por_nombre.keys()):
        s = stats_por_nombre[nombre_p]
        if s.frames_gt == 0 and s.tp == 0 and s.fp_id == 0:
            continue
        prec_p = s.tp / max(s.tp + s.fp_id, 1)
        rec_p = s.tp / max(s.tp + s.fn_miss, 1)
        lineas.append(
            f"  {_nombre_corto(nombre_p):<25}"
            f" {s.frames_gt:>6}"
            f" {s.tp:>6}"
            f" {s.fn_miss:>6}"
            f" {s.fn_no_det:>6}"
            f" {s.fp_id:>6}"
            f" {s.id_switches:>6}"
            f" {prec_p * 100:>5.1f}%"
            f" {rec_p * 100:>5.1f}%"
        )

    lineas += [
        "",
        "LEYENDA TABLA:",
        "  GT_F   : frames donde el GT dice que aparece esta persona",
        "  TP     : frames reconocida correctamente (cara visible + ID correcta)",
        "  FN_M   : cara visible pero identificación incorrecta",
        "  FN_ND  : GT presente pero cara no detectada (posible espalda)",
        "  FP_ID  : sistema la confundió con otra persona",
        "  ID_SW  : cambios de identidad predicha en frames consecutivos",
        "  Prec   : TP / (TP + FP_ID)",
        "  Rec    : TP / (TP + FN_M)  — solo sobre frames con cara visible",
        "",
        SEP,
        "",
    ]
    return "\n".join(lineas)


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════
def _progreso(actual, total, t_ini, ancho=46):
    """Muestra una barra de progreso en la consola.

    Args:
        actual (int): Progreso actual.
        total (int): Valor total de completitud.
        t_ini (float): Timestamp de inicio para calcular ETA.
        ancho (int): Ancho de la barra en caracteres.
    """
    pct = actual / max(total, 1)
    ll = int(pct * ancho)
    bar = "█" * ll + "░" * (ancho - ll)
    elap = time.time() - t_ini
    eta = (elap / pct * (1 - pct)) if pct > 0 else 0
    print(
        f"\r  [{bar}] {pct * 100:5.1f}%  F{actual}/{total}  ETA {eta:.0f}s  ",
        end="",
        flush=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    """Punto de entrada para el script de validación GT."""
    parser = argparse.ArgumentParser(
        description="Validador GT — compara detecciones con ground truth MOT",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Ruta al video")
    parser.add_argument("--gt", required=True, help="Ruta al archivo GT (.txt MOT)")
    parser.add_argument(
        "--video_id",
        required=True,
        help="ID del video para seleccionar mapeo GT ('47' o '48')",
    )
    parser.add_argument(
        "--db",
        default="data/embeddings",
        help="Carpeta PKL galería  (default: data/embeddings)",
    )
    parser.add_argument(
        "--output",
        default="data/validaciones",
        help="Carpeta de reportes  (default: data/validaciones)",
    )
    parser.add_argument("--det_size", type=int, default=640)
    parser.add_argument(
        "--skip", type=int, default=0, help="Analizar 1 frame de cada N  (0=todos)"
    )
    parser.add_argument(
        "--iou_min",
        type=float,
        default=IOU_MATCH_MIN,
        help=f"IoU mínimo para matching GT↔det  (default: {IOU_MATCH_MIN})",
    )
    # Opción para agregar tracks extra via CLI: --extra_map "10:cesar_angeles,11:cesar_angeles"
    parser.add_argument(
        "--extra_map",
        default="",
        help="Mapeos adicionales: 'track_id:nombre,track_id:nombre'",
    )
    args = parser.parse_args()

    skip_real = max(args.skip, 1)

    print(f"\n{SEP}")
    print("  VALIDADOR GT v1.0  |  buffalo_l")
    print(SEP)

    # Resolver mapeo del GT
    if args.video_id not in GT_CONFIGS:
        print(f"  [WARN] video_id '{args.video_id}' no está en GT_CONFIGS.")
        print(f"  IDs disponibles: {list(GT_CONFIGS.keys())}")
        print(f"  Usando mapeo vacío (todos los tracks → DESCONOCIDO).")
        track_map: dict[int, str] = {}
    else:
        track_map = dict(GT_CONFIGS[args.video_id])

    # Mapeos extra desde CLI
    if args.extra_map:
        for par in args.extra_map.split(","):
            par = par.strip()
            if ":" not in par:
                continue
            tid_str, nombre_extra = par.split(":", 1)
            try:
                track_map[int(tid_str.strip())] = nombre_extra.strip()
                print(f"  Extra map: track {tid_str} → {nombre_extra.strip()}")
            except ValueError:
                pass

    print(f"  Mapeo GT:")
    for tid, nombre in sorted(track_map.items()):
        print(f"    Track {tid:2d} → {nombre}")
    print(f"  IoU mínimo: {args.iou_min}")
    print(f"  Skip: {skip_real}\n")

    # Verificar archivos
    for ruta, etq in [(args.video, "video"), (args.gt, "GT")]:
        if not os.path.exists(ruta):
            print(f"  ERROR: {etq} no encontrado → {ruta}")
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    galeria = GaleriaValidador(args.db)
    motor = MotorValidador(det_size=args.det_size)

    resultado = validar_video(
        video_path=args.video,
        gt_path=args.gt,
        track_map=track_map,
        motor=motor,
        galeria=galeria,
        output_dir=args.output,
        skip=skip_real,
        iou_min=args.iou_min,
    )

    if resultado:
        print(f"\n  Reportes en: {os.path.abspath(args.output)}\n")


if __name__ == "__main__":
    main()
