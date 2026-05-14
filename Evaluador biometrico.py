"""
evaluador_biometrico_pro.py  v2.0  — Pipeline Multihilo + Métricas Avanzadas
=============================================================================
Modelo de concurrencia
──────────────────────
  Hilo LECTOR     → lee/decodifica frames del disco (I/O bound) → frame_queue
  Hilo INFERENCIA → GPU inference + matching (GPU/CPU bound)    ← frame_queue

  Mientras la GPU procesa el frame N, el lector ya tiene el N+1 listo en cola.
  Esto elimina la espera de I/O entre inferencias (~20-35 % más throughput).

Mejoras v2
──────────
  * Umbral de aceptación : 1.25  (igual que reconocedor 2.0)
  * Filtro de Estabilidad: solo cuenta una detección como válida si el mismo
    rostro aparece ≥ N_FRAMES_ESTABILIDAD frames consecutivos.
  * Detecciones Dudosas  : rostros con similitud entre SIM_DUDOSA_MIN y
    SIM_DUDOSA_MAX (zona gris al borde del umbral).
  * Confianza separada   : promedio independiente Empleados / Desconocidos.
  * Apariciones por persona: episodios distintos confirmados por sujeto.
  * Similitud corregida  : sim = max(0, 1 − dist/2)  →  rango real [0,1].
  * CSV escrito en streaming: sin acumulación en RAM para videos largos.

Uso
───
  python evaluador_biometrico_pro.py --videos ./grabaciones --db data/embeddings
  python evaluador_biometrico_pro.py --videos ./grabaciones --db data/embeddings
         --det_size 1280 --skip 2
"""

import cv2
import os
import sys
import csv
import time
import queue
import pickle
import threading
import argparse
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from insightface.app import FaceAnalysis

warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS GLOBALES
# ══════════════════════════════════════════════════════════════════════════════
UMBRAL_CONFIRMAR     = 1.25   # L2 < este valor → reconocido  (igual que reconocedor 2.0)
TAMANO_MIN_PX        = 20     # Ignorar rostros más estrechos (px)
N_FRAMES_ESTABILIDAD = 5      # Mínimo frames consecutivos para contar aparición
MAX_DIST_CENTROS_PX  = 120    # Radio para asociar rostros entre frames
STICKY_FRAMES        = 8      # Frames muertos antes de cerrar un episodio
COLA_MAXSIZE         = 32     # Buffer entre hilo lector e inferencia

# Zona de detecciones dudosas (similitud normalizada 0-1)
SIM_DUDOSA_MIN       = 0.40
SIM_DUDOSA_MAX       = 0.55

SEPARADOR  = "=" * 58
SEPARADOR_S = "─" * 58


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR DE INFERENCIA
# ══════════════════════════════════════════════════════════════════════════════
class MotorEvaluador:
    """
    Envuelve FaceAnalysis (buffalo_l) con det_size configurable.
    Una instancia por hilo de inferencia — no thread-safe internamente.
    """

    def __init__(self, det_size: int = 640):
        size = (det_size, det_size)
        print(f"  Cargando buffalo_l  det_size={size} ...")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=size)
        for m in self.app.models:
            print(f"    sub-modelo: {m}")
        print()

    def procesar_frame(self, frame: np.ndarray) -> list:
        """
        Input : ndarray BGR.
        Output: list de dicts {bbox, embedding, res:(w,h), pose:(pitch,yaw,roll)}
        Solo incluye rostros con ancho >= TAMANO_MIN_PX.
        """
        if frame is None or frame.size == 0:
            return []
        try:
            faces = self.app.get(frame)
        except Exception as exc:
            print(f"\n  [WARN] Error inferencia: {exc}")
            return []

        out = []
        for face in faces:
            bbox  = face.bbox.astype(int)
            ancho = int(bbox[2] - bbox[0])
            if ancho < TAMANO_MIN_PX:
                continue
            try:
                p, y, r = face.pose
            except Exception:
                p, y, r = 0.0, 0.0, 0.0
            emb = face.normed_embedding
            if emb is None:
                continue
            out.append({
                "bbox"     : bbox,
                "embedding": emb,
                "res"      : (ancho, int(bbox[3] - bbox[1])),
                "pose"     : (float(p), float(y), float(r)),
            })
        return out


# ══════════════════════════════════════════════════════════════════════════════
# GALERIA EN RAM
# ══════════════════════════════════════════════════════════════════════════════
class GaleriaEval:
    """
    Carga todos los PKL y expone buscar() con operación vectorizada.
    Compatible con formato v3 (gallery multi-angulo del registro v6).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._galeria: dict[str, np.ndarray] = {}
        self._matrix_global: np.ndarray | None = None
        self._id_map: list[str] = []
        self.cargar()

    def cargar(self):
        self._galeria = {}
        if not os.path.exists(self.db_path):
            print(f"  AVISO: ruta DB no existe -> {self.db_path}")
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
        total_embs = sum(m.shape[0] for m in self._galeria.values())
        print(f"  DB: {ok} personas | {total_embs} embeddings"
              + (f" | {err} errores" if err else ""))
        for sid, mat in self._galeria.items():
            print(f"    {sid:<40} {mat.shape[0]} emb/s")
        print()

    def _extraer(self, data: dict) -> list:
        embs = []
        if "gallery" in data:
            for e in data["gallery"]:
                v = e["embedding"].astype(np.float32)
                n = np.linalg.norm(v)
                if n > 0:
                    embs.append(v / n)
        elif "embedding" in data:
            v = data["embedding"].astype(np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                embs.append(v / n)
        return embs

    def _rebuild(self):
        filas, ids = [], []
        for sid, mat in self._galeria.items():
            for emb in mat:
                filas.append(emb)
                ids.append(sid)
        self._matrix_global = np.array(filas, dtype=np.float32) if filas else None
        self._id_map = ids

    def buscar(self, query: np.ndarray) -> tuple:
        """
        Busqueda vectorizada L2 contra toda la DB en una operacion matricial.
        Retorna (reconocido:bool, sid:str|None, dist_l2:float, similitud:float)
        similitud = max(0, 1 - dist/2)  →  escala lineal [0,1] para L2 en [0,2]
        """
        if self._matrix_global is None:
            return False, None, 2.0, 0.0
        n = np.linalg.norm(query)
        q = query / n if n > 0 else query
        # [PERF]: Una sola operacion matricial — sin bucles Python
        dists = np.linalg.norm(self._matrix_global - q, axis=1)
        idx   = int(np.argmin(dists))
        dist  = float(dists[idx])
        sid   = self._id_map[idx]
        sim   = max(0.0, 1.0 - dist / 2.0)
        return (dist < UMBRAL_CONFIRMAR), sid, dist, sim

    def __len__(self):
        return len(self._galeria)


# ══════════════════════════════════════════════════════════════════════════════
# FILTRO DE ESTABILIDAD — Maquina de Episodios
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class _Track:
    centro          : tuple
    sid             : str | None = None
    dist            : float      = 2.0
    sim             : float      = 0.0
    frames_activos  : int        = 0   # frames consecutivos en el episodio actual
    frames_muertos  : int        = 0   # frames sin detectar esta cara
    episodio_valido : bool       = False  # True una vez que activos >= N_FRAMES
    episodio_contado: bool       = False  # Para no contar el mismo episodio doble


class FiltradorEstabilidad:
    """
    Asocia rostros detectados con tracks persistentes usando centroides.

    Un "episodio" es una secuencia continua de >= N_FRAMES_ESTABILIDAD frames
    donde el mismo rostro aparece. Cada episodio nuevo = una nueva aparicion.

    La funcion procesar() retorna:
      list of (bbox, sid|None, dist, sim, es_valida:bool, es_dudosa:bool)
    """

    def __init__(self):
        self._tracks: list[_Track] = []
        self.apariciones: dict[str, int] = defaultdict(int)

    @staticmethod
    def _centro(bbox) -> tuple:
        x1, y1, x2, y2 = map(int, bbox)
        return (x1 + x2) // 2, (y1 + y2) // 2

    @staticmethod
    def _dist_px(c1, c2) -> float:
        return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))

    def procesar(self, rostros: list, galeria: GaleriaEval) -> list:
        usados = set()
        salida = []

        for r in rostros:
            centro = self._centro(r["bbox"])
            rec, sid, dist, sim = galeria.buscar(r["embedding"])

            # Zona dudosa: similitud en rango SIM_DUDOSA_MIN..SIM_DUDOSA_MAX
            es_dudosa = (SIM_DUDOSA_MIN <= sim <= SIM_DUDOSA_MAX)

            # --- Asociar con track mas cercano (vecino mas proximo) ---
            mejor_idx, mejor_d = None, MAX_DIST_CENTROS_PX + 1
            for i, t in enumerate(self._tracks):
                if i in usados:
                    continue
                d = self._dist_px(centro, t.centro)
                if d < mejor_d:
                    mejor_d, mejor_idx = d, i

            if mejor_idx is not None:
                t = self._tracks[mejor_idx]
                t.centro         = centro
                t.sid            = sid if rec else None
                t.dist           = dist
                t.sim            = sim
                t.frames_activos += 1
                t.frames_muertos  = 0
                usados.add(mejor_idx)
            else:
                # Cara nueva: crear track
                t = _Track(
                    centro=centro,
                    sid=sid if rec else None,
                    dist=dist, sim=sim, frames_activos=1,
                )
                self._tracks.append(t)
                usados.add(len(self._tracks) - 1)

            # --- Validar episodio ---
            if (t.frames_activos >= N_FRAMES_ESTABILIDAD
                    and not t.episodio_contado
                    and rec and sid):
                t.episodio_valido  = True
                t.episodio_contado = True
                self.apariciones[sid] += 1

            salida.append((r["bbox"], t.sid, t.dist, t.sim,
                           t.episodio_valido, es_dudosa))

        # --- Envejecer tracks sin deteccion este frame ---
        for i, t in enumerate(self._tracks):
            if i not in usados:
                t.frames_muertos += 1
                if t.frames_muertos > STICKY_FRAMES:
                    # Cerrar episodio: si reaparece contara como uno nuevo
                    t.episodio_valido  = False
                    t.episodio_contado = False
                    t.frames_activos   = 0

        # Garbage collect
        self._tracks = [
            t for t in self._tracks
            if t.frames_muertos <= STICKY_FRAMES * 3
        ]

        return salida


# ══════════════════════════════════════════════════════════════════════════════
# HILO LECTOR (I/O bound)
# ══════════════════════════════════════════════════════════════════════════════
_SENTINEL = object()  # Token de fin de stream


def _hilo_lector(cap: cv2.VideoCapture, frame_q: queue.Queue,
                 skip: int, stop_evt: threading.Event):
    """
    Lee frames del VideoCapture y los encola para el hilo de inferencia.
    Con skip > 1 descarta frames intermedios sin procesarlos.
    """
    idx = 0
    while not stop_evt.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if skip > 1 and (idx % skip != 0):
            continue
        # Bloqueante con timeout para respetar stop_evt
        while not stop_evt.is_set():
            try:
                frame_q.put((idx, frame), timeout=0.05)
                break
            except queue.Full:
                continue
    frame_q.put(_SENTINEL)


# ══════════════════════════════════════════════════════════════════════════════
# ACUMULADOR DE METRICAS
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Metricas:
    detecciones_brutas  : int   = 0
    empleados_brutos    : int   = 0
    desconocidos_brutos : int   = 0
    empleados_validos   : int   = 0
    desconocidos_validos: int   = 0
    dudosos             : int   = 0
    tamano_min_px       : int   = 9999
    yaw_max_reconocido  : float = 0.0
    pitch_max_reconocido: float = 0.0
    sims_empleados      : list  = field(default_factory=list)
    sims_desconocidos   : list  = field(default_factory=list)
    frames_analizados   : int   = 0
    frames_con_rostros  : int   = 0


# ══════════════════════════════════════════════════════════════════════════════
# PROCESADOR DE VIDEO INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════
def procesar_video(video_path: str, motor: MotorEvaluador,
                   galeria: GaleriaEval, output_dir: str,
                   skip: int) -> dict | None:

    nombre = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n{'='*58}")
    print(f"  VIDEO : {nombre}")
    print(f"{'='*58}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] No se pudo abrir: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"  Frames  : {total_frames}  |  FPS: {fps_video:.1f}"
          f"  |  Duracion: {total_frames/fps_video:.1f}s")
    print(f"  Umbral  : {UMBRAL_CONFIRMAR}  |  "
          f"Estabilidad: {N_FRAMES_ESTABILIDAD} frames  |  "
          f"Skip: {'OFF' if skip <= 1 else skip}\n")

    # Preparar salida
    carpeta_out = os.path.join(output_dir, nombre)
    os.makedirs(carpeta_out, exist_ok=True)
    csv_path = os.path.join(carpeta_out, f"metricas_{nombre}.csv")
    txt_path = os.path.join(carpeta_out, f"reporte_{nombre}.txt")

    # Lanzar hilo lector
    frame_q  = queue.Queue(maxsize=COLA_MAXSIZE)
    stop_evt = threading.Event()
    t_lector = threading.Thread(
        target=_hilo_lector,
        args=(cap, frame_q, skip, stop_evt),
        daemon=True, name=f"Lector-{nombre[:16]}",
    )
    t_lector.start()

    # Estado
    filtro    = FiltradorEstabilidad()
    m         = Metricas()
    t_inicio  = time.time()
    frames_proc = 0

    # Columnas CSV
    columnas = [
        "Frame", "Detecciones_Brutas", "Empleados_Brutos", "Desconocidos_Brutos",
        "Empleados_Validos", "Desconocidos_Validos", "Dudosos",
        "Sim_Prom_Empleados", "Sim_Prom_Desconocidos",
        "Tamano_Min_PX", "Yaw_Max", "Pitch_Max",
    ]

    # Escritura CSV en streaming (no acumula en memoria)
    f_csv = open(csv_path, "w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f_csv, fieldnames=columnas, delimiter=",")
    writer.writeheader()

    try:
        while True:
            try:
                item = frame_q.get(timeout=30)
            except queue.Empty:
                print("\n  [WARN] Timeout 30s — el lector puede haber fallado.")
                break

            if item is _SENTINEL:
                break

            frame_idx, frame = item
            frames_proc += 1
            m.frames_analizados += 1

            # Inferencia GPU
            rostros = motor.procesar_frame(frame)

            # Filtro de estabilidad + matching
            resultados = filtro.procesar(rostros, galeria)

            # --- Metricas del frame ---
            n_brutos = len(resultados)
            n_emp_b = n_desc_b = n_emp_v = n_desc_v = n_dud = 0
            sims_emp_f = []
            sims_desc_f = []
            tam_min_f = yaw_max_f = pitch_max_f = 0
            tam_min_f = 9999

            for i, (bbox, sid, dist, sim, es_valida, es_dudosa) in enumerate(resultados):
                ancho_px  = int(bbox[2] - bbox[0])
                tam_min_f = min(tam_min_f, ancho_px)
                m.tamano_min_px = min(m.tamano_min_px, ancho_px)

                es_empleado = (sid is not None)

                if es_empleado:
                    n_emp_b += 1
                    m.empleados_brutos += 1
                    m.sims_empleados.append(sim)
                    sims_emp_f.append(sim)
                    if es_valida:
                        n_emp_v += 1
                        m.empleados_validos += 1
                    # Angulo del rostro reconocido
                    if i < len(rostros):
                        p_abs = abs(rostros[i]["pose"][0])
                        y_abs = abs(rostros[i]["pose"][1])
                        m.yaw_max_reconocido   = max(m.yaw_max_reconocido, y_abs)
                        m.pitch_max_reconocido = max(m.pitch_max_reconocido, p_abs)
                else:
                    n_desc_b += 1
                    m.desconocidos_brutos += 1
                    m.sims_desconocidos.append(sim)
                    sims_desc_f.append(sim)
                    if es_valida:
                        n_desc_v += 1
                        m.desconocidos_validos += 1

                if es_dudosa:
                    n_dud += 1
                    m.dudosos += 1

            m.detecciones_brutas += n_brutos
            if n_brutos > 0:
                m.frames_con_rostros += 1

            # Angulos maximos del frame (todos los rostros)
            for r in rostros:
                yaw_max_f   = max(yaw_max_f,   abs(r["pose"][1]))
                pitch_max_f = max(pitch_max_f, abs(r["pose"][0]))

            sim_emp_f  = round(float(np.mean(sims_emp_f)),  4) if sims_emp_f  else ""
            sim_desc_f = round(float(np.mean(sims_desc_f)), 4) if sims_desc_f else ""

            writer.writerow({
                "Frame"                : frame_idx,
                "Detecciones_Brutas"   : n_brutos,
                "Empleados_Brutos"     : n_emp_b,
                "Desconocidos_Brutos"  : n_desc_b,
                "Empleados_Validos"    : n_emp_v,
                "Desconocidos_Validos" : n_desc_v,
                "Dudosos"              : n_dud,
                "Sim_Prom_Empleados"   : sim_emp_f,
                "Sim_Prom_Desconocidos": sim_desc_f,
                "Tamano_Min_PX"        : tam_min_f if tam_min_f < 9999 else 0,
                "Yaw_Max"              : round(yaw_max_f, 1),
                "Pitch_Max"            : round(pitch_max_f, 1),
            })

            # Progreso
            if frames_proc % 300 == 0 or frames_proc == 1:
                _progreso(frame_idx, total_frames, t_inicio)

    finally:
        stop_evt.set()
        t_lector.join(timeout=5)
        cap.release()
        f_csv.close()

    _progreso(total_frames, total_frames, t_inicio)
    print()

    # Calculos finales
    sim_emp_g  = round(float(np.mean(m.sims_empleados)),    4) if m.sims_empleados  else 0.0
    sim_desc_g = round(float(np.mean(m.sims_desconocidos)), 4) if m.sims_desconocidos else 0.0
    ang_max    = round(max(m.yaw_max_reconocido, m.pitch_max_reconocido), 1)
    tam_min    = m.tamano_min_px if m.tamano_min_px < 9999 else 0

    reporte = _generar_reporte(
        nombre=nombre, m=m,
        sim_emp_g=sim_emp_g, sim_desc_g=sim_desc_g,
        tam_min=tam_min, ang_max=ang_max,
        apariciones=dict(filtro.apariciones),
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(reporte)
    print(reporte)

    return {
        "nombre"              : nombre,
        "frames_analizados"   : m.frames_analizados,
        "frames_con_rostros"  : m.frames_con_rostros,
        "detecciones_brutas"  : m.detecciones_brutas,
        "empleados_brutos"    : m.empleados_brutos,
        "desconocidos_brutos" : m.desconocidos_brutos,
        "empleados_validos"   : m.empleados_validos,
        "desconocidos_validos": m.desconocidos_validos,
        "dudosos"             : m.dudosos,
        "sim_empleados"       : sim_emp_g,
        "sim_desconocidos"    : sim_desc_g,
        "tamano_min_px"       : tam_min,
        "angulo_max_rec"      : ang_max,
        "apariciones"         : dict(filtro.apariciones),
        "csv_path"            : csv_path,
        "txt_path"            : txt_path,
    }


# ══════════════════════════════════════════════════════════════════════════════
# GENERADORES DE TEXTO
# ══════════════════════════════════════════════════════════════════════════════
def _nombre_display(sid: str) -> str:
    """'Jose_Fernando_Loaiza_Ramirez' -> 'Jose Fernando Loaiza'"""
    return " ".join(sid.split("_")[:3])


def _generar_reporte(nombre, m: Metricas, sim_emp_g, sim_desc_g,
                     tam_min, ang_max, apariciones: dict) -> str:
    tasa = m.frames_con_rostros / max(m.frames_analizados, 1) * 100
    lineas = [
        SEPARADOR,
        f"RESUMEN DE VIDEO: {nombre}",
        SEPARADOR,
        "",
        "RESULTADOS DE CONTEO (DETECCIONES BRUTAS):",
        f"  Rostros detectados totales     : {m.detecciones_brutas}",
        f"  Empleados detectados           : {m.empleados_brutos}",
        f"  Personas desconocidas          : {m.desconocidos_brutos}",
        f"  Detecciones dudosas (zona gris): {m.dudosos}",
        f"    (similitud {SIM_DUDOSA_MIN:.2f}-{SIM_DUDOSA_MAX:.2f} — revisa umbral si es alto)",
        "",
        f"RESULTADOS CON FILTRO DE ESTABILIDAD (>={N_FRAMES_ESTABILIDAD} frames consecutivos):",
        f"  Empleados confirmados          : {m.empleados_validos}",
        f"  Desconocidos confirmados       : {m.desconocidos_validos}",
        "",
        "RENDIMIENTO DE DETECCION:",
        f"  Resolucion minima detectada    : {tam_min} px",
        f"  Angulo maximo reconocido       : {ang_max} grados",
        "",
        "CONFIANZA SEPARADA (similitud 0-1):",
        f"  Similitud promedio Empleados   : {sim_emp_g:.4f}",
        f"  Similitud promedio Desconocidos: {sim_desc_g:.4f}",
        "",
        "COBERTURA:",
        f"  Frames analizados              : {m.frames_analizados}",
        f"  Frames con al menos un rostro  : {m.frames_con_rostros}",
        f"  Tasa de actividad              : {tasa:.1f}%",
        "",
    ]

    if apariciones:
        lineas += [
            "APARICIONES POR PERSONA (episodios distintos confirmados):",
            f"  {'PERSONA':<35} {'APARICIONES':>11}",
            f"  {'─'*35} {'─'*11}",
        ]
        for sid, cnt in sorted(apariciones.items(),
                               key=lambda x: x[1], reverse=True):
            lineas.append(f"  {_nombre_display(sid):<35} {cnt:>11}")
    else:
        lineas.append("  (ningun empleado confirmado en este video)")

    lineas += ["", SEPARADOR, ""]
    return "\n".join(lineas)


def _generar_informe_global(resultados: list, output_dir: str):
    ruta = os.path.join(output_dir, "Informe_Global_Promedios.txt")

    n          = len(resultados)
    tot_frames = sum(r["frames_analizados"]    for r in resultados)
    tot_bru    = sum(r["detecciones_brutas"]   for r in resultados)
    tot_emp_b  = sum(r["empleados_brutos"]     for r in resultados)
    tot_des_b  = sum(r["desconocidos_brutos"]  for r in resultados)
    tot_emp_v  = sum(r["empleados_validos"]    for r in resultados)
    tot_des_v  = sum(r["desconocidos_validos"] for r in resultados)
    tot_dud    = sum(r["dudosos"]              for r in resultados)
    prom_emp   = float(np.mean([r["sim_empleados"]    for r in resultados]))
    prom_desc  = float(np.mean([r["sim_desconocidos"] for r in resultados]))
    min_px     = min((r["tamano_min_px"] for r in resultados if r["tamano_min_px"] > 0),
                     default=0)
    max_ang    = max(r["angulo_max_rec"] for r in resultados)

    # Apariciones acumuladas globales
    ap_global: dict[str, int] = defaultdict(int)
    for r in resultados:
        for sid, cnt in r.get("apariciones", {}).items():
            ap_global[sid] += cnt

    lineas = [
        SEPARADOR,
        "INFORME GLOBAL — SISTEMA BIOMETRICO BUFFALO_L",
        SEPARADOR,
        f"Videos procesados              : {n}",
        f"Frames totales analizados      : {tot_frames}",
        "",
        "TOTALES ACUMULADOS (BRUTOS):",
        f"  Detecciones totales          : {tot_bru}",
        f"  Empleados detectados         : {tot_emp_b}",
        f"  Desconocidos detectados      : {tot_des_b}",
        f"  Dudosos (zona gris)          : {tot_dud}",
        "",
        f"TOTALES CONFIRMADOS (filtro >={N_FRAMES_ESTABILIDAD} frames):",
        f"  Empleados confirmados        : {tot_emp_v}",
        f"  Desconocidos confirmados     : {tot_des_v}",
        "",
        "SIMILITUD PROMEDIO GLOBAL:",
        f"  Empleados                    : {prom_emp:.4f}",
        f"  Desconocidos                 : {prom_desc:.4f}",
        "",
        "LIMITES OPERATIVOS GLOBALES:",
        f"  Tamano minimo detectado      : {min_px} px",
        f"  Angulo maximo reconocido     : {max_ang} grados",
        "",
        SEPARADOR_S,
        "APARICIONES GLOBALES POR PERSONA:",
        SEPARADOR_S,
        f"  {'PERSONA':<35} {'APARICIONES':>11}",
        f"  {'─'*35} {'─'*11}",
    ]
    for sid, cnt in sorted(ap_global.items(), key=lambda x: x[1], reverse=True):
        lineas.append(f"  {_nombre_display(sid):<35} {cnt:>11}")

    lineas += [
        "",
        SEPARADOR_S,
        "DETALLE POR VIDEO:",
        SEPARADOR_S,
        f"  {'VIDEO':<26} {'DET':>7} {'EMP_B':>6} {'DES_B':>6}"
        f" {'EMP_V':>6} {'DUD':>5} {'S_EMP':>6} {'S_DES':>6} {'ANG':>5}",
        f"  {'─'*26} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*5} {'─'*6} {'─'*6} {'─'*5}",
    ]
    for r in resultados:
        nv = r["nombre"][:26]
        lineas.append(
            f"  {nv:<26}"
            f" {r['detecciones_brutas']:>7}"
            f" {r['empleados_brutos']:>6}"
            f" {r['desconocidos_brutos']:>6}"
            f" {r['empleados_validos']:>6}"
            f" {r['dudosos']:>5}"
            f" {r['sim_empleados']:>6.3f}"
            f" {r['sim_desconocidos']:>6.3f}"
            f" {r['angulo_max_rec']:>5.1f}"
        )

    lineas += ["", SEPARADOR, ""]
    texto = "\n".join(lineas)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto)
    print("\n" + texto)
    print(f"  Informe global guardado -> {ruta}\n")


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════
def _progreso(actual: int, total: int, t_ini: float, ancho: int = 44):
    pct    = actual / max(total, 1)
    llenos = int(pct * ancho)
    barra  = "█" * llenos + "░" * (ancho - llenos)
    elap   = time.time() - t_ini
    eta    = (elap / pct * (1 - pct)) if pct > 0 else 0
    print(f"\r  [{barra}] {pct*100:5.1f}%  Frame {actual}/{total}"
          f"  ETA {eta:.0f}s   ", end="", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Evaluador biometrico v2 — pipeline multihilo buffalo_l",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--videos",      default=None,
                        help="Carpeta con los videos a evaluar")
    parser.add_argument("--db",          default="data/embeddings",
                        help="Carpeta PKL de la galeria  (default: data/embeddings)")
    parser.add_argument("--output",      default="data/evaluaciones",
                        help="Carpeta de reportes         (default: data/evaluaciones)")
    parser.add_argument("--det_size",    type=int, default=640,
                        help="Resolucion detector: 640 o 1280  (default: 640)")
    parser.add_argument("--skip",        type=int, default=0,
                        help="Analizar 1 frame de cada N  (0 = todos)")
    parser.add_argument("--extensiones", default="mp4,avi,mov,mkv,ts",
                        help="Formatos separados por coma")
    args = parser.parse_args()

    skip_real = max(args.skip, 1)

    print(f"\n{SEPARADOR}")
    print("  EVALUADOR BIOMETRICO PRO  v2.0  |  buffalo_l")
    print(SEPARADOR)
    print(f"  det_size         : {args.det_size}x{args.det_size}")
    print(f"  Umbral L2        : {UMBRAL_CONFIRMAR}")
    print(f"  Estabilidad      : >={N_FRAMES_ESTABILIDAD} frames consecutivos")
    print(f"  Zona dudosa      : similitud {SIM_DUDOSA_MIN}-{SIM_DUDOSA_MAX}")
    print(f"  Skip             : {skip_real}")
    print(f"  Threading        : Lector (hilo) --[queue]--> Inferencia GPU (main)")

    carpeta_videos = args.videos
    if not carpeta_videos:
        carpeta_videos = input("\n  Carpeta de videos: ").strip().strip('"\'')
    if not os.path.isdir(carpeta_videos):
        print(f"\n  ERROR: no existe -> {carpeta_videos}")
        sys.exit(1)

    exts = tuple(f".{e.strip().lower()}" for e in args.extensiones.split(","))
    videos = sorted([
        os.path.join(carpeta_videos, f)
        for f in os.listdir(carpeta_videos)
        if os.path.splitext(f)[1].lower() in exts
    ])
    if not videos:
        print(f"\n  ERROR: ningun video {exts} en {carpeta_videos}")
        sys.exit(1)

    print(f"\n  Videos encontrados: {len(videos)}")
    for v in videos:
        print(f"    {os.path.basename(v)}")

    print()
    galeria = GaleriaEval(args.db)
    if len(galeria) == 0:
        print("  AVISO: galeria vacia — todos los rostros = Desconocido.")
        if input("  Continuar? [s/n]: ").strip().lower() not in ("s","si","y","yes"):
            sys.exit(0)

    motor = MotorEvaluador(det_size=args.det_size)
    os.makedirs(args.output, exist_ok=True)

    resultados_globales = []
    t_ini_total = time.time()

    for idx, vp in enumerate(videos, 1):
        print(f"\n  [{idx}/{len(videos)}] {os.path.basename(vp)}")
        res = procesar_video(vp, motor, galeria, args.output, skip_real)
        if res:
            resultados_globales.append(res)

    print(f"\n{SEPARADOR}")
    print(f"  Lote completo en {time.time()-t_ini_total:.1f}s")
    print(SEPARADOR)

    if resultados_globales:
        _generar_informe_global(resultados_globales, args.output)
    else:
        print("  AVISO: ningun video procesado correctamente.")

    print(f"  Reportes en: {os.path.abspath(args.output)}\n")


if __name__ == "__main__":
    main()