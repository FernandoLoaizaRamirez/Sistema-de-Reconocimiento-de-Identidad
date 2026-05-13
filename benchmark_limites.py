"""
benchmark_limites.py — Mide los límites reales del reconocedor
===============================================================
Sin modificar ningún archivo del sistema. Lee los PKLs existentes,
carga los embeddings, y pasa un video de prueba frame a frame midiendo:

  - A qué yaw/pitch el reconocedor empieza a fallar
  - A qué distancia L2 deja de reconocer
  - La tasa de acierto por zona angular
  - El umbral efectivo real vs el umbral configurado

Exporta un CSV con todos los intentos y genera una tabla resumen.

Uso:
  python benchmark_limites.py --video video_prueba.mp4 --db_path data
  python benchmark_limites.py --video video_prueba.mp4 --db_path data --umbral 1.05
  python benchmark_limites.py --video video_prueba.mp4 --db_path data --csv resultados.csv
"""

import cv2
import os
import sys
import pickle
import argparse
import numpy as np
import csv
import time

sys.path.insert(0, os.path.dirname(__file__))
try:
    from src.face_engine import FaceEngine
except ModuleNotFoundError:
    from src.face_engine import FaceEngine


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
PITCH_IDX       = 0
YAW_IDX         = 1
TAMANO_MIN      = 60
NITIDEZ_MIN     = 50.0
UMBRAL_DEFAULT  = 1.05   # misma que UMBRAL_RECONOCIMIENTO en registro_automatico.py

# Bins de yaw para el análisis por zona (en grados)
YAW_BINS  = [(-180,-60), (-60,-40), (-40,-20), (-20,-10), (-10,0),
             (0,10),     (10,20),   (20,40),   (40,60),   (60,180)]
PITCH_BINS= [(-180,-30), (-30,-15), (-15,0), (0,15), (15,30), (30,180)]

YAW_LABELS   = ["<-60","-60a-40","-40a-20","-20a-10","-10a0",
                 "0a10","10a20","20a40","40a60",">60"]
PITCH_LABELS = ["<-30","-30a-15","-15a0","0a15","15a30",">30"]


# ==========================================================
# CARGA DE DB
# ==========================================================

def cargar_embeddings(db_path: str) -> dict:
    """
    Carga todos los pkl de embeddings en un dict:
      { subject_id: [array(512,), ...] }
    Compatible con formato v2 (gallery) y v1 (legacy).
    """
    carpeta = os.path.join(db_path, "embeddings")
    if not os.path.exists(carpeta):
        print(f"ERROR: No se encontro la carpeta {carpeta}")
        sys.exit(1)

    db = {}
    for fname in os.listdir(carpeta):
        if not fname.endswith("_embedding.pkl"):
            continue
        subject_id = fname.replace("_embedding.pkl", "")
        ruta = os.path.join(carpeta, fname)
        try:
            with open(ruta, "rb") as f:
                data = pickle.load(f)

            embeddings = []
            if data.get("version") == 2:
                for entry in data["gallery"]:
                    emb = entry["embedding"]
                    norm = np.linalg.norm(emb)
                    embeddings.append(emb / norm if norm > 0 else emb)
            else:
                emb = data["embedding"]
                norm = np.linalg.norm(emb)
                embeddings.append(emb / norm if norm > 0 else emb)

            db[subject_id] = embeddings
        except Exception as e:
            print(f"  AVISO: Error cargando {fname}: {e}")

    return db


def buscar_en_db(query_emb: np.ndarray, db: dict, umbral: float) -> tuple:
    """
    Busca el sujeto más cercano en la DB usando distancia L2.
    Retorna (encontrado, subject_id, dist_minima)
    """
    norm = np.linalg.norm(query_emb)
    q    = query_emb / norm if norm > 0 else query_emb

    mejor_id, mejor_dist = None, np.inf
    for subject_id, embeddings in db.items():
        matrix = np.array(embeddings)
        dists  = np.linalg.norm(matrix - q, axis=1)
        d_min  = dists.min()
        if d_min < mejor_dist:
            mejor_dist = d_min
            mejor_id   = subject_id

    encontrado = mejor_dist < umbral
    return encontrado, mejor_id, float(mejor_dist)


# ==========================================================
# HELPERS DE POSE
# ==========================================================

def _extraer_pose(rostro) -> tuple:
    pose = rostro.get("pose", (0, 0, 0))
    return float(pose[PITCH_IDX]), float(pose[YAW_IDX])


def _nitidez(frame, bbox) -> float:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if roi.size == 0: return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _bin_idx(valor, bins) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= valor < hi:
            return i
    return len(bins) - 1


# ==========================================================
# BENCHMARK PRINCIPAL
# ==========================================================

def ejecutar_benchmark(video_path: str, db: dict, umbral: float) -> list:
    """
    Procesa el video frame a frame. Por cada rostro detectado que pase
    el filtro mínimo de nitidez y tamaño, intenta reconocerlo en la DB
    y registra el resultado con todos sus metadatos.

    Retorna lista de dicts con un registro por intento.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir: {video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"\n  Video: {os.path.basename(video_path)}")
    print(f"  Frames: {total}  |  FPS: {fps:.1f}  |  Duracion: {total/fps:.1f}s")
    print(f"  Sujetos en DB: {len(db)}  |  Umbral L2: {umbral}\n")

    registros = []
    frame_num = 0
    intentos  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rostros = engine_global.procesar_frame(frame)

        for r in rostros:
            ancho = r["res"][0]
            if ancho < TAMANO_MIN:
                continue
            nit = _nitidez(frame, r["bbox"])
            if nit < NITIDEZ_MIN:
                continue

            pitch, yaw = _extraer_pose(r)
            emb        = r["embedding"]

            t0 = time.perf_counter()
            encontrado, subject_id, dist = buscar_en_db(emb, db, umbral)
            lat_ms = (time.perf_counter() - t0) * 1000

            registros.append({
                "frame"      : frame_num,
                "pitch"      : round(pitch, 2),
                "yaw"        : round(yaw,   2),
                "nitidez"    : round(nit,   1),
                "ancho_px"   : ancho,
                "dist_l2"    : round(dist,  4),
                "reconocido" : int(encontrado),
                "subject_id" : subject_id if encontrado else "DESCONOCIDO",
                "lat_ms"     : round(lat_ms, 3),
            })
            intentos += 1

        if frame_num % 20 == 0:
            pct = int(frame_num / max(total, 1) * 100)
            aciertos = sum(r["reconocido"] for r in registros)
            tasa = aciertos / len(registros) * 100 if registros else 0
            print(f"\r  [{pct:3d}%] {intentos} intentos  tasa={tasa:.1f}%  "
                  f"frame={frame_num}", end="", flush=True)

        frame_num += 1

    cap.release()
    print(f"\r  [100%] {intentos} intentos procesados{' '*30}")
    return registros


# ==========================================================
# ANÁLISIS Y TABLAS
# ==========================================================

def analizar_por_yaw(registros: list):
    """Tasa de reconocimiento agrupada por bin de yaw."""
    bins_data = {label: {"total": 0, "ok": 0, "dist_sum": 0.0}
                 for label in YAW_LABELS}

    for r in registros:
        idx   = _bin_idx(r["yaw"], YAW_BINS)
        label = YAW_LABELS[idx]
        bins_data[label]["total"]    += 1
        bins_data[label]["ok"]       += r["reconocido"]
        bins_data[label]["dist_sum"] += r["dist_l2"]

    print(f"\n  LIMITES POR YAW (rotacion horizontal)")
    print(f"  {'YAW (grados)':>14}  {'INTENTOS':>8}  {'ACIERTOS':>8}  "
          f"{'TASA':>7}  {'DIST_MEDIA':>10}")
    print(f"  {'─'*55}")
    for label in YAW_LABELS:
        d = bins_data[label]
        if d["total"] == 0:
            continue
        tasa      = d["ok"] / d["total"] * 100
        dist_media = d["dist_sum"] / d["total"]
        alerta    = " <-- LIMITE" if tasa < 50 and d["total"] >= 5 else ""
        print(f"  {label:>14}  {d['total']:>8}  {d['ok']:>8}  "
              f"{tasa:>6.1f}%  {dist_media:>10.4f}{alerta}")


def analizar_por_pitch(registros: list):
    """Tasa de reconocimiento agrupada por bin de pitch."""
    bins_data = {label: {"total": 0, "ok": 0, "dist_sum": 0.0}
                 for label in PITCH_LABELS}

    for r in registros:
        idx   = _bin_idx(r["pitch"], PITCH_BINS)
        label = PITCH_LABELS[idx]
        bins_data[label]["total"]    += 1
        bins_data[label]["ok"]       += r["reconocido"]
        bins_data[label]["dist_sum"] += r["dist_l2"]

    print(f"\n  LIMITES POR PITCH (inclinacion vertical)")
    print(f"  {'PITCH (grados)':>14}  {'INTENTOS':>8}  {'ACIERTOS':>8}  "
          f"{'TASA':>7}  {'DIST_MEDIA':>10}")
    print(f"  {'─'*55}")
    for label in PITCH_LABELS:
        d = bins_data[label]
        if d["total"] == 0:
            continue
        tasa      = d["ok"] / d["total"] * 100
        dist_media = d["dist_sum"] / d["total"]
        alerta    = " <-- LIMITE" if tasa < 50 and d["total"] >= 5 else ""
        print(f"  {label:>14}  {d['total']:>8}  {d['ok']:>8}  "
              f"{tasa:>6.1f}%  {dist_media:>10.4f}{alerta}")


def analizar_distribucion_dist(registros: list, umbral: float):
    """Histograma de distancias L2 — muestra dónde está la masa de datos vs umbral."""
    dists = [r["dist_l2"] for r in registros]
    if not dists:
        return

    dists_arr = np.array(dists)
    bins_dist = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.5, 99.0]
    labels_d  = ["<0.3","0.3-0.5","0.5-0.7","0.7-0.8","0.8-0.9","0.9-1.0",
                 "1.0-1.05","1.05-1.1","1.1-1.2","1.2-1.5",">1.5"]

    print(f"\n  DISTRIBUCION DE DISTANCIAS L2  (umbral={umbral})")
    print(f"  {'RANGO L2':>12}  {'FRAMES':>8}  {'%':>6}  RECONOCIDO")
    print(f"  {'─'*45}")
    for i, label in enumerate(labels_d):
        lo = bins_dist[i]
        hi = bins_dist[i+1]
        mask = (dists_arr >= lo) & (dists_arr < hi)
        cnt  = int(mask.sum())
        if cnt == 0:
            continue
        pct_total = cnt / len(dists) * 100
        es_rec    = "SI" if hi <= umbral else ("LIMITE" if lo < umbral <= hi else "NO")
        marcador  = " <--" if lo < umbral <= hi else ""
        print(f"  {label:>12}  {cnt:>8}  {pct_total:>5.1f}%  {es_rec}{marcador}")

    print(f"\n  Distancia minima  : {dists_arr.min():.4f}")
    print(f"  Distancia maxima  : {dists_arr.max():.4f}")
    print(f"  Distancia media   : {dists_arr.mean():.4f}")
    print(f"  Percentil 25      : {np.percentile(dists_arr, 25):.4f}")
    print(f"  Percentil 75      : {np.percentile(dists_arr, 75):.4f}")
    print(f"  Umbral configurado: {umbral}")

    # Umbral efectivo sugerido: el percentil 90 de los frames donde sí reconoció
    dists_ok = dists_arr[[r["reconocido"] == 1 for r in registros]]
    if len(dists_ok) > 0:
        umbral_sugerido = float(np.percentile(dists_ok, 95))
        print(f"  Umbral sugerido (P95 de aciertos): {umbral_sugerido:.4f}")


def resumen_global(registros: list, umbral: float):
    total    = len(registros)
    aciertos = sum(r["reconocido"] for r in registros)
    tasa     = aciertos / total * 100 if total else 0

    print(f"\n{'='*60}")
    print(f"  RESUMEN GLOBAL")
    print(f"{'='*60}")
    print(f"  Total intentos      : {total}")
    print(f"  Reconocidos         : {aciertos} ({tasa:.1f}%)")
    print(f"  No reconocidos      : {total - aciertos} ({100-tasa:.1f}%)")
    print(f"  Umbral L2 usado     : {umbral}")
    if registros:
        lat_media = np.mean([r["lat_ms"] for r in registros])
        print(f"  Latencia media      : {lat_media:.2f} ms por frame")


def exportar_csv(registros: list, ruta_csv: str):
    if not registros:
        return
    campos = list(registros[0].keys())
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(registros)
    print(f"\n  CSV guardado: {ruta_csv}  ({len(registros)} filas)")


# ==========================================================
# ENTRY POINT
# ==========================================================

engine_global = None  # se inicializa en main para no cargar el modelo dos veces

def main():
    global engine_global

    parser = argparse.ArgumentParser(description="Benchmark de limites del reconocedor")
    parser.add_argument("--video",    required=False, default=None,
                        help="Video de prueba")
    parser.add_argument("--db_path",  default="data",
                        help="Carpeta raiz de la DB")
    parser.add_argument("--umbral",   type=float, default=UMBRAL_DEFAULT,
                        help=f"Umbral L2 de reconocimiento (default={UMBRAL_DEFAULT})")
    parser.add_argument("--csv",      default="benchmark_limites.csv",
                        help="Ruta del CSV de salida")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  BENCHMARK DE LIMITES DEL RECONOCEDOR")
    print("="*60)
    print(f"  DB         : {os.path.abspath(args.db_path)}")
    print(f"  Umbral L2  : {args.umbral}")

    video_path = args.video
    if not video_path:
        video_path = input("\n  Ruta del video de prueba: ").strip().strip('"').strip("'")
    if not os.path.exists(video_path):
        print(f"\nERROR: Video no encontrado: {video_path}")
        sys.exit(1)

    # Cargar DB
    print(f"\n  Cargando embeddings...")
    db = cargar_embeddings(args.db_path)
    if not db:
        print("  ERROR: DB vacia. Registra personas primero.")
        sys.exit(1)
    print(f"  {len(db)} sujetos cargados:")
    for sid, embs in db.items():
        print(f"    {sid}  ({len(embs)} embedding/s)")

    # Cargar motor
    print()
    engine_global = FaceEngine()

    # Ejecutar benchmark
    registros = ejecutar_benchmark(video_path, db, args.umbral)

    if not registros:
        print("\n  ERROR: No se obtuvieron datos. Revisa video y DB.")
        sys.exit(1)

    # Análisis
    resumen_global(registros, args.umbral)
    analizar_distribucion_dist(registros, args.umbral)
    analizar_por_yaw(registros)
    analizar_por_pitch(registros)

    # Export
    exportar_csv(registros, args.csv)

    print(f"\n  Abre '{args.csv}' en Excel para analizar frame a frame.")
    print(f"  Columnas: frame, pitch, yaw, nitidez, ancho_px, dist_l2, reconocido\n")


if __name__ == "__main__":
    main()