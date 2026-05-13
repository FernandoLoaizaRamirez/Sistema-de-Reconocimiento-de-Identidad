"""
benchmark_limites.py — Limites del reconocedor (compatible con PKL v4)
=======================================================================
Lee PKLs con galería multi-ángulo (1 PKL por persona) y mide:
  - Tasa de reconocimiento por zona de yaw y pitch
  - Distribución de distancias L2
  - A partir de qué ángulo/distancia falla el reconocedor

Uso:
  python benchmark_limites.py --db_path data
  python benchmark_limites.py --video "C:\\ruta\\completa\\video.mp4" --db_path data
  python benchmark_limites.py --video "C:\\ruta\\video.mp4" --umbral 0.9 --csv out.csv
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
PITCH_IDX      = 0
YAW_IDX        = 1
TAMANO_MIN     = 60
NITIDEZ_MIN    = 50.0
UMBRAL_DEFAULT = 1.05

YAW_BINS   = [(-180,-60),(-60,-40),(-40,-20),(-20,-10),(-10,0),
               (0,10),(10,20),(20,40),(40,60),(60,180)]
YAW_LABELS = ["<-60","-60a-40","-40a-20","-20a-10","-10a0",
               "0a10","10a20","20a40","40a60",">60"]

PITCH_BINS   = [(-180,-30),(-30,-15),(-15,0),(0,15),(15,30),(30,180)]
PITCH_LABELS = ["<-30","-30a-15","-15a0","0a15","15a30",">30"]


# ==========================================================
# CARGA DE DB — formato v4 (1 PKL por persona, galería multi-ángulo)
# ==========================================================

def cargar_db(db_path: str) -> dict:
    """
    Carga todos los PKLs y devuelve:
      { 'Jose_Loaiza': [array(512,), array(512,), ...], ... }

    Compatible con:
      - v4: 1 PKL por persona con gallery de 5 ángulos  ← formato nuevo
      - v2: 1 PKL por ángulo (formato anterior)          ← retrocompatible
      - v1: legacy single embedding                       ← retrocompatible
    """
    carpeta = os.path.join(db_path, "embeddings")
    if not os.path.exists(carpeta):
        print(f"ERROR: No existe la carpeta {carpeta}")
        sys.exit(1)

    db = {}   # { subject_id: [emb1, emb2, ...] }

    for fname in sorted(os.listdir(carpeta)):
        if not fname.endswith("_embedding.pkl"):
            continue
        ruta = os.path.join(carpeta, fname)
        try:
            with open(ruta, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"  AVISO: Error leyendo {fname}: {e}")
            continue

        version = data.get("version", 1)

        if version == 2 and "subject_id" in data:
            # ── Formato v4: PKL con subject_id explícito ──────────────
            subject_id = data["subject_id"]
            embeddings = []
            for entry in data.get("gallery", []):
                emb  = entry["embedding"]
                norm = np.linalg.norm(emb)
                embeddings.append(emb / norm if norm > 0 else emb)

            # Agregar o combinar si el mismo sujeto tiene varios PKLs
            if subject_id not in db:
                db[subject_id] = embeddings
            else:
                db[subject_id].extend(embeddings)

        elif version == 2:
            # ── Formato anterior: 1 PKL por ángulo ────────────────────
            # El subject_id se infiere quitando el sufijo del ángulo del nombre
            raw_id = fname.replace("_embedding.pkl", "")
            # Intentar quitar sufijos de pose conocidos
            sufijos = ["_frontal", "_frontal_arriba", "_frontal_abajo",
                       "_tres_cuartos_izq", "_tres_cuartos_der"]
            subject_id = raw_id
            for suf in sufijos:
                if raw_id.endswith(suf):
                    subject_id = raw_id[:-len(suf)]
                    break

            embeddings = []
            for entry in data.get("gallery", []):
                emb  = entry["embedding"]
                norm = np.linalg.norm(emb)
                embeddings.append(emb / norm if norm > 0 else emb)

            if subject_id not in db:
                db[subject_id] = embeddings
            else:
                db[subject_id].extend(embeddings)

        else:
            # ── Formato v1 legacy ──────────────────────────────────────
            subject_id = fname.replace("_embedding.pkl", "")
            emb  = data["embedding"]
            norm = np.linalg.norm(emb)
            emb  = emb / norm if norm > 0 else emb
            if subject_id not in db:
                db[subject_id] = [emb]
            else:
                db[subject_id].append(emb)

    return db


def buscar_en_db(query_emb: np.ndarray, db: dict, umbral: float) -> tuple:
    norm = np.linalg.norm(query_emb)
    q    = query_emb / norm if norm > 0 else query_emb

    mejor_id, mejor_dist = None, np.inf
    for subject_id, embeddings in db.items():
        matrix = np.array(embeddings, dtype=np.float32)
        dists  = np.linalg.norm(matrix - q, axis=1)
        d_min  = float(dists.min())
        if d_min < mejor_dist:
            mejor_dist = d_min
            mejor_id   = subject_id

    return (mejor_dist < umbral), mejor_id, mejor_dist


# ==========================================================
# HELPERS
# ==========================================================

def _extraer_pose(rostro) -> tuple:
    pose = rostro.get("pose", (0, 0, 0))
    return float(pose[PITCH_IDX]), float(pose[YAW_IDX])


def _nitidez(frame, bbox) -> float:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _bin_idx(valor, bins) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= valor < hi:
            return i
    return len(bins) - 1


# ==========================================================
# BENCHMARK
# ==========================================================

def ejecutar_benchmark(video_path: str, engine, db: dict, umbral: float) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir: {video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"\n  Video: {os.path.basename(video_path)}")
    print(f"  Frames: {total}  |  FPS: {fps:.1f}  |  Duracion: {total/fps:.1f}s")
    print(f"  Personas en DB: {len(db)}  |  Umbral L2: {umbral}")

    # Imprimir embeddings totales por persona
    for sid, embs in db.items():
        print(f"    {sid}  ({len(embs)} embedding/s)")

    registros = []
    frame_num = intentos = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rostros = engine.procesar_frame(frame)

        for r in rostros:
            if r["res"][0] < TAMANO_MIN:
                continue
            nit = _nitidez(frame, r["bbox"])
            if nit < NITIDEZ_MIN:
                continue

            pitch, yaw = _extraer_pose(r)

            t0 = time.perf_counter()
            encontrado, subject_id, dist = buscar_en_db(r["embedding"], db, umbral)
            lat_ms = (time.perf_counter() - t0) * 1000

            registros.append({
                "frame"      : frame_num,
                "pitch"      : round(pitch, 2),
                "yaw"        : round(yaw,   2),
                "nitidez"    : round(nit,   1),
                "ancho_px"   : r["res"][0],
                "dist_l2"    : round(dist,  4),
                "reconocido" : int(encontrado),
                "subject_id" : subject_id if encontrado else "DESCONOCIDO",
                "lat_ms"     : round(lat_ms, 3),
            })
            intentos += 1

        if frame_num % 20 == 0:
            pct      = int(frame_num / max(total, 1) * 100)
            aciertos = sum(r["reconocido"] for r in registros)
            tasa     = aciertos / len(registros) * 100 if registros else 0
            print(f"\r  [{pct:3d}%] {intentos} intentos  tasa={tasa:.1f}%",
                  end="", flush=True)

        frame_num += 1

    cap.release()
    print(f"\r  [100%] {intentos} intentos procesados{' '*30}")
    return registros


# ==========================================================
# ANÁLISIS
# ==========================================================

def resumen_global(registros, umbral):
    total    = len(registros)
    aciertos = sum(r["reconocido"] for r in registros)
    tasa     = aciertos / total * 100 if total else 0
    lat_med  = float(np.mean([r["lat_ms"] for r in registros])) if registros else 0

    print(f"\n{'='*60}")
    print(f"  RESUMEN GLOBAL")
    print(f"{'='*60}")
    print(f"  Total intentos   : {total}")
    print(f"  Reconocidos      : {aciertos}  ({tasa:.1f}%)")
    print(f"  No reconocidos   : {total - aciertos}  ({100-tasa:.1f}%)")
    print(f"  Umbral L2        : {umbral}")
    print(f"  Latencia media   : {lat_med:.2f} ms")


def analizar_distribucion_dist(registros, umbral):
    dists = np.array([r["dist_l2"] for r in registros])
    if len(dists) == 0:
        return

    bins_d  = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.5, 99.0]
    labels_d = ["<0.3","0.3-0.5","0.5-0.7","0.7-0.8","0.8-0.9","0.9-1.0",
                "1.0-1.05","1.05-1.1","1.1-1.2","1.2-1.5",">1.5"]

    print(f"\n  DISTRIBUCION DE DISTANCIAS L2  (umbral={umbral})")
    print(f"  {'RANGO':>12}  {'FRAMES':>7}  {'%':>6}  ESTADO")
    print(f"  {'─'*42}")

    for i, label in enumerate(labels_d):
        lo = bins_d[i]; hi = bins_d[i+1]
        cnt = int(((dists >= lo) & (dists < hi)).sum())
        if cnt == 0:
            continue
        pct = cnt / len(dists) * 100
        if hi <= umbral:
            estado = "RECONOCIDO"
        elif lo < umbral <= hi:
            estado = "LIMITE <--"
        else:
            estado = "RECHAZADO "
        print(f"  {label:>12}  {cnt:>7}  {pct:>5.1f}%  {estado}")

    print(f"\n  Min={dists.min():.4f}  Max={dists.max():.4f}  "
          f"Media={dists.mean():.4f}  P75={np.percentile(dists,75):.4f}")

    dists_ok = dists[[r["reconocido"] == 1 for r in registros]]
    if len(dists_ok) > 0:
        print(f"  Umbral sugerido (P95 aciertos): {np.percentile(dists_ok, 95):.4f}")


def analizar_por_yaw(registros):
    acum = {l: {"t": 0, "ok": 0, "ds": 0.0} for l in YAW_LABELS}
    for r in registros:
        l = YAW_LABELS[_bin_idx(r["yaw"], YAW_BINS)]
        acum[l]["t"]  += 1
        acum[l]["ok"] += r["reconocido"]
        acum[l]["ds"] += r["dist_l2"]

    print(f"\n  LIMITES POR YAW")
    print(f"  {'YAW':>12}  {'TOTAL':>7}  {'ACIERTOS':>8}  {'TASA':>7}  {'DIST_MED':>9}")
    print(f"  {'─'*50}")
    for l in YAW_LABELS:
        d = acum[l]
        if d["t"] == 0:
            continue
        tasa = d["ok"] / d["t"] * 100
        dm   = d["ds"] / d["t"]
        alerta = "  <-- LIMITE" if tasa < 50 and d["t"] >= 5 else ""
        print(f"  {l:>12}  {d['t']:>7}  {d['ok']:>8}  {tasa:>6.1f}%  {dm:>9.4f}{alerta}")


def analizar_por_pitch(registros):
    acum = {l: {"t": 0, "ok": 0, "ds": 0.0} for l in PITCH_LABELS}
    for r in registros:
        l = PITCH_LABELS[_bin_idx(r["pitch"], PITCH_BINS)]
        acum[l]["t"]  += 1
        acum[l]["ok"] += r["reconocido"]
        acum[l]["ds"] += r["dist_l2"]

    print(f"\n  LIMITES POR PITCH")
    print(f"  {'PITCH':>12}  {'TOTAL':>7}  {'ACIERTOS':>8}  {'TASA':>7}  {'DIST_MED':>9}")
    print(f"  {'─'*50}")
    for l in PITCH_LABELS:
        d = acum[l]
        if d["t"] == 0:
            continue
        tasa = d["ok"] / d["t"] * 100
        dm   = d["ds"] / d["t"]
        alerta = "  <-- LIMITE" if tasa < 50 and d["t"] >= 5 else ""
        print(f"  {l:>12}  {d['t']:>7}  {d['ok']:>8}  {tasa:>6.1f}%  {dm:>9.4f}{alerta}")


def exportar_csv(registros, ruta):
    if not registros:
        return
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(registros[0].keys()))
        writer.writeheader()
        writer.writerows(registros)
    print(f"\n  CSV exportado: {ruta}  ({len(registros)} filas)")
    print(f"  Columnas: frame, pitch, yaw, nitidez, ancho_px, dist_l2, reconocido, subject_id, lat_ms")


# ==========================================================
# ENTRY POINT
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   default=None,
                        help="Ruta completa al video de prueba")
    parser.add_argument("--db_path", default="data")
    parser.add_argument("--umbral",  type=float, default=UMBRAL_DEFAULT)
    parser.add_argument("--csv",     default="benchmark_limites.csv")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  BENCHMARK DE LIMITES DEL RECONOCEDOR")
    print("="*60)
    print(f"  DB        : {os.path.abspath(args.db_path)}")
    print(f"  Umbral L2 : {args.umbral}")

    # Pedir video si no se pasó por argumento
    video_path = args.video
    if not video_path:
        print("\n  IMPORTANTE: escribe la ruta COMPLETA al video.")
        print(r"  Ejemplo:  C:\Users\megag\Documents\Proyecto Practicas\video.mp4")
        video_path = input("\n  Ruta del video: ").strip().strip('"').strip("'")

    if not os.path.exists(video_path):
        print(f"\n  ERROR: Video no encontrado en:\n  {video_path}")
        print(f"\n  Comprueba que la ruta es correcta y que el archivo existe.")
        sys.exit(1)

    # Cargar DB
    print(f"\n  Cargando DB...")
    db = cargar_db(args.db_path)
    if not db:
        print("  ERROR: DB vacia.")
        sys.exit(1)

    # Cargar motor
    print()
    engine = FaceEngine()

    # Benchmark
    registros = ejecutar_benchmark(video_path, engine, db, args.umbral)

    if not registros:
        print("\n  ERROR: Sin datos. Verifica el video y la DB.")
        sys.exit(1)

    # Análisis
    resumen_global(registros, args.umbral)
    analizar_distribucion_dist(registros, args.umbral)
    analizar_por_yaw(registros)
    analizar_por_pitch(registros)
    exportar_csv(registros, args.csv)

    print(f"\n  Listo. Abre '{args.csv}' en Excel para analisis detallado.\n")


if __name__ == "__main__":
    main()