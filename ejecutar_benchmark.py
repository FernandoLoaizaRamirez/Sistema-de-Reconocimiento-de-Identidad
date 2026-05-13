"""
ejecutar_benchmark.py — Benchmark de Reconocimiento Biométrico Gold v2
=======================================================================
Tres responsabilidades:
  1. Benchmark de DETECCIÓN pura (herencia del script original).
  2. Benchmark de RECONOCIMIENTO con N templates (1 vs galería completa).
  3. Exportación de resultados a CSV para análisis externo.

Uso:
  python ejecutar_benchmark.py \
      --videos video1.mp4 video2.mp4 \
      --db_path data/embeddings \
      --modelo BUFFALO_L \
      --max_templates 5           # hasta cuántos templates probar (1..5)
      --output_csv resultados_benchmark.csv
"""

import cv2
import pandas as pd
import numpy as np
import os
import time
import argparse
import pickle
import copy
from src.face_engine import FaceEngine
from src.face_library import FaceLibrary

# ==========================================================
# CONFIGURACIÓN — espejo de registro_automatico.py
# ==========================================================
UMBRAL_RECONOCIMIENTO  = 1.05
FILTRO_TAMANO_MIN_PX   = 110
FILTRO_YAW_ESTRICTO    = 15.0
FILTRO_PITCH_MIN       = -40.0
FILTRO_PITCH_MAX       = 25.0
FILTRO_NITIDEZ_MIN     = 280.0
FILTRO_BRIGHTNESS_MIN  = 85.0
FILTRO_BRIGHTNESS_MAX  = 220.0


# ==========================================================
# HELPERS
# ==========================================================

def calcular_calidad_imagen(frame, bbox):
    """Brillo (luma media) y nitidez (varianza de Laplaciano) del ROI facial."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if roi.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pasa_qc(ancho, yaw, pitch, nitidez, brillo, cortada) -> tuple:
    """
    Aplica todos los filtros QC (incluidos DT-01 y DT-02 ya corregidos).
    Retorna (pasa: bool, motivo_rechazo: str).
    """
    if cortada:
        return False, "CORTADA"
    if ancho < FILTRO_TAMANO_MIN_PX:
        return False, "TAMAÑO"
    if abs(yaw) > FILTRO_YAW_ESTRICTO:
        return False, "YAW"
    if not (FILTRO_PITCH_MIN <= pitch <= FILTRO_PITCH_MAX):
        return False, "PITCH"
    if nitidez < FILTRO_NITIDEZ_MIN:
        return False, "NITIDEZ"
    if not (FILTRO_BRIGHTNESS_MIN <= brillo <= FILTRO_BRIGHTNESS_MAX):
        return False, "BRILLO"
    return True, "OK"


def construir_libreria_n_templates(library_full: FaceLibrary, n: int) -> FaceLibrary:
    """
    Crea una FaceLibrary sintética con solo los primeros N templates
    de cada sujeto (ordenados por score descendente).

    Esto permite simular el rendimiento con 1, 2, 3, 4 y 5 templates
    sin necesidad de vídeos adicionales.
    """
    lib_n = FaceLibrary.__new__(FaceLibrary)
    lib_n.path    = library_full.path
    lib_n.face_db = {}

    for subject_id, record in library_full.face_db.items():
        gallery_sorted = sorted(record['gallery'], key=lambda e: e['score'], reverse=True)
        lib_n.face_db[subject_id] = {
            'gallery'     : copy.deepcopy(gallery_sorted[:n]),
            'update_count': record.get('update_count', 0),
        }
    return lib_n


# ==========================================================
# BENCHMARK 1 — DETECCIÓN PURA (preserva lógica original)
# ==========================================================

def benchmark_deteccion(video_path: str, engine: FaceEngine, modelo_label: str) -> pd.DataFrame:
    """
    Mide la capacidad del motor para DETECTAR rostros frame a frame,
    independientemente de si hay identidades registradas en la DB.
    Hereda y extiende la lógica del script original.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ⚠️  No se pudo abrir: {video_path}")
        return pd.DataFrame()

    log = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"  📹 Detección: {os.path.basename(video_path)} | {total_frames} frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0     = time.perf_counter()
        rostros = engine.procesar_frame(frame)
        lat_ms  = (time.perf_counter() - t0) * 1000

        if not rostros:
            log.append({
                "frame": frame_count, "modelo": modelo_label,
                "status": "PERDIDO", "latencia_ms": lat_ms,
                "yaw": None, "pitch": None,
                "brillo": None, "nitidez": None,
                "dist_px": None, "qc_resultado": None,
            })
        else:
            for r in rostros:
                x1, y1, x2, y2 = [int(v) for v in r['bbox']]
                p_val, y_val, _ = r.get('pose', (0, 0, 0))
                brillo, nitidez = calcular_calidad_imagen(frame, r['bbox'])
                ancho = x2 - x1
                cortada = (y1 <= 15 or y2 >= h_video - 15 or
                           x1 <= 15 or x2 >= w_video - 15)

                _, qc_motivo = pasa_qc(ancho, y_val, p_val, nitidez, brillo, cortada)

                log.append({
                    "frame": frame_count, "modelo": modelo_label,
                    "status": "DETECTADO", "latencia_ms": lat_ms,
                    "yaw": round(y_val, 2), "pitch": round(p_val, 2),
                    "brillo": round(brillo, 1), "nitidez": round(nitidez, 1),
                    "dist_px": ancho, "qc_resultado": qc_motivo,
                })

        frame_count += 1

    cap.release()
    return pd.DataFrame(log) if log else pd.DataFrame()


# ==========================================================
# BENCHMARK 2 — RECONOCIMIENTO MULTI-TEMPLATE
# ==========================================================

def benchmark_reconocimiento(
    video_path : str,
    engine     : FaceEngine,
    library    : FaceLibrary,
    n_templates: int,
) -> pd.DataFrame:
    """
    Para cada frame con rostro detectado que pase el QC Gate completo,
    intenta reconocer usando una librería limitada a N templates por sujeto.

    Métricas registradas por intento:
      - reconocido (bool)
      - distancia L2 al match más cercano
      - subject_id matcheado
      - n_templates usados en la comparación
    """
    lib_n = construir_libreria_n_templates(library, n_templates)

    if not lib_n.face_db:
        print(f"  ⚠️  DB vacía para N={n_templates}. Skipping.")
        return pd.DataFrame()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return pd.DataFrame()

    h_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    log = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rostros = engine.procesar_frame(frame)

        for r in rostros:
            x1, y1, x2, y2 = [int(v) for v in r['bbox']]
            p_val, y_val, _ = r.get('pose', (0, 0, 0))
            brillo, nitidez = calcular_calidad_imagen(frame, r['bbox'])
            ancho = x2 - x1
            cortada = (y1 <= 15 or y2 >= h_video - 15 or
                       x1 <= 15 or x2 >= w_video - 15)

            qc_pass, _ = pasa_qc(ancho, y_val, p_val, nitidez, brillo, cortada)
            if not qc_pass:
                continue  # Solo probamos reconocimiento en frames de calidad

            t0 = time.perf_counter()
            encontrado, subject_id, distancia = lib_n.buscar(
                r['embedding'], UMBRAL_RECONOCIMIENTO
            )
            lat_ms = (time.perf_counter() - t0) * 1000

            log.append({
                "frame"       : frame_count,
                "n_templates" : n_templates,
                "reconocido"  : encontrado,
                "distancia_l2": round(float(distancia), 4),
                "subject_id"  : subject_id if encontrado else "DESCONOCIDO",
                "latencia_ms" : round(lat_ms, 3),
                "yaw"         : round(y_val, 2),
                "nitidez"     : round(nitidez, 1),
            })

        frame_count += 1

    cap.release()
    return pd.DataFrame(log) if log else pd.DataFrame()


# ==========================================================
# INFORME DE CONSOLA (hereda estilo del script original)
# ==========================================================

def imprimir_informe_deteccion(df: pd.DataFrame, modelo_label: str):
    det = df[df['status'] == "DETECTADO"]
    print(f"\n{'='*45}")
    print(f"  INFORME TÉCNICO — {modelo_label}")
    print(f"{'='*45}")
    print(f"  FRAMES TOTALES  : {len(df)}")
    print(f"  TASA DETECCIÓN  : {len(det)/len(df)*100:.1f}%" if len(df) else "  Sin datos")

    if not det.empty:
        qc_ok = det[det['qc_resultado'] == 'OK']
        print(f"  FRAMES PASAN QC : {len(qc_ok)} ({len(qc_ok)/len(det)*100:.1f}% de detectados)")
        print(f"  LATENCIA PROM.  : {df['latencia_ms'].mean():.2f} ms")
        print(f"  FPS ESTIMADOS   : {1000/df['latencia_ms'].mean():.1f}")
        print(f"\n  COBERTURA YAW   : {det['yaw'].min():.1f}° .. {det['yaw'].max():.1f}°")
        print(f"  COBERTURA PITCH : {det['pitch'].min():.1f}° .. {det['pitch'].max():.1f}°")
        print(f"\n  BRILLO (luma)   : prom={det['brillo'].mean():.1f}  min={det['brillo'].min():.1f}")
        print(f"  NITIDEZ (lap)   : prom={det['nitidez'].mean():.1f}  min={det['nitidez'].min():.1f}")
        print(f"  TAMAÑO MÍNIMO   : {det['dist_px'].min()} px")

        rechazos = det[det['qc_resultado'] != 'OK']['qc_resultado'].value_counts()
        if not rechazos.empty:
            print(f"\n  RECHAZOS QC:")
            for motivo, cnt in rechazos.items():
                print(f"    {motivo:12s}: {cnt}")

    print(f"{'='*45}")


def imprimir_informe_reconocimiento(resultados_n: dict):
    print(f"\n{'='*45}")
    print(f"  COMPARATIVA MULTI-TEMPLATE")
    print(f"{'='*45}")
    print(f"  {'N templates':>12} | {'Tasa recono.':>12} | {'Dist. prom':>10} | {'Latencia':>9}")
    print(f"  {'-'*50}")
    for n, df in sorted(resultados_n.items()):
        if df.empty:
            continue
        tasa = df['reconocido'].mean() * 100
        dist = df['distancia_l2'].mean()
        lat  = df['latencia_ms'].mean()
        print(f"  {n:>12} | {tasa:>11.1f}% | {dist:>10.4f} | {lat:>7.3f} ms")
    print(f"{'='*45}")


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Biometría Gold v2")
    parser.add_argument("--videos",        nargs="+", default=["Video angulos.mp4"])
    parser.add_argument("--db_path",       default="data/embeddings")
    parser.add_argument("--modelo",        default="BUFFALO_L")
    parser.add_argument("--max_templates", type=int, default=5)
    parser.add_argument("--output_csv",    default="resultados_benchmark.csv")
    args = parser.parse_args()

    engine  = FaceEngine()
    library = FaceLibrary(path=args.db_path)

    # --- FASE 1: Benchmark de detección pura ---
    print("\n🔬 FASE 1 — BENCHMARK DE DETECCIÓN")
    det_dfs = []
    for v in args.videos:
        if not os.path.exists(v):
            print(f"  ⚠️  Video no encontrado: {v}")
            continue
        df = benchmark_deteccion(v, engine, args.modelo)
        if not df.empty:
            df['video'] = os.path.basename(v)
            det_dfs.append(df)

    if det_dfs:
        df_det = pd.concat(det_dfs, ignore_index=True)
        imprimir_informe_deteccion(df_det, args.modelo)
        df_det.to_csv(args.output_csv.replace(".csv", "_deteccion.csv"), index=False)
        print(f"\n  💾 CSV guardado: {args.output_csv.replace('.csv', '_deteccion.csv')}")
    else:
        df_det = pd.DataFrame()
        print("  ❌ No se procesó ningún video.")

    # --- FASE 2: Benchmark de reconocimiento 1..N templates ---
    if library.face_db:
        print(f"\n🧬 FASE 2 — COMPARATIVA 1..{args.max_templates} TEMPLATES")
        resultados_n = {}

        for n in range(1, args.max_templates + 1):
            print(f"\n  📐 Probando con {n} template(s)...")
            rec_dfs = []
            for v in args.videos:
                if not os.path.exists(v):
                    continue
                df = benchmark_reconocimiento(v, engine, library, n)
                if not df.empty:
                    df['video'] = os.path.basename(v)
                    rec_dfs.append(df)

            if rec_dfs:
                resultados_n[n] = pd.concat(rec_dfs, ignore_index=True)

        if resultados_n:
            imprimir_informe_reconocimiento(resultados_n)

            # Exportar CSV consolidado de reconocimiento
            df_rec_all = pd.concat(resultados_n.values(), ignore_index=True)
            csv_rec = args.output_csv.replace(".csv", "_reconocimiento.csv")
            df_rec_all.to_csv(csv_rec, index=False)
            print(f"\n  💾 CSV guardado: {csv_rec}")

            # Preparar datos para la gráfica
            resumen = []
            for n, df in sorted(resultados_n.items()):
                if not df.empty:
                    resumen.append({
                        "n_templates": n,
                        "tasa_pct"   : round(df['reconocido'].mean() * 100, 2),
                        "dist_media" : round(df['distancia_l2'].mean(), 4),
                        "lat_ms"     : round(df['latencia_ms'].mean(), 3),
                        "n_intentos" : len(df),
                    })
            df_resumen = pd.DataFrame(resumen)
            csv_resumen = args.output_csv.replace(".csv", "_resumen_templates.csv")
            df_resumen.to_csv(csv_resumen, index=False)
            print(f"  💾 Resumen para gráfica: {csv_resumen}")
            print(f"\n  ℹ️  Abre el artefacto 'benchmark_chart' para visualizar los resultados.")
    else:
        print("\n  ⚠️  FaceLibrary vacía — omitiendo Fase 2. Ejecuta registro_automatico.py primero.")