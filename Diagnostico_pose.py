"""
diagnostico_pose.py — Inspecciona los valores reales de pose que reporta InsightFace
=====================================================================================
Ejecuta esto PRIMERO con tu vídeo antes de usar registro_por_video.py.
Te imprime en tiempo real los valores exactos de pitch/yaw que el motor
detecta frame a frame, para que puedas calibrar las ventanas angulares.

Uso:
  python diagnostico_pose.py
  python diagnostico_pose.py --video mi_video.mp4 --cada 10
"""

import cv2
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
try:
    from src.face_engine import FaceEngine
except ModuleNotFoundError:
    from src.face_engine import FaceEngine


def diagnostico(video_path: str, cada_n_frames: int = 5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌  No se pudo abrir: {video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"\n📹 {os.path.basename(video_path)}  |  {total} frames  |  {fps:.0f} FPS")
    print(f"   Mostrando 1 de cada {cada_n_frames} frames con rostro detectado\n")
    print(f"{'FRAME':>6}  {'POSE[0]':>9}  {'POSE[1]':>9}  {'POSE[2]':>9}  "
          f"{'ANCHO':>6}  {'NITIDEZ':>9}  ZONA_ESTIMADA")
    print("─" * 80)

    engine    = FaceEngine()
    frame_num = 0
    detectados = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        rostros = engine.procesar_frame(frame)
        if not rostros:
            continue

        # Solo imprimir cada N frames detectados para no saturar la consola
        detectados += 1
        if detectados % cada_n_frames != 0:
            continue

        r = max(rostros, key=lambda x: x["res"][0])  # el más grande
        p0, p1, p2 = r.get("pose", (0, 0, 0))
        ancho       = r["res"][0]

        # Nitidez del ROI
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        h, w = frame.shape[:2]
        roi  = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        nit  = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Estimación de zona según ambas interpretaciones del eje
        # InsightFace devuelve pose = (pitch, yaw, roll)
        # Pero en algunas builds es (yaw, pitch, roll) — verifica con el vídeo
        pitch_a, yaw_a = p0, p1   # interpretación A: pose[0]=pitch, pose[1]=yaw
        pitch_b, yaw_b = p1, p0   # interpretación B: pose[0]=yaw,   pose[1]=pitch

        zona_a = _estimar_zona(pitch_a, yaw_a)
        zona_b = _estimar_zona(pitch_b, yaw_b)

        print(f"{frame_num:>6}  {p0:>+9.2f}  {p1:>+9.2f}  {p2:>+9.2f}  "
              f"{ancho:>6}  {nit:>9.1f}  "
              f"A={zona_a} / B={zona_b}")

    cap.release()
    print("─" * 80)
    print(f"\n  Total frames: {frame_num}  |  Con rostro: {detectados}")
    print("""
  ─────────────────────────────────────────────────────
  CÓMO INTERPRETAR:

  pose[0], pose[1], pose[2] son los tres valores brutos.
  InsightFace buffalo_l:  pose = (pitch, yaw, roll)
    pitch > 0 → cara mira ARRIBA
    pitch < 0 → cara mira ABAJO
    yaw   > 0 → cara girada a la DERECHA
    yaw   < 0 → cara girada a la IZQUIERDA

  Columna A asume pose[0]=pitch, pose[1]=yaw  (más común)
  Columna B asume pose[0]=yaw,   pose[1]=pitch (alternativa)

  Observa tu vídeo y decide cuál zona_A o zona_B coincide
  con el movimiento real que hiciste. Luego ajusta en
  registro_por_video.py qué índice es pitch y cuál es yaw.
  ─────────────────────────────────────────────────────
""")


def _estimar_zona(pitch, yaw) -> str:
    """Etiqueta aproximada de zona para ayudar en el diagnóstico."""
    if abs(yaw) <= 15 and abs(pitch) <= 15:
        return "FRONTAL    "
    if abs(yaw) <= 15 and pitch > 15:
        return "ARRIBA     "
    if abs(yaw) <= 15 and pitch < -15:
        return "ABAJO      "
    if yaw < -15 and abs(pitch) <= 20:
        return "GIRO_IZQ   "
    if yaw > +15 and abs(pitch) <= 20:
        return "GIRO_DER   "
    return f"OTRO       "


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None)
    parser.add_argument("--cada",  type=int, default=5, help="Imprimir 1 de cada N frames detectados")
    args = parser.parse_args()

    video = args.video
    if not video:
        video = input("📂  Ruta del vídeo: ").strip().strip('"').strip("'")
    if not os.path.exists(video):
        print(f"❌  No encontrado: {video}")
        sys.exit(1)

    diagnostico(video, args.cada)