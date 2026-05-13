"""
registro_por_video.py — Un solo PKL por persona con 6 ángulos (v6)
===================================================================
Cambios respecto a v5:
  - HUD de una sola línea: solo imprime cuando hay un evento (nueva
    captura o mejora). Sin flood de consola.
  - Soporte GPU: argumento --gpu (ctx_id=0) para acelerar inferencia.
  - Ventanas diagonales más permisivas: umbral de entrada reducido a
    ±10° en yaw y pitch para facilitar la captura.

Estructura del PKL:
  {
    'version': 3,
    'subject_id': 'nombre_del_video',
    'gallery': [
        {'embedding': array(512,), 'pose_tag': 'frontal',            ...},
        {'embedding': array(512,), 'pose_tag': 'frontal_abajo',      ...},
        {'embedding': array(512,), 'pose_tag': 'tres_cuartos_izq',   ...},
        {'embedding': array(512,), 'pose_tag': 'tres_cuartos_der',   ...},
        {'embedding': array(512,), 'pose_tag': 'diagonal_abajo_izq', ...},
        {'embedding': array(512,), 'pose_tag': 'diagonal_abajo_der', ...},
    ],
    'registered_at': timestamp,
    'update_count': 0,
  }

Archivo : data/embeddings/<nombre>_embedding.pkl
Imágenes: data/raw/<nombre>/pose_<tag>.jpg
"""

import cv2
import os
import sys
import pickle
import time
import argparse
import numpy as np
from src.face_engine import FaceEngine

sys.path.insert(0, os.path.dirname(__file__))


# ==========================================================
# CONFIGURACIÓN DE EJES Y FILTROS TÉCNICOS
# ==========================================================
PITCH_IDX   = 0
YAW_IDX     = 1

NITIDEZ_MIN = 50.0
TAMANO_MIN  = 60

# Ancho fijo del HUD. Ajusta al ancho de tu terminal si es necesario.
HUD_WIDTH   = 110


# ==========================================================
# ÁNGULOS OBJETIVO
# ==========================================================
# Diagonales: umbral de entrada reducido a ±10° (antes ±18°) y punto
# ideal acercado al borde para que frames con giro moderado + barbilla
# ligeramente baja ya clasifiquen.
POSES_OBJETIVO = {
    "frontal": {
        "short"      : "FRONTAL",
        "yaw_min"    : -18,  "yaw_max"    : +18,
        "pitch_min"  : -18,  "pitch_max"  : +18,
        "yaw_ideal"  :   0,  "pitch_ideal":   0,
        "label"      : "Frontal puro",
        "instruccion": "Mira directo a la camara",
    },
    "frontal_abajo": {
        "short"      : "F_ABAJO",
        "yaw_min"    : -18,  "yaw_max"    : +18,
        "pitch_min"  : -55,  "pitch_max"  : -18,
        "yaw_ideal"  :   0,  "pitch_ideal": -32,
        "label"      : "Frontal abajo",
        "instruccion": "Baja la cabeza / mira hacia el suelo",
    },
    "tres_cuartos_izq": {
        "short"      : "3/4_IZQ",
        "yaw_min"    : -65,  "yaw_max"    : -18,
        "pitch_min"  : -20,  "pitch_max"  : +20,
        "yaw_ideal"  : -38,  "pitch_ideal":   0,
        "label"      : "3/4 izquierdo",
        "instruccion": "Gira la cabeza hacia tu izquierda",
    },
    "tres_cuartos_der": {
        "short"      : "3/4_DER",
        "yaw_min"    : +18,  "yaw_max"    : +65,
        "pitch_min"  : -20,  "pitch_max"  : +20,
        "yaw_ideal"  : +38,  "pitch_ideal":   0,
        "label"      : "3/4 derecho",
        "instruccion": "Gira la cabeza hacia tu derecha",
    },
    "diagonal_abajo_izq": {
        "short"      : "DIAG_IZQ",
        # Umbral de entrada reducido de -18 a -10 en ambos ejes.
        "yaw_min"    : -65,  "yaw_max"    : -10,
        "pitch_min"  : -55,  "pitch_max"  : -10,
        # Ideal más cercano al borde para maximizar capturas válidas.
        "yaw_ideal"  : -30,  "pitch_ideal": -25,
        "label"      : "Diagonal abajo izquierda",
        "instruccion": "Gira a tu izquierda y baja ligeramente la barbilla",
    },
    "diagonal_abajo_der": {
        "short"      : "DIAG_DER",
        "yaw_min"    : +10,  "yaw_max"    : +65,
        "pitch_min"  : -55,  "pitch_max"  : -10,
        "yaw_ideal"  : +30,  "pitch_ideal": -25,
        "label"      : "Diagonal abajo derecha",
        "instruccion": "Gira a tu derecha y baja ligeramente la barbilla",
    },
}

ORDEN_POSES = [
    "frontal",
    "frontal_abajo",
    "tres_cuartos_izq",
    "tres_cuartos_der",
    "diagonal_abajo_izq",
    "diagonal_abajo_der",
]


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


def _dist_ideal(pitch, yaw, cfg) -> float:
    return float(np.sqrt((pitch - cfg["pitch_ideal"])**2 +
                         (yaw   - cfg["yaw_ideal"]  )**2))


def _pasa_ventana(pitch, yaw, cfg) -> bool:
    return (cfg["yaw_min"] <= yaw <= cfg["yaw_max"] and
            cfg["pitch_min"] <= pitch <= cfg["pitch_max"])


def _crop_rostro(frame, bbox, margen=50):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    return frame[max(0, y1-margen):min(h, y2+margen),
                 max(0, x1-margen):min(w, x2+margen)]


def _zona_actual(pitch, yaw) -> str:
    """Clasifica la posición actual de la cabeza en texto corto."""
    izq  = yaw   < -10
    der  = yaw   > +10
    abaj = pitch < -10
    arri = pitch > +18

    if not izq and not der and not abaj and not arri: return "FRONTAL"
    if not izq and not der and arri:                  return "ARRIBA"
    if not izq and not der and abaj:                  return "F_ABAJO"
    if izq and not abaj:                              return "3/4_IZQ"
    if der and not abaj:                              return "3/4_DER"
    if izq and abaj:                                  return "DIAG_IZQ"
    if der and abaj:                                  return "DIAG_DER"
    return "TRANS"


def _barra_estado(capturas, mejoras) -> str:
    """Cadena compacta de estado para el HUD de una sola línea."""
    partes = []
    for tag in ORDEN_POSES:
        short  = POSES_OBJETIVO[tag]["short"]
        icono  = "OK" if capturas[tag] else "--"
        m      = mejoras[tag]
        sufijo = f"+{m}" if capturas[tag] and m > 0 else ""
        partes.append(f"[{icono}]{short}{sufijo}")
    return "  ".join(partes)


def _imprimir_hud(capturas, mejoras, pct, completados,
                  total, pitch=None, yaw=None):
    """
    Sobreescribe UNA sola línea en consola con \\r.
    La cadena se rellena a HUD_WIDTH para borrar residuos anteriores.
    """
    barra = _barra_estado(capturas, mejoras)
    zona  = _zona_actual(pitch, yaw) if pitch is not None else "---"
    info  = f"[{pct:3d}%] {completados}/{total}  {zona:<10}"
    linea = f"  {barra}  {info}"
    linea = linea[:HUD_WIDTH].ljust(HUD_WIDTH)
    print(f"\r{linea}", end="", flush=True)


def _nombre_desde_video(video_path: str) -> str:
    """Extrae el nombre del sujeto del stem del archivo de video."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    safe = stem.replace(" ", "_").replace('"', "").replace("'", "")
    return safe


# ==========================================================
# GUARDADO
# ==========================================================

def guardar_resultado(nombre: str, capturas: dict, db_path: str):
    carpeta_raw = os.path.join(db_path, "raw", nombre)
    carpeta_emb = os.path.join(db_path, "embeddings")
    os.makedirs(carpeta_raw, exist_ok=True)
    os.makedirs(carpeta_emb, exist_ok=True)

    gallery = []
    imagenes_guardadas = 0

    for tag in ORDEN_POSES:
        datos = capturas.get(tag)
        if datos is None:
            continue

        emb  = datos["embedding"]
        norm = np.linalg.norm(emb)
        emb  = emb / norm if norm > 0 else emb

        gallery.append({
            "embedding" : emb,
            "pose_tag"  : tag,
            "dist_ideal": round(datos["dist"], 3),
            "pitch"     : datos["pitch"],
            "yaw"       : datos["yaw"],
            "nitidez"   : datos["nitidez"],
            "timestamp" : datos["timestamp"],
        })

        ruta_jpg = os.path.join(carpeta_raw, f"pose_{tag}.jpg")
        cv2.imwrite(ruta_jpg, datos["crop"])
        imagenes_guardadas += 1

    if not gallery:
        print("  ERROR: Galeria vacia, no hay nada que guardar.")
        return 0

    payload = {
        "version"      : 3,
        "subject_id"   : nombre,
        "gallery"      : gallery,
        "registered_at": time.time(),
        "update_count" : 0,
    }

    # Escritura atómica: temp → rename.
    ruta_pkl = os.path.join(carpeta_emb, f"{nombre}_embedding.pkl")
    tmp = ruta_pkl + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, ruta_pkl)

    print(f"\n  PKL guardado : {os.path.basename(ruta_pkl)}")
    print(f"  Angulos      : {len(gallery)}/6")
    for entry in gallery:
        print(f"    {entry['pose_tag']:<24} dist={entry['dist_ideal']:.2f}  "
              f"yaw={entry['yaw']:+.1f}  pitch={entry['pitch']:+.1f}")
    print(f"  Imagenes     : data/raw/{nombre}/")

    return imagenes_guardadas


# ==========================================================
# EXTRACCIÓN CON REFINAMIENTO
# ==========================================================

def extraer_angulos(video_path: str, engine, modo_diagnostico=False) -> tuple:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\nERROR: No se pudo abrir: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30
    n_poses      = len(ORDEN_POSES)

    print(f"\n  Video    : {os.path.basename(video_path)}")
    print(f"  Frames   : {total_frames}  |  FPS: {fps_video:.1f}  |  "
          f"Duracion: {total_frames / fps_video:.1f}s\n")

    if modo_diagnostico:
        print(f"  {'FRAME':>6}  {'PITCH':>8}  {'YAW':>8}  {'NIT':>8}  "
              f"{'ANCHO':>6}  ZONA")
        print(f"  {'─'*62}")

    capturas         = {tag: None for tag in POSES_OBJETIVO}
    mejoras          = {tag: 0    for tag in POSES_OBJETIVO}
    frame_num        = 0
    prev_completados = -1  # Para detectar cambios en el HUD

    if not modo_diagnostico:
        _imprimir_hud(capturas, mejoras, 0, 0, n_poses)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rostros = engine.procesar_frame(frame)

        ultimo_pitch, ultimo_yaw = None, None
        hubo_evento = False

        for r in rostros:
            ancho = r["res"][0]
            if ancho < TAMANO_MIN:
                continue

            nit        = _nitidez(frame, r["bbox"])
            pitch, yaw = _extraer_pose(r)
            emb        = r["embedding"]

            ultimo_pitch, ultimo_yaw = pitch, yaw

            if modo_diagnostico:
                print(f"  {frame_num:>6}  {pitch:>+8.2f}  {yaw:>+8.2f}  "
                      f"{nit:>8.1f}  {ancho:>6}  {_zona_actual(pitch, yaw)}")

            if nit < NITIDEZ_MIN:
                continue

            for tag, cfg in POSES_OBJETIVO.items():
                if not _pasa_ventana(pitch, yaw, cfg):
                    continue

                dist       = _dist_ideal(pitch, yaw, cfg)
                es_primera = capturas[tag] is None
                es_mejor   = not es_primera and dist < capturas[tag]["dist"]

                if es_primera or es_mejor:
                    capturas[tag] = {
                        "dist"     : dist,
                        "embedding": emb.copy(),
                        "crop"     : _crop_rostro(frame, r["bbox"]),
                        "pitch"    : round(pitch, 2),
                        "yaw"      : round(yaw,   2),
                        "nitidez"  : round(nit,   1),
                        "timestamp": time.time(),
                        "frame_num": frame_num,
                    }
                    if es_mejor:
                        mejoras[tag] += 1
                    hubo_evento = True  # Hay algo nuevo que mostrar

        # ── Actualización del HUD ────────────────────────────────────
        # Solo se redibuya en dos casos:
        #   1) Hubo una captura nueva o mejora (evento relevante).
        #   2) Cambió el número total de poses completadas.
        # Así se evita el flood de líneas.
        if not modo_diagnostico:
            completados = sum(1 for v in capturas.values() if v is not None)
            pct         = int(frame_num / max(total_frames, 1) * 100)

            if hubo_evento or completados != prev_completados:
                _imprimir_hud(capturas, mejoras, pct, completados,
                              n_poses, ultimo_pitch, ultimo_yaw)
                prev_completados = completados

        frame_num += 1

    cap.release()
    completados = sum(1 for v in capturas.values() if v is not None)
    if not modo_diagnostico:
        _imprimir_hud(capturas, mejoras, 100, completados, n_poses)
        print()   # Salto limpio al terminar
    return capturas, mejoras


# ==========================================================
# RESUMEN
# ==========================================================

def imprimir_resumen(capturas: dict, mejoras: dict):
    print(f"\n  {'─'*78}")
    print(f"  {'ANGULO':<28} {'YAW':>6} {'PITCH':>6} {'NIT':>7} "
          f"{'FRAME':>6} {'DIST_IDEAL':>10} {'MEJORAS':>7}  OK")
    print(f"  {'─'*78}")
    for tag in ORDEN_POSES:
        cfg  = POSES_OBJETIVO[tag]
        dato = capturas[tag]
        if dato:
            print(f"  {cfg['label']:<28} {dato['yaw']:>+6.1f} {dato['pitch']:>+6.1f} "
                  f"{dato['nitidez']:>7.1f} {dato['frame_num']:>6} "
                  f"{dato['dist']:>10.2f} {mejoras[tag]:>7}  SI")
        else:
            print(f"  {cfg['label']:<28} {'—':>6} {'—':>6} {'—':>7} "
                  f"{'—':>6} {'—':>10} {'—':>7}  NO")
    print(f"  {'─'*78}")
    print(f"  DIST_IDEAL = distancia al angulo perfecto (menor = mejor)")
    print(f"  MEJORAS    = veces que se reemplazo por un frame mas cercano al ideal")


# ==========================================================
# ENTRY POINT
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       default=None,
                        help="Ruta al archivo de video")
    parser.add_argument("--db_path",     default="data",
                        help="Directorio raiz de la base de datos")
    parser.add_argument("--diagnostico", action="store_true",
                        help="Imprime telemetria sin guardar")
    # [TÉCNICO]: ctx_id=0 → primera GPU CUDA; ctx_id=-1 → CPU.
    parser.add_argument("--gpu",         action="store_true",
                        help="Usar GPU (CUDA) para la inferencia")
    args = parser.parse_args()

    print("\n" + "="*64)
    print("  REGISTRO BIOMETRICO POR VIDEO - 6 ANGULOS  v6")
    print("="*64)
    dispositivo = "GPU (CUDA)" if args.gpu else "CPU"
    print(f"  Dispositivo : {dispositivo}")
    print(f"  1 PKL por persona con todos los angulos dentro")

    if not args.diagnostico:
        print(f"\n  Instrucciones:")
        for tag in ORDEN_POSES:
            cfg = POSES_OBJETIVO[tag]
            print(f"    [{cfg['short']:<10}]  {cfg['instruccion']}")

    # Validar ruta de video
    video_path = args.video
    if not video_path:
        video_path = input("\n  Ruta del video: ").strip().strip('"').strip("'")
    if not os.path.exists(video_path):
        print(f"\nERROR: No encontrado: {video_path}")
        sys.exit(1)

    # Nombre derivado automáticamente del archivo de video.
    nombre = _nombre_desde_video(video_path)
    print(f"\n  Sujeto      : '{nombre}'  (del nombre del archivo)")

    if not args.diagnostico:
        ruta_pkl_existente = os.path.join(args.db_path, "embeddings",
                                          f"{nombre}_embedding.pkl")
        if os.path.exists(ruta_pkl_existente):
            resp = input(
                f"\n  Ya existe registro para '{nombre}'. Sobreescribir? [s/n]: "
            ).strip().lower()
            if resp not in ("s", "si", "y", "yes"):
                print("  Cancelado.")
                sys.exit(0)

    print()

    # Inicializar motor con o sin GPU.
    ctx_id = 0 if args.gpu else -1
    engine = FaceEngine(ctx_id=ctx_id)

    capturas, mejoras = extraer_angulos(video_path, engine, args.diagnostico)

    if args.diagnostico:
        print(f"\n  Diagnostico completo.")
        sys.exit(0)

    imprimir_resumen(capturas, mejoras)

    completados = sum(1 for v in capturas.values() if v is not None)
    if completados == 0:
        print("\n  ERROR: No se capturo ningun angulo.")
        sys.exit(1)

    if completados < 6:
        print(f"\n  AVISO: Solo {completados}/6 angulos capturados.")
        resp = input("  Guardar igualmente? [s/n]: ").strip().lower()
        if resp not in ("s", "si", "y", "yes"):
            sys.exit(0)

    print()
    guardados = guardar_resultado(nombre, capturas, args.db_path)

    print(f"\n{'='*64}")
    print(f"  REGISTRO COMPLETADO : {nombre}")
    print(f"  1 PKL con {guardados} angulos  ->  data/embeddings/")
    print(f"  {guardados} imagenes           ->  data/raw/{nombre}/")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()