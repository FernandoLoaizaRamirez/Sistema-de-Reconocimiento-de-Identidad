"""
registro_por_video.py — Registro biométrico multi-ángulo desde video
=====================================================================
Extrae hasta 6 embeddings (uno por ángulo de pose) del video indicado
y los guarda en un único PKL por persona.

Uso:
    python registro_por_video.py
    python registro_por_video.py --video "ruta/al/video.mp4"
    python registro_por_video.py --video "ruta/al/video.mp4" --diagnostico

Estructura del PKL generado:
    {
        "subject_id"   : "Nombre_Apellido",
        "gallery"      : [
            {
                "embedding" : np.ndarray (512,) normalizado L2,
                "pose_tag"  : str,
                "dist_ideal": float,
                "pitch"     : float,
                "yaw"       : float,
                "nitidez"   : float,
                "timestamp" : float,
            },
            ...  (hasta 6 entradas, una por ángulo)
        ],
        "registered_at": float  (unix timestamp),
    }

Archivos generados:
    data/embeddings/<nombre>_embedding.pkl
    data/raw/<nombre>/pose_<tag>.jpg
"""

import argparse
import os
import pickle
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.face_engine import FaceEngine

# ==========================================================
# PARÁMETROS DE CALIDAD
# ==========================================================
NITIDEZ_MIN = 50.0  # Varianza de Laplaciano mínima para aceptar un frame
TAMANO_MIN = 60  # Ancho mínimo del rostro en píxeles
HUD_WIDTH = 110  # Ancho de la línea de estado en consola

# Índices del vector pose de InsightFace
PITCH_IDX = 0
YAW_IDX = 1


# ==========================================================
# DEFINICIÓN DE ÁNGULOS OBJETIVO
# ==========================================================
POSES_OBJETIVO = {
    "frontal": {
        "short": "Frontal",
        "yaw_min": -18,
        "yaw_max": +18,
        "pitch_min": -18,
        "pitch_max": +18,
        "yaw_ideal": 0,
        "pitch_ideal": 0,
        "label": "Frontal puro",
        "instruccion": "Mira directo a la camara",
    },
    "frontal_abajo": {
        "short": "F. Abajo",
        "yaw_min": -18,
        "yaw_max": +18,
        "pitch_min": -55,
        "pitch_max": -18,
        "yaw_ideal": 0,
        "pitch_ideal": -32,
        "label": "Frontal abajo",
        "instruccion": "Baja la cabeza / mira hacia el suelo",
    },
    "tres_cuartos_izq": {
        "short": "3/4 Izq.",
        "yaw_min": -65,
        "yaw_max": -18,
        "pitch_min": -20,
        "pitch_max": +20,
        "yaw_ideal": -38,
        "pitch_ideal": 0,
        "label": "3/4 izquierdo",
        "instruccion": "Gira la cabeza hacia tu izquierda",
    },
    "tres_cuartos_der": {
        "short": "3/4 Der.",
        "yaw_min": +18,
        "yaw_max": +65,
        "pitch_min": -20,
        "pitch_max": +20,
        "yaw_ideal": +38,
        "pitch_ideal": 0,
        "label": "3/4 derecho",
        "instruccion": "Gira la cabeza hacia tu derecha",
    },
    "diagonal_abajo_izq": {
        "short": "Diag. Izq.",
        "yaw_min": -65,
        "yaw_max": -10,
        "pitch_min": -55,
        "pitch_max": -10,
        "yaw_ideal": -30,
        "pitch_ideal": -25,
        "label": "Diagonal abajo izquierda",
        "instruccion": "Gira a tu izquierda y baja ligeramente la barbilla",
    },
    "diagonal_abajo_der": {
        "short": "Diag. Der.",
        "yaw_min": +10,
        "yaw_max": +65,
        "pitch_min": -55,
        "pitch_max": -10,
        "yaw_ideal": +30,
        "pitch_ideal": -25,
        "label": "Diagonal abajo derecha",
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
# UTILIDADES INTERNAS
# ==========================================================


def _nitidez(frame: np.ndarray, bbox) -> float:
    """Calcula la nitidez de una región del rostro usando la varianza del Laplaciano.

    Args:
        frame (np.ndarray): El frame completo.
        bbox (tuple|list): Coordenadas del rostro (x1, y1, x2, y2).

    Returns:
        float: Valor de nitidez (varianza del Laplaciano).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    roi = frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _dist_ideal(pitch: float, yaw: float, cfg: dict) -> float:
    """Calcula la distancia euclidiana al ángulo ideal configurado.

    Args:
        pitch (float): Ángulo de pitch actual.
        yaw (float): Ángulo de yaw actual.
        cfg (dict): Configuración del ángulo objetivo.

    Returns:
        float: Distancia euclidiana.
    """
    return float(
        np.sqrt((pitch - cfg["pitch_ideal"]) ** 2 + (yaw - cfg["yaw_ideal"]) ** 2)
    )


def _pasa_ventana(pitch: float, yaw: float, cfg: dict) -> bool:
    """Verifica si la pose actual cae dentro de la ventana angular permitida.

    Args:
        pitch (float): Ángulo de pitch actual.
        yaw (float): Ángulo de yaw actual.
        cfg (dict): Configuración del ángulo objetivo.

    Returns:
        bool: True si está dentro del rango.
    """
    return (
        cfg["yaw_min"] <= yaw <= cfg["yaw_max"]
        and cfg["pitch_min"] <= pitch <= cfg["pitch_max"]
    )


def _crop_rostro(frame: np.ndarray, bbox, margen: int = 50) -> np.ndarray:
    """Recorta el rostro de la imagen con un margen adicional.

    Args:
        frame (np.ndarray): El frame completo.
        bbox (tuple|list): Coordenadas del rostro.
        margen (int): Píxeles extra de margen.

    Returns:
        np.ndarray: Imagen recortada del rostro.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    return frame[
        max(0, y1 - margen) : min(h, y2 + margen),
        max(0, x1 - margen) : min(w, x2 + margen),
    ]


def _zona_actual(pitch: float, yaw: float) -> str:
    """Clasifica la posición de la cabeza en un nombre descriptivo para el HUD.

    Args:
        pitch (float): Ángulo de pitch.
        yaw (float): Ángulo de yaw.

    Returns:
        str: Nombre descriptivo de la zona (ej. 'Frontal', '3/4 Izq.').
    """
    izq = yaw < -10
    der = yaw > +10
    abaj = pitch < -10
    arri = pitch > +18
    if not izq and not der and not abaj and not arri:
        return "Frontal"
    if not izq and not der and arri:
        return "Arriba"
    if not izq and not der and abaj:
        return "F. Abajo"
    if izq and not abaj:
        return "3/4 Izq."
    if der and not abaj:
        return "3/4 Der."
    if izq and abaj:
        return "Diag. Izq."
    if der and abaj:
        return "Diag. Der."
    return "Transición"


def _barra_estado(capturas: dict, mejoras: dict) -> str:
    """Genera una cadena compacta con el estado de capturas para el HUD.

    Args:
        capturas (dict): Estado actual de capturas por ángulo.
        mejoras (dict): Contador de mejoras por ángulo.

    Returns:
        str: Cadena formateada para consola.
    """
    partes = []
    for tag in ORDEN_POSES:
        short = POSES_OBJETIVO[tag]["short"]
        icono = "OK" if capturas[tag] else "--"
        sufijo = f"+{mejoras[tag]}" if capturas[tag] and mejoras[tag] > 0 else ""
        partes.append(f"[{icono}]{short}{sufijo}")
    return "  ".join(partes)


def _imprimir_hud(
    capturas: dict,
    mejoras: dict,
    pct: int,
    completados: int,
    total: int,
    pitch: float = None,
    yaw: float = None,
):
    """Sobreescribe la línea actual en consola con el estado del proceso.

    Args:
        capturas (dict): Diccionario de capturas realizadas.
        mejoras (dict): Diccionario de mejoras realizadas.
        pct (int): Porcentaje de progreso del video.
        completados (int): Cantidad de ángulos capturados.
        total (int): Total de ángulos objetivo.
        pitch (float, optional): Ángulo pitch actual.
        yaw (float, optional): Ángulo yaw actual.
    """
    barra = _barra_estado(capturas, mejoras)
    zona = _zona_actual(pitch, yaw) if pitch is not None else "---"
    info = f"[{pct:3d}%] {completados}/{total}  {zona:<10}"
    linea = f"  {barra}  {info}"
    print(f"\r{linea[:HUD_WIDTH].ljust(HUD_WIDTH)}", end="", flush=True)


def _nombre_desde_video(video_path: str) -> str:
    """Extrae y normaliza el nombre del sujeto a partir de la ruta del video.

    Args:
        video_path (str): Ruta al archivo de video.

    Returns:
        str: Nombre normalizado (sin espacios, en minúsculas/Mayúsculas según origen).
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return stem.replace(" ", "_").replace('"', "").replace("'", "")


# ==========================================================
# EXTRACCIÓN DE ÁNGULOS
# ==========================================================


def extraer_angulos(
    video_path: str, engine: FaceEngine, modo_diagnostico: bool = False
) -> tuple[dict, dict]:
    """Procesa el video para capturar el mejor embedding de cada ángulo objetivo.

    Args:
        video_path (str): Ruta al video de entrada.
        engine (FaceEngine): Motor de procesamiento facial.
        modo_diagnostico (bool): Si es True, imprime telemetría en lugar de mostrar HUD.

    Returns:
        tuple[dict, dict]: (capturas, mejoras) donde capturas contiene los datos del rostro
            por cada ángulo y mejoras el conteo de refinamientos realizados.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\nERROR: No se pudo abrir: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    n_poses = len(ORDEN_POSES)

    print(f"\n  Video    : {os.path.basename(video_path)}")
    print(
        f"  Frames   : {total_frames}  |  FPS: {fps_video:.1f}  |  "
        f"Duracion: {total_frames / fps_video:.1f}s\n"
    )

    if modo_diagnostico:
        print(
            f"  {'FRAME':>6}  {'PITCH':>8}  {'YAW':>8}  {'NIT':>8}  {'ANCHO':>6}  ZONA"
        )
        print(f"  {'─' * 62}")

    capturas = {tag: None for tag in POSES_OBJETIVO}
    mejoras = {tag: 0 for tag in POSES_OBJETIVO}
    frame_num = 0
    prev_completados = -1

    if not modo_diagnostico:
        _imprimir_hud(capturas, mejoras, 0, 0, n_poses)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rostros = engine.procesar_frame(frame)
        ult_pitch = None
        ult_yaw = None
        hubo_evento = False

        for r in rostros:
            if r["res"][0] < TAMANO_MIN:
                continue

            nit = _nitidez(frame, r["bbox"])
            pitch, yaw, _ = r["pose"]
            ult_pitch, ult_yaw = pitch, yaw

            if modo_diagnostico:
                print(
                    f"  {frame_num:>6}  {pitch:>+8.2f}  {yaw:>+8.2f}  "
                    f"{nit:>8.1f}  {r['res'][0]:>6}  {_zona_actual(pitch, yaw)}"
                )

            if nit < NITIDEZ_MIN:
                continue

            for tag, cfg in POSES_OBJETIVO.items():
                if not _pasa_ventana(pitch, yaw, cfg):
                    continue

                dist = _dist_ideal(pitch, yaw, cfg)
                es_primera = capturas[tag] is None
                es_mejor = not es_primera and dist < capturas[tag]["dist"]

                if es_primera or es_mejor:
                    capturas[tag] = {
                        "dist": dist,
                        "embedding": r["embedding"].copy(),
                        "crop": _crop_rostro(frame, r["bbox"]),
                        "pitch": round(pitch, 2),
                        "yaw": round(yaw, 2),
                        "nitidez": round(nit, 1),
                        "timestamp": time.time(),
                    }
                    if es_mejor:
                        mejoras[tag] += 1
                    hubo_evento = True

        if not modo_diagnostico:
            completados = sum(1 for v in capturas.values() if v is not None)
            pct = int(frame_num / max(total_frames, 1) * 100)
            if hubo_evento or completados != prev_completados:
                _imprimir_hud(
                    capturas, mejoras, pct, completados, n_poses, ult_pitch, ult_yaw
                )
                prev_completados = completados

        frame_num += 1

    cap.release()

    completados = sum(1 for v in capturas.values() if v is not None)
    if not modo_diagnostico:
        _imprimir_hud(capturas, mejoras, 100, completados, n_poses)
        print()

    return capturas, mejoras


# ==========================================================
# RESUMEN EN CONSOLA
# ==========================================================


def imprimir_resumen(capturas: dict, mejoras: dict):
    """Muestra un resumen en consola de los ángulos capturados y su calidad.

    Args:
        capturas (dict): Datos de los rostros capturados.
        mejoras (dict): Conteo de mejoras por cada ángulo.
    """
    print(f"\n  {'─' * 78}")
    print(
        f"  {'ANGULO':<28} {'YAW':>6} {'PITCH':>6} {'NIT':>7} "
        f"{'DIST_IDEAL':>10} {'MEJORAS':>7}  OK"
    )
    print(f"  {'─' * 78}")
    for tag in ORDEN_POSES:
        cfg = POSES_OBJETIVO[tag]
        dato = capturas[tag]
        if dato:
            print(
                f"  {cfg['label']:<28} {dato['yaw']:>+6.1f} {dato['pitch']:>+6.1f} "
                f"{dato['nitidez']:>7.1f} {dato['dist']:>10.2f} "
                f"{mejoras[tag]:>7}  SI"
            )
        else:
            print(
                f"  {cfg['label']:<28} {'—':>6} {'—':>6} {'—':>7} "
                f"{'—':>10} {'—':>7}  NO"
            )
    print(f"  {'─' * 78}")
    print(f"  DIST_IDEAL = distancia al angulo perfecto (menor = mejor)")
    print(f"  MEJORAS    = veces que se reemplazo por frame mas cercano al ideal")


# ==========================================================
# GUARDADO ATÓMICO
# ==========================================================


def guardar_resultado(nombre: str, capturas: dict, db_path: str) -> int:
    """Guarda los resultados del registro (PKL e imágenes) de forma atómica.

    Args:
        nombre (str): Identidad del sujeto.
        capturas (dict): Datos de los rostros capturados.
        db_path (str): Directorio base de la base de datos.

    Returns:
        int: Cantidad de ángulos guardados exitosamente.
    """
    carpeta_raw = os.path.join(db_path, "raw", nombre)
    carpeta_emb = os.path.join(db_path, "embeddings")
    os.makedirs(carpeta_raw, exist_ok=True)
    os.makedirs(carpeta_emb, exist_ok=True)

    gallery = []
    for tag in ORDEN_POSES:
        datos = capturas.get(tag)
        if datos is None:
            continue

        emb = datos["embedding"].astype(np.float32)
        norm = np.linalg.norm(emb)
        emb = emb / norm if norm > 0 else emb

        gallery.append(
            {
                "embedding": emb,
                "pose_tag": tag,
                "dist_ideal": round(datos["dist"], 3),
                "pitch": datos["pitch"],
                "yaw": datos["yaw"],
                "nitidez": datos["nitidez"],
                "timestamp": datos["timestamp"],
            }
        )

        cv2.imwrite(os.path.join(carpeta_raw, f"pose_{tag}.jpg"), datos["crop"])

    if not gallery:
        print("  ERROR: Galeria vacia, nada que guardar.")
        return 0

    payload = {
        "subject_id": nombre,
        "gallery": gallery,
        "registered_at": time.time(),
    }

    ruta_pkl = os.path.join(carpeta_emb, f"{nombre}_embedding.pkl")
    tmp = ruta_pkl + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, ruta_pkl)

    print(f"\n  PKL guardado : {os.path.basename(ruta_pkl)}")
    print(f"  Angulos      : {len(gallery)}/6")
    for entry in gallery:
        print(
            f"    {entry['pose_tag']:<24}  dist={entry['dist_ideal']:.2f}  "
            f"yaw={entry['yaw']:+.1f}  pitch={entry['pitch']:+.1f}"
        )
    print(f"  Imagenes     : data/raw/{nombre}/")
    return len(gallery)


# ==========================================================
# ENTRY POINT
# ==========================================================


def main():
    """Punto de entrada principal para el registro biométrico por video."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="Ruta al archivo de video")
    parser.add_argument(
        "--db_path",
        default="data",
        help="Directorio raiz de la base de datos (default: data)",
    )
    parser.add_argument(
        "--diagnostico",
        action="store_true",
        help="Imprime telemetria de pose/nitidez sin guardar",
    )
    args = parser.parse_args()

    print("\n" + "=" * 64)
    print("  REGISTRO BIOMETRICO POR VIDEO — 6 ANGULOS")
    print("=" * 64)
    print("  Dispositivo : GPU (CUDA) con fallback a CPU")

    if not args.diagnostico:
        print(f"\n  Instrucciones de pose:")
        for tag in ORDEN_POSES:
            cfg = POSES_OBJETIVO[tag]
            print(f"    [{cfg['short']:<10}]  {cfg['instruccion']}")

    # Ruta del video
    video_path = args.video
    if not video_path:
        video_path = input("\n  Ruta del video: ").strip().strip('"').strip("'")
    if not os.path.exists(video_path):
        print(f"\nERROR: No encontrado: {video_path}")
        sys.exit(1)

    nombre = _nombre_desde_video(video_path)
    print(f"\n  Sujeto      : '{nombre}'  (del nombre del archivo)")

    if not args.diagnostico:
        ruta_pkl = os.path.join(args.db_path, "embeddings", f"{nombre}_embedding.pkl")
        if os.path.exists(ruta_pkl):
            resp = (
                input(f"\n  Ya existe registro para '{nombre}'. Sobreescribir? [s/n]: ")
                .strip()
                .lower()
            )
            if resp not in ("s", "si", "y", "yes"):
                print("  Cancelado.")
                sys.exit(0)

    print()
    engine = FaceEngine(det_size=640)
    capturas, mejoras = extraer_angulos(video_path, engine, args.diagnostico)

    if args.diagnostico:
        print("\n  Diagnostico completo.")
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
    guardar_resultado(nombre, capturas, args.db_path)

    print(f"\n{'=' * 64}")
    print(f"  REGISTRO COMPLETADO : {nombre}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
