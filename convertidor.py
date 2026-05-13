"""
convertidor.py — Registro manual por galería de imágenes
=========================================================
Flujo:
  1. Lee hasta 5 imágenes de la carpeta  ./convertidor/
  2. Genera el embedding de cada una con FaceEngine (buffalo_l)
  3. Solicita en terminal el nombre con el que se registrará la persona
  4. Guarda UN solo .pkl en  ./data/embeddings/<nombre>_embedding.pkl
     con el formato v2 (galería multi-template) compatible con FaceLibrary

Uso:
  python convertidor.py
  python convertidor.py --carpeta ruta/custom --db_path ruta/custom/embeddings
"""

import cv2
import os
import sys
import pickle
import time
import argparse
import numpy as np

# Ajusta el path si face_engine está en subcarpeta src/
sys.path.insert(0, os.path.dirname(__file__))
try:
    from src.face_engine import FaceEngine
except ModuleNotFoundError:
    from src.face_engine import FaceEngine


# ==========================================================
# CONFIGURACIÓN POR DEFECTO
# ==========================================================
CARPETA_IMAGENES = "convertidor"
CARPETA_DB       = os.path.join("data", "embeddings")
EXTENSIONES      = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_IMAGENES     = 5


# ==========================================================
# HELPERS
# ==========================================================

def _get_pose_tag(pitch: float, yaw: float) -> str:
    if abs(yaw) <= 5 and abs(pitch) <= 10:
        return "frontal"
    if yaw > 5:
        return "slight_right"
    if yaw < -5:
        return "slight_left"
    return "up" if pitch > 10 else "down"


def _pedir_nombre() -> str:
    """Solicita al usuario el nombre hasta recibir uno válido."""
    while True:
        nombre = input("\n📝  Escribe el nombre con el que se registrará esta persona: ").strip()
        if not nombre:
            print("    ⚠️  El nombre no puede estar vacío. Inténtalo de nuevo.")
            continue
        # Reemplazar espacios por guion bajo para nombre de archivo seguro
        nombre_safe = nombre.replace(" ", "_")
        confirmar = input(f"    ¿Confirmas el nombre '{nombre_safe}'? [s/n]: ").strip().lower()
        if confirmar in ("s", "si", "sí", "y", "yes"):
            return nombre_safe
        print("    Volvamos a intentarlo...")


def _verificar_no_duplicado(nombre: str, carpeta_db: str) -> bool:
    """Avisa si ya existe un pkl con ese nombre y deja decidir al usuario."""
    ruta = os.path.join(carpeta_db, f"{nombre}_embedding.pkl")
    if os.path.exists(ruta):
        print(f"\n    ⚠️  Ya existe un registro para '{nombre}' en la DB.")
        resp = input("    ¿Deseas SOBREESCRIBIRLO? [s/n]: ").strip().lower()
        return resp in ("s", "si", "sí", "y", "yes")
    return True


# ==========================================================
# NÚCLEO
# ==========================================================

def procesar_imagenes(carpeta_img: str, engine: FaceEngine) -> list:
    """
    Lee las imágenes de la carpeta, extrae el embedding de cada una
    y devuelve una lista de entradas de galería.

    Retorna:
        list of dict: [{embedding, score, pose_tag, fuente}, ...]
    """
    archivos = sorted([
        f for f in os.listdir(carpeta_img)
        if os.path.splitext(f)[1].lower() in EXTENSIONES
    ])

    if not archivos:
        print(f"\n❌  No se encontraron imágenes en '{carpeta_img}'.")
        return []

    if len(archivos) > MAX_IMAGENES:
        print(f"\n⚠️  Se encontraron {len(archivos)} imágenes. Solo se procesarán las primeras {MAX_IMAGENES}.")
        archivos = archivos[:MAX_IMAGENES]

    print(f"\n{'─'*50}")
    print(f"  Imágenes encontradas: {len(archivos)}")
    print(f"{'─'*50}")

    entradas = []

    for idx, archivo in enumerate(archivos, 1):
        ruta_completa = os.path.join(carpeta_img, archivo)
        print(f"\n  [{idx}/{len(archivos)}] Procesando: {archivo}")

        frame = cv2.imread(ruta_completa)
        if frame is None:
            print(f"    ❌ No se pudo leer la imagen. Saltando.")
            continue

        rostros = engine.procesar_frame(frame)

        if not rostros:
            print(f"    ⚠️  No se detectó ningún rostro. Saltando.")
            continue

        if len(rostros) > 1:
            print(f"    ⚠️  Se detectaron {len(rostros)} rostros. Se usará el de mayor tamaño.")

        # Elegir el rostro más grande (más cercano a cámara = más fiable)
        rostro = max(rostros, key=lambda r: r['res'][0] * r['res'][1])

        emb   = rostro['embedding']
        norm  = np.linalg.norm(emb)
        if norm == 0:
            print(f"    ❌ Embedding inválido (norma=0). Saltando.")
            continue
        emb = emb / norm  # normalización defensiva

        p_val, y_val, _ = rostro.get('pose', (0, 0, 0))
        ancho, alto     = rostro['res']

        # Score compuesto idéntico al usado en registro_automatico.py
        roi       = frame[max(0, rostro['bbox'][1]):rostro['bbox'][3],
                          max(0, rostro['bbox'][0]):rostro['bbox'][2]]
        gray      = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        nitidez   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        score     = nitidez - abs(y_val) * 5
        pose_tag  = _get_pose_tag(p_val, y_val)

        entrada = {
            'embedding': emb,
            'score'    : round(score, 2),
            'pose_tag' : pose_tag,
            'timestamp': time.time(),
            'fuente'   : archivo,      # solo para trazabilidad, no lo usa FaceLibrary
        }
        entradas.append(entrada)

        print(f"    ✅ Embedding generado | pose={pose_tag} | "
              f"yaw={y_val:.1f}° | nitidez={nitidez:.1f} | score={score:.1f}")

    return entradas


def guardar_pkl(nombre: str, entradas: list, carpeta_db: str):
    """
    Persiste la galería en formato v2 compatible con FaceLibrary.
    Escritura atómica: .tmp → os.replace()
    """
    os.makedirs(carpeta_db, exist_ok=True)
    ruta_pkl = os.path.join(carpeta_db, f"{nombre}_embedding.pkl")

    payload = {
        'version'     : 2,
        'gallery'     : entradas,          # lista de dicts con embedding/score/pose_tag
        'registered_at': time.time(),
        'update_count': 0,
    }

    tmp = ruta_pkl + ".tmp"
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, ruta_pkl)             # atómico en POSIX y Windows

    print(f"\n  💾 Guardado en: {ruta_pkl}")
    print(f"  📦 Templates en galería: {len(entradas)}")


# ==========================================================
# ENTRY POINT
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="Convertidor de imágenes a galería biométrica v2")
    parser.add_argument("--carpeta",  default=CARPETA_IMAGENES, help="Carpeta con las imágenes fuente")
    parser.add_argument("--db_path",  default=CARPETA_DB,       help="Carpeta de destino de embeddings")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  CONVERTIDOR BIOMÉTRICO — GALERÍA MULTI-TEMPLATE")
    print("=" * 50)
    print(f"  Carpeta fuente : {os.path.abspath(args.carpeta)}")
    print(f"  Destino DB     : {os.path.abspath(args.db_path)}")

    # Verificar carpeta de imágenes
    if not os.path.exists(args.carpeta):
        print(f"\n❌  La carpeta '{args.carpeta}' no existe.")
        print(f"    Créala y coloca hasta {MAX_IMAGENES} imágenes de la persona.")
        sys.exit(1)

    # Cargar modelo UNA sola vez (caro en tiempo)
    print("\n")
    engine = FaceEngine()

    # Procesar imágenes → lista de entradas de galería
    entradas = procesar_imagenes(args.carpeta, engine)

    if not entradas:
        print("\n❌  No se pudo generar ningún embedding. Revisa las imágenes y vuelve a intentarlo.")
        sys.exit(1)

    # Resumen antes de pedir nombre
    print(f"\n{'─'*50}")
    print(f"  Embeddings generados: {len(entradas)} / {MAX_IMAGENES}")
    poses = [e['pose_tag'] for e in entradas]
    print(f"  Poses detectadas    : {', '.join(poses)}")
    print(f"{'─'*50}")

    # Solicitar nombre en terminal
    nombre = _pedir_nombre()

    # Verificar duplicado
    if not _verificar_no_duplicado(nombre, args.db_path):
        print("\n  Operación cancelada. No se guardó ningún archivo.")
        sys.exit(0)

    # Guardar
    guardar_pkl(nombre, entradas, args.db_path)

    print(f"\n{'='*50}")
    print(f"  ✅ REGISTRO COMPLETADO: {nombre}")
    print(f"     {len(entradas)} templates guardados en la galería.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()