"""
registro_por_video.py — Un solo PKL por persona con 5 ángulos (v4)
===================================================================
Estructura del PKL generado:
  {
    'version': 2,
    'subject_id': 'Jose_Loaiza',
    'gallery': [
        {'embedding': array(512,), 'pose_tag': 'frontal',          'dist_ideal': 2.1, 'timestamp': ...},
        {'embedding': array(512,), 'pose_tag': 'frontal_arriba',   'dist_ideal': 1.8, 'timestamp': ...},
        {'embedding': array(512,), 'pose_tag': 'frontal_abajo',    'dist_ideal': 3.2, 'timestamp': ...},
        {'embedding': array(512,), 'pose_tag': 'tres_cuartos_izq', 'dist_ideal': 4.5, 'timestamp': ...},
        {'embedding': array(512,), 'pose_tag': 'tres_cuartos_der', 'dist_ideal': 2.9, 'timestamp': ...},
    ],
    'registered_at': timestamp,
    'update_count': 0,
  }

Archivo: data/embeddings/<nombre>_embedding.pkl   (uno solo por persona)
Imágenes: data/raw/<nombre>/pose_<tag>.jpg         (5 imágenes por persona)
"""

import cv2
import os
import sys
import pickle
import time
import argparse
import numpy as np
from src.face_engine import FaceEngine

# [OPERATIVO]: Gestión de dependencias y rutas del sistema.
# [TÉCNICO]: Inyecta el directorio del script en el sys.path para permitir importaciones relativas de módulos internos (src).
sys.path.insert(0, os.path.dirname(__file__))





# ==========================================================
# CONFIGURACIÓN DE EJES Y FILTROS TÉCNICOS
# ==========================================================
# [OPERATIVO]: Índices para interpretar el vector de rotación de la cabeza.
# [TÉCNICO]: Mapeo de posición en la tupla 'pose': 0 para Pitch (arriba/abajo) y 1 para Yaw (izq/der).
PITCH_IDX   = 0
YAW_IDX     = 1

# [OPERATIVO]: Criterios de calidad mínima para la base de datos biométrica.
# [TÉCNICO]: NITIDEZ_MIN define el umbral del Laplaciano; TAMANO_MIN el ancho en píxeles del rostro.
NITIDEZ_MIN = 50.0
TAMANO_MIN  = 60


# ==========================================================
# ÁNGULOS OBJETIVO (DICCIONARIO DE REFERENCIA)
# ==========================================================
# [OPERATIVO]: Define las 5 zonas espaciales necesarias para un registro facial completo y robusto.
# [TÉCNICO]: Estructura de metadatos que define ventanas de aceptación (min/max) y el punto óptimo (ideal).
POSES_OBJETIVO = {
    "frontal": {
        "cara_num": 10,
        "yaw_min": -18,   "yaw_max": +18,
        "pitch_min": -18, "pitch_max": +18,
        "yaw_ideal": 0,   "pitch_ideal": 0,
        "label": "Frontal puro (cara 10)",
        "instruccion": "Mira directo a la camara",
    },
    "frontal_arriba": {
        "cara_num": 5,
        "yaw_min": -18,   "yaw_max": +18,
        "pitch_min": +18, "pitch_max": +55,
        "yaw_ideal": 0,   "pitch_ideal": +32,
        "label": "Frontal leve arriba (cara 5)",
        "instruccion": "Levanta la cabeza / mira hacia arriba",
    },
    "frontal_abajo": {
        "cara_num": 15,
        "yaw_min": -18,   "yaw_max": +18,
        "pitch_min": -55, "pitch_max": -18,
        "yaw_ideal": 0,   "pitch_ideal": -32,
        "label": "Frontal leve abajo (cara 15)",
        "instruccion": "Baja la cabeza / mira hacia el suelo",
    },
    "tres_cuartos_izq": {
        "cara_num": 8,
        "yaw_min": -65,   "yaw_max": -18,
        "pitch_min": -20, "pitch_max": +20,
        "yaw_ideal": -38, "pitch_ideal": 0,
        "label": "3/4 izquierdo (cara 8)",
        "instruccion": "Gira la cabeza hacia tu izquierda",
    },
    "tres_cuartos_der": {
        "cara_num": 12,
        "yaw_min": +18,   "yaw_max": +65,
        "pitch_min": -20, "pitch_max": +20,
        "yaw_ideal": +38, "pitch_ideal": 0,
        "label": "3/4 derecho (cara 12)",
        "instruccion": "Gira la cabeza hacia tu derecha",
    },
}

# [OPERATIVO]: Determina la secuencia de captura para el flujo de trabajo.
# [TÉCNICO]: Lista ordenada de claves para iterar consistentemente sobre el diccionario de poses.
ORDEN_POSES = [
    "frontal", "frontal_arriba", "frontal_abajo",
    "tres_cuartos_izq", "tres_cuartos_der",
]

# ==========================================================
# HELPERS - FUNCIONES DE APOYO TÉCNICO
# ==========================================================

# [OPERATIVO]: Extrae los ángulos de rotación de la cabeza para determinar hacia dónde mira el usuario.
# [TÉCNICO]: Input: objeto rostro de InsightFace. Output: tuple(float, float). Recupera Pitch (X) y Yaw (Y) del vector de pose.
def _extraer_pose(rostro) -> tuple:
    pose = rostro.get("pose", (0, 0, 0))
    return float(pose[PITCH_IDX]), float(pose[YAW_IDX])


# [OPERATIVO]: Mide el nivel de detalle de la imagen para descartar capturas movidas o desenfocadas.
# [TÉCNICO]: Input: frame completo y Bounding Box. Output: float. Calcula la varianza del operador Laplaciano en la Región de Interés (ROI).
def _nitidez(frame, bbox) -> float:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    # [OPERATIVO]: Recorta el rostro con protección de límites de imagen.
    roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if roi.size == 0:
        return 0.0
    # [OPERATIVO]: Convierte a gris para analizar cambios de intensidad en los bordes.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# [OPERATIVO]: Calcula la desviación matemática entre la pose actual y la pose "perfecta" deseada.
# [TÉCNICO]: Input: ángulos actuales y configuración objetivo. Output: float. Implementa la Distancia Euclidiana entre coordenadas angulares.
def _dist_ideal(pitch, yaw, cfg) -> float:
    return float(np.sqrt((pitch - cfg["pitch_ideal"])**2 +
                         (yaw   - cfg["yaw_ideal"]  )**2))


# [OPERATIVO]: Verifica si la cabeza del usuario se encuentra dentro de los límites permitidos para una zona específica.
# [TÉCNICO]: Input: ángulos y config de ventana. Output: bool. Realiza una validación de rango lógico (Min <= Valor <= Max).
def _pasa_ventana(pitch, yaw, cfg) -> bool:
    return (cfg["yaw_min"] <= yaw <= cfg["yaw_max"] and
            cfg["pitch_min"] <= pitch <= cfg["pitch_max"])


# [OPERATIVO]: Extrae un recorte del rostro incluyendo un margen adicional para facilitar el reconocimiento posterior.
# [TÉCNICO]: Input: frame, bbox y margen de seguridad. Output: ndarray (crop). Slicing matricial con padding dinámico.
def _crop_rostro(frame, bbox, margen=50):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    return frame[max(0, y1-margen):min(h, y2+margen),
                 max(0, x1-margen):min(w, x2+margen)]


# [OPERATIVO]: Clasifica en lenguaje humano la posición actual de la cabeza para retroalimentación del usuario.
# [TÉCNICO]: Input: ángulos Pitch/Yaw. Output: str. Clasificador de lógica difusa basado en umbrales de 18 grados.
def _zona_actual(pitch, yaw) -> str:
    if abs(yaw) <= 18 and abs(pitch) <= 18: return "FRONTAL  "
    if abs(yaw) <= 18 and pitch >  18:      return "ARRIBA   "
    if abs(yaw) <= 18 and pitch < -18:      return "ABAJO    "
    if yaw < -18:                           return "GIRO_IZQ "
    if yaw > +18:                           return "GIRO_DER "
    return "TRANS    "


# [OPERATIVO]: Genera una representación visual del progreso de captura (Checklist en tiempo real).
# [TÉCNICO]: Input: dict de capturas y mejoras. Output: str. Construye un HUD de texto para la consola indicando estados y optimizaciones.
def _barra(capturas, mejoras) -> str:
    partes = []
    for tag in ORDEN_POSES:
        num   = POSES_OBJETIVO[tag]["cara_num"]
        icono = "OK" if capturas[tag] else "--"
        m     = mejoras[tag]
        # [OPERATIVO]: Indica visualmente si se ha encontrado una imagen mejor que la capturada inicialmente.
        sufijo = f"+{m}" if capturas[tag] and m > 0 else ""
        partes.append(f"[{icono}]C{num}{sufijo}")
    return "  ".join(partes)


# [OPERATIVO]: Gestiona la interacción por consola para definir la identidad del nuevo registro.
# [TÉCNICO]: Input: Entrada de teclado. Output: str (sanitizado). Limpia espacios y convierte nombres a formato de archivo (Safe String).
def _pedir_nombre() -> str:
    while True:
        nombre = input("\n  Nombre de la persona: ").strip()
        if not nombre:
            print("  El nombre no puede estar vacio.")
            continue
        # [OPERATIVO]: Reemplaza espacios por guiones bajos para evitar errores en rutas de archivo.
        nombre_safe = nombre.replace(" ", "_")
        conf = input(f"  Confirmas '{nombre_safe}'? [s/n]: ").strip().lower()
        if conf in ("s", "si", "y", "yes"):
            return nombre_safe


# ==========================================================
# GUARDADO — UN SOLO PKL POR PERSONA
# ==========================================================

# [OPERATIVO]: Orquestador de persistencia que consolida la identidad biométrica en el disco duro.
# [TÉCNICO]: Input: nombre (id), dict de capturas (RAM), path base. Output: int (conteo de archivos escritos).
def guardar_resultado(nombre: str, capturas: dict, db_path: str):
    """
    Guarda UN solo archivo PKL con toda la galería de la persona,
    más una imagen JPG por cada ángulo capturado.
    """
    # [OPERATIVO]: Define y asegura la existencia de las rutas físicas para imágenes (raw) y vectores (embeddings).
    # [TÉCNICO]: Uso de os.path.join para compatibilidad multiplataforma y os.makedirs con exist_ok para evitar excepciones de IO.
    carpeta_raw = os.path.join(db_path, "raw", nombre)
    carpeta_emb = os.path.join(db_path, "embeddings")
    os.makedirs(carpeta_raw, exist_ok=True)
    os.makedirs(carpeta_emb, exist_ok=True)

    # [OPERATIVO]: Estructura la lista de perfiles faciales que compondrán la galería del sujeto.
    # [TÉCNICO]: Inicialización de lista de diccionarios y acumulador entero.
    gallery = []
    imagenes_guardadas = 0

    # [OPERATIVO]: Itera sobre las poses capturadas para normalizar los datos antes del guardado definitivo.
    # [TÉCNICO]: Bucle basado en la constante ORDEN_POSES para mantener consistencia en la estructura del PKL.
    for tag in ORDEN_POSES:
        datos = capturas.get(tag)
        if datos is None:
            continue

        # [OPERATIVO]: Asegura que la firma matemática (embedding) tenga una magnitud unitaria para comparaciones futuras.
        # [TÉCNICO]: Normalización L2 manual usando la norma euclidiana de NumPy ($$v = v / ||v||$$).
        emb  = datos["embedding"]
        norm = np.linalg.norm(emb)
        emb  = emb / norm if norm > 0 else emb

        # [OPERATIVO]: Empaqueta metadatos técnicos y biométricos de cada ángulo de visión.
        # [TÉCNICO]: Append de diccionarios con tipos de datos nativos y redondeo de precisión para optimizar espacio.
        gallery.append({
            "embedding" : emb,
            "pose_tag"  : tag,
            "dist_ideal": round(datos["dist"], 3),   # qué tan cerca estuvo del ángulo perfecto
            "pitch"     : datos["pitch"],
            "yaw"       : datos["yaw"],
            "nitidez"   : datos["nitidez"],
            "timestamp" : datos["timestamp"],
        })

        # [OPERATIVO]: Exporta la imagen recortada del rostro a formato JPG para auditoría visual.
        # [TÉCNICO]: Escritura a disco mediante cv2.imwrite; vincula la imagen física con el registro del PKL.
        ruta_jpg = os.path.join(carpeta_raw, f"pose_{tag}.jpg")
        cv2.imwrite(ruta_jpg, datos["crop"])
        imagenes_guardadas += 1

    # [OPERATIVO]: Control de seguridad para evitar registros nulos en el sistema.
    # [TÉCNICO]: Validación de longitud de lista 'gallery' antes de proceder a la serialización.
    if not gallery:
        print("  ERROR: Galeria vacia, no hay nada que guardar.")
        return 0

    # [OPERATIVO]: Construye el objeto de datos final con metadatos de versión y trazabilidad.
    # [TÉCNICO]: Creación del 'payload' (Schema v2) con ID de sujeto y marca de tiempo de registro.
    payload = {
        "version"      : 2,
        "subject_id"   : nombre,          # campo nuevo: identidad de la persona
        "gallery"      : gallery,         # lista de todos los ángulos
        "registered_at": time.time(),
        "update_count" : 0,
    }

    # [OPERATIVO]: Ejecuta un guardado atómico para proteger la integridad de la base de datos.
    # [TÉCNICO]: Serialización mediante pickle. Se escribe primero un archivo '.tmp' y luego se renombra (os.replace) para evitar archivos corruptos en caso de fallo eléctrico.
    ruta_pkl = os.path.join(carpeta_emb, f"{nombre}_embedding.pkl")
    tmp = ruta_pkl + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, ruta_pkl)  # escritura atómica

    # [OPERATIVO]: Informa al usuario sobre la ubicación y el éxito de la operación.
    # [TÉCNICO]: Formateo de salida por consola con resumen de métricas capturadas.
    print(f"\n  PKL guardado: {os.path.basename(ruta_pkl)}")
    print(f"  Angulos en galeria: {len(gallery)}/5")
    for entry in gallery:
        print(f"    {entry['pose_tag']:<22} dist_ideal={entry['dist_ideal']:.2f}  "
              f"yaw={entry['yaw']:+.1f}  pitch={entry['pitch']:+.1f}")
    print(f"  Imagenes en: data/raw/{nombre}/")

    return imagenes_guardadas


# ==========================================================
# EXTRACCIÓN CON REFINAMIENTO
# ==========================================================

# [OPERATIVO]: Motor principal de análisis de video para la captura selectiva de poses biométricas.
# [TÉCNICO]: Input: path (str), motor de inferencia (FaceEngine), flag diagnóstico. Output: tuple(dict, dict).
def extraer_angulos(video_path: str, engine, modo_diagnostico=False) -> tuple:
    # [OPERATIVO]: Inicializa la comunicación con el archivo de video.
    # [TÉCNICO]: Instancia de VideoCapture para decodificación de stream mediante OpenCV.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\nERROR: No se pudo abrir: {video_path}")
        sys.exit(1)

    # [OPERATIVO]: Extrae metadatos del video para el cálculo de progreso y tiempos.
    # [TÉCNICO]: Recuperación de propiedades CAP_PROP_FRAME_COUNT y FPS del descriptor de video.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"\n  Video: {os.path.basename(video_path)}")
    print(f"  Frames: {total_frames}  |  FPS: {fps_video:.1f}  |  "
          f"Duracion: {total_frames / fps_video:.1f}s\n")

    # [OPERATIVO]: Configura la interfaz visual de depuración si se solicita el modo diagnóstico.
    # [TÉCNICO]: Impresión de cabeceras de tabla para monitoreo de telemetría facial en tiempo real.
    if modo_diagnostico:
        print(f"  {'FRAME':>6}  {'PITCH':>8}  {'YAW':>8}  {'NIT':>8}  "
              f"{'ANCHO':>6}  ZONA")
        print(f"  {'─'*58}")

    # [OPERATIVO]: Inicializa los contenedores para las 5 mejores capturas y sus contadores de optimización.
    # [TÉCNICO]: Estructuras de datos (dict comprehension) para seguimiento de estados por cada 'pose_tag'.
    capturas = {tag: None for tag in POSES_OBJETIVO}
    mejoras  = {tag: 0    for tag in POSES_OBJETIVO}
    frame_num = 0

    print(f"  {_barra(capturas, mejoras)}\n")

    # [OPERATIVO]: Bucle de procesamiento secuencial de imágenes del video.
    # [TÉCNICO]: Iteración sobre el buffer de video hasta el fin del stream (ret=False).
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # [OPERATIVO]: El motor de IA localiza y analiza todas las caras presentes en el frame actual.
        # [TÉCNICO]: Llamada al método de inferencia que retorna detecciones (BBoxes, Poses, Embeddings).
        rostros = engine.procesar_frame(frame)

        for r in rostros:
            # [OPERATIVO]: Filtra detecciones por tamaño para asegurar densidad de datos biométricos.
            # [TÉCNICO]: Validación de resolución horizontal mínima contra la constante TAMANO_MIN.
            ancho = r["res"][0]
            if ancho < TAMANO_MIN:
                continue

            # [OPERATIVO]: Calcula la calidad visual (enfoque) y la orientación de la cabeza.
            # [TÉCNICO]: Ejecución de algoritmos de Varianza de Laplaciano y extracción de ángulos de Euler (P/Y).
            nit        = _nitidez(frame, r["bbox"])
            pitch, yaw = _extraer_pose(r)
            emb        = r["embedding"]

            # [OPERATIVO]: Muestra los datos técnicos de cada cara si el modo diagnóstico está activo.
            # [TÉCNICO]: Formateo de logs con telemetría de pose y nitidez por frame procesado.
            if modo_diagnostico:
                print(f"  {frame_num:>6}  {pitch:>+8.2f}  {yaw:>+8.2f}  "
                      f"{nit:>8.1f}  {ancho:>6}  {_zona_actual(pitch, yaw)}")

            # [OPERATIVO]: Ignora frames que no cumplen con el estándar de nitidez requerido.
            # [TÉCNICO]: Gatekeeper de calidad basado en umbral de frecuencia espacial.
            if nit < NITIDEZ_MIN:
                continue

            # [OPERATIVO]: Clasifica la cara detectada en una de las 5 zonas de registro permitidas.
            # [TÉCNICO]: Evaluación de la pose contra los límites definidos en el diccionario POSES_OBJETIVO.
            for tag, cfg in POSES_OBJETIVO.items():
                if not _pasa_ventana(pitch, yaw, cfg):
                    continue

                # [OPERATIVO]: Calcula qué tan centrada está la pose respecto al ángulo ideal.
                # [TÉCNICO]: Cálculo de norma vectorial entre pose actual y 'pitch_ideal'/'yaw_ideal'.
                dist = _dist_ideal(pitch, yaw, cfg)
                es_primera = capturas[tag] is None
                es_mejor   = not es_primera and dist < capturas[tag]["dist"]

                # [OPERATIVO]: Actualiza el registro si es el primero o si es más preciso que el anterior.
                # [TÉCNICO]: Lógica de refinamiento: guarda metadatos, copia de embedding y recorte de imagen (crop).
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

        # [OPERATIVO]: Actualiza la barra de progreso y el HUD informativo cada 10 frames.
        # [TÉCNICO]: Cálculo de porcentaje de avance y retroalimentación de telemetría actual.
        if not modo_diagnostico and frame_num % 10 == 0:
            completados = sum(1 for v in capturas.values() if v is not None)
            pct = int(frame_num / max(total_frames, 1) * 100)
            pose_str = ""
            if rostros:
                p, y = _extraer_pose(rostros[0])
                pose_str = f"yaw={y:+.1f} pitch={p:+.1f} {_zona_actual(p, y)}"
            print(f"\r  {_barra(capturas, mejoras)}  [{pct:3d}%] {completados}/5  {pose_str:<36}",
                  end="", flush=True)

        frame_num += 1

    # [OPERATIVO]: Cierra el descriptor de video y muestra el estado final de la captura.
    # [TÉCNICO]: Liberación de memoria y retorno de estructuras de datos procesadas.
    cap.release()
    completados = sum(1 for v in capturas.values() if v is not None)
    print(f"\r  {_barra(capturas, mejoras)}  [100%] {completados}/5{' '*50}")
    return capturas, mejoras


# ==========================================================
# RESUMEN - MÓDULO DE AUDITORÍA FINAL
# ==========================================================

# [OPERATIVO]: Genera un reporte detallado en tabla sobre la calidad de la sesión de captura.
# [TÉCNICO]: Input: dict de capturas y dict de mejoras. Output: None (Print a consola).
def imprimir_resumen(capturas: dict, mejoras: dict):
    print(f"\n  {'─'*74}")
    print(f"  {'ANGULO':<32} {'YAW':>6} {'PITCH':>6} {'NIT':>7} "
          f"{'FRAME':>6} {'DIST_IDEAL':>10} {'MEJORAS':>7}  OK")
    print(f"  {'─'*74}")
    for tag in ORDEN_POSES:
        cfg  = POSES_OBJETIVO[tag]
        dato = capturas[tag]
        if dato:
            # [OPERATIVO]: Muestra métricas de precisión y nitidez para cada ángulo exitoso.
            # [TÉCNICO]: Formateo de strings con alineación y precisión decimal para legibilidad de telemetría.
            print(f"  {cfg['label']:<32} {dato['yaw']:>+6.1f} {dato['pitch']:>+6.1f} "
                  f"{dato['nitidez']:>7.1f} {dato['frame_num']:>6} "
                  f"{dato['dist']:>10.2f} {mejoras[tag]:>7}  SI")
        else:
            # [OPERATIVO]: Indica visualmente los ángulos que no se pudieron recolectar.
            print(f"  {cfg['label']:<32} {'—':>6} {'—':>6} {'—':>7} "
                  f"{'—':>6} {'—':>10} {'—':>7}  NO")
    print(f"  {'─'*74}")
    print(f"  DIST_IDEAL = distancia al angulo perfecto (menor = mejor)")
    print(f"  MEJORAS    = veces que se reemplazo por un frame mas cercano al ideal")


# ==========================================================
# ENTRY POINT - ORQUESTADOR DEL SISTEMA
# ==========================================================

def main():
    # [OPERATIVO]: Configura la interfaz de línea de comandos (CLI).
    # [TÉCNICO]: Uso de argparse para gestionar parámetros de entrada: ruta de video, base de datos y flags.
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       default=None)
    parser.add_argument("--db_path",     default="data")
    parser.add_argument("--diagnostico", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*64)
    print("  REGISTRO BIOMETRICO POR VIDEO - 5 ANGULOS  v4")
    print("="*64)
    print("  1 PKL por persona con todos los angulos dentro")

    # [OPERATIVO]: Muestra guía rápida de movimientos al usuario si no es modo diagnóstico.
    if not args.diagnostico:
        print(f"\n  Instrucciones:")
        for tag in ORDEN_POSES:
            cfg = POSES_OBJETIVO[tag]
            print(f"    Cara {cfg['cara_num']:>2}  {cfg['instruccion']}")

    # [OPERATIVO]: Validación y limpieza de la ruta del archivo de video.
    # [TÉCNICO]: Manejo de strings para eliminar comillas accidentales y validación de existencia con os.path.exists.
    video_path = args.video
    if not video_path:
        video_path = input("\n  Ruta del video: ").strip().strip('"').strip("'")
    if not os.path.exists(video_path):
        print(f"\nERROR: No encontrado: {video_path}")
        sys.exit(1)

    nombre = None
    if not args.diagnostico:
        # [OPERATIVO]: Proceso de registro de identidad y control de duplicados.
        # [TÉCNICO]: Verificación de archivos PKL pre-existentes para evitar sobreescritura accidental.
        nombre = _pedir_nombre()
        ruta_pkl_existente = os.path.join(args.db_path, "embeddings",
                                          f"{nombre}_embedding.pkl")
        if os.path.exists(ruta_pkl_existente):
            resp = input(f"\n  Ya existe registro para '{nombre}'. Sobreescribir? [s/n]: ").strip().lower()
            if resp not in ("s", "si", "y", "yes"):
                print("  Cancelado.")
                sys.exit(0)

    print()
    # [OPERATIVO]: Inicialización del motor de IA y arranque del proceso de escaneo.
    # [TÉCNICO]: Instancia FaceEngine y llamada al bucle principal de extracción.
    engine   = FaceEngine()
    capturas, mejoras = extraer_angulos(video_path, engine, args.diagnostico)

    # [OPERATIVO]: Salida temprana si solo se requería análisis de telemetría.
    if args.diagnostico:
        print(f"\n  Diagnostico completo.")
        sys.exit(0)

    # [OPERATIVO]: Presentación de resultados y validación de cuota mínima de captura.
    imprimir_resumen(capturas, mejoras)

    completados = sum(1 for v in capturas.values() if v is not None)
    if completados == 0:
        print("\n  ERROR: No se capturo ningun angulo.")
        sys.exit(1)

    # [OPERATIVO]: Manejo de casos donde la galería está incompleta (menos de 5 ángulos).
    if completados < 5:
        print(f"\n  AVISO: Solo {completados}/5 angulos capturados.")
        resp = input("  Guardar igualmente? [s/n]: ").strip().lower()
        if resp not in ("s", "si", "y", "yes"):
            sys.exit(0)

    print()
    # [OPERATIVO]: Persistencia final de datos en disco (Imágenes + Embeddings).
    # [TÉCNICO]: Llamada al módulo de guardado atómico.
    guardados = guardar_resultado(nombre, capturas, args.db_path)

    # [OPERATIVO]: Cierre de sesión y confirmación de rutas de guardado.
    print(f"\n{'='*64}")
    print(f"  REGISTRO COMPLETADO: {nombre}")
    print(f"  1 PKL con {guardados} angulos  ->  data/embeddings/")
    print(f"  {guardados} imagenes           ->  data/raw/{nombre}/")
    print(f"{'='*64}\n")

if __name__ == "__main__":
    main()