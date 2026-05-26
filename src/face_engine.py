"""
face_engine.py — Motor de inferencia facial
============================================
Modelo : buffalo_l  (InsightFace / ONNX)
Hardware: CUDA GPU por defecto, CPU como fallback automático
Salida  : lista de dicts normalizados por frame

Cada dict contiene:
  bbox      : np.ndarray int [x1, y1, x2, y2]
  embedding : np.ndarray float32 (512,)  normalizado L2
  res       : (ancho_px, alto_px)
  pose      : (pitch, yaw, roll) en grados
"""

import warnings

import numpy as np
from insightface.app import FaceAnalysis

warnings.filterwarnings("ignore", category=FutureWarning)


class FaceEngine:
    """Motor de inferencia facial que envuelve el modelo buffalo_l de InsightFace.

    Intenta utilizar GPU (CUDA) por defecto y cambia automáticamente a CPU si no hay
    hardware compatible disponible. Proporciona una interfaz simplificada para
    obtener detecciones, embeddings y poses.

    Attributes:
        _det_size (int): Resolución del detector configurada.
        app (FaceAnalysis): Instancia del motor de análisis facial de InsightFace.
    """

    def __init__(self, det_size: int = 640):
        """Inicializa el motor de inferencia.

        Args:
            det_size (int): Resolución del detector (cuadrada).
                640 → rápido, recomendado para rostros cercanos.
                1280 → más preciso, útil para cámaras lejanas o rostros pequeños.
        """
        self._det_size = det_size
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))
        print(
            f"FaceEngine listo — buffalo_l  det_size={det_size}x{det_size}  [GPU→CPU fallback]"
        )

    # ------------------------------------------------------------------

    def procesar_frame(self, frame: np.ndarray) -> list[dict]:
        """Ejecuta detección, alineación, reconocimiento y estimación de pose.

        Args:
            frame (np.ndarray): Imagen en formato BGR (H, W, 3).

        Returns:
            list[dict]: Lista de diccionarios con la información de cada rostro detectado.
                Cada diccionario contiene 'bbox', 'embedding', 'res' y 'pose'.
                Retorna una lista vacía si no hay rostros o el frame es inválido.
        """
        if frame is None or frame.size == 0:
            return []

        try:
            faces = self.app.get(frame)
        except Exception as e:
            print(f"  [FaceEngine] Error en inferencia: {e}")
            return []

        resultados = []
        for face in faces:
            emb = face.normed_embedding
            if emb is None:
                continue

            bbox = face.bbox.astype(int)
            ancho = int(bbox[2] - bbox[0])
            alto = int(bbox[3] - bbox[1])

            try:
                pitch, yaw, roll = face.pose
            except Exception:
                pitch, yaw, roll = 0.0, 0.0, 0.0

            resultados.append(
                {
                    "bbox": bbox,
                    "embedding": emb.astype(np.float32),
                    "res": (ancho, alto),
                    "pose": (float(pitch), float(yaw), float(roll)),
                }
            )

        return resultados
