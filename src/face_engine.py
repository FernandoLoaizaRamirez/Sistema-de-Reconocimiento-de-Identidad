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

import numpy as np
import warnings
from insightface.app import FaceAnalysis

warnings.filterwarnings("ignore", category=FutureWarning)


class FaceEngine:
    """
    Envuelve FaceAnalysis (buffalo_l) con una interfaz limpia.
    Siempre intenta GPU (CUDAExecutionProvider); si no está disponible,
    ONNX Runtime cae automáticamente a CPU.
    """

    def __init__(self, det_size: int = 640):
        """
        Parameters
        ----------
        det_size : int
            Resolución del detector (cuadrada).
            640  → rápido, detecta rostros desde ~80 px de ancho.
            1280 → más lento, detecta rostros desde ~40 px (cámaras lejanas).
        """
        self._det_size = det_size
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))
        print(f"FaceEngine listo — buffalo_l  det_size={det_size}x{det_size}  [GPU→CPU fallback]")

    # ------------------------------------------------------------------

    def procesar_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Ejecuta detección + alineación + reconocimiento + pose sobre un frame.

        Parameters
        ----------
        frame : np.ndarray  BGR  (H, W, 3)

        Returns
        -------
        list[dict]  — vacío si no hay rostros o frame inválido.
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

            bbox  = face.bbox.astype(int)
            ancho = int(bbox[2] - bbox[0])
            alto  = int(bbox[3] - bbox[1])

            try:
                pitch, yaw, roll = face.pose
            except Exception:
                pitch, yaw, roll = 0.0, 0.0, 0.0

            resultados.append({
                "bbox"     : bbox,
                "embedding": emb.astype(np.float32),
                "res"      : (ancho, alto),
                "pose"     : (float(pitch), float(yaw), float(roll)),
            })

        return resultados