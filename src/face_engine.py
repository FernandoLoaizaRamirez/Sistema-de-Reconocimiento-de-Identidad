import cv2
import numpy as np
from insightface.app import FaceAnalysis
import warnings

# [OPERATIVO]: Silencia avisos de depreciación de librerías subyacentes para una consola limpia.
warnings.filterwarnings("ignore", category=FutureWarning)

class FaceEngine:
    # [OPERATIVO]: Constructor de la clase. Inicializa y carga en memoria los modelos de IA.
    # [TÉCNICO]: Configura InsightFace con el paquete 'buffalo_l', priorizando aceleración por hardware (CUDA).
    def __init__(self, ctx_id=0):
        # Cambiamos 'antelopev2' por 'buffalo_l'
        nombre_modelo = 'buffalo_l' 
        print(f"--- 🔄 CARGANDO MODELO: {nombre_modelo} ---")
        
        # [TÉCNICO]: providers define el orden de ejecución: primero intenta GPU (CUDA), si falla usa CPU.
        self.app = FaceAnalysis(name=nombre_modelo, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # [TÉCNICO]: det_size define la resolución de entrada para el detector; (640, 640) es el estándar de equilibrio.
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        # [OPERATIVO]: Auditoría de carga para confirmar que todos los sub-componentes (det, rec, pose) están listos.
        for model in self.app.models:
            print(f"📦 Sub-modelo cargado: {model}")

    # [OPERATIVO]: Analiza una imagen única (frame) para extraer toda la información facial disponible.
    # [TÉCNICO]: Input: ndarray (BGR). Output: list(dict). Retorna coordenadas, vectores matemáticos y pose.
    def procesar_frame(self, frame):
        # [OPERATIVO]: Validación de integridad de la imagen de entrada.
        if frame is None or frame.size == 0:
            return []
        
        # [TÉCNICO]: Ejecución de la inferencia completa: Detección + Alineación + Reconocimiento + Pose.
        faces = self.app.get(frame)
        resultados = []
        
        for face in faces:
            # [OPERATIVO]: Calcula el tamaño físico del rostro detectado para filtros de distancia.
            bbox = face.bbox.astype(int)
            ancho = bbox[2] - bbox[0]
            alto = bbox[3] - bbox[1]
            
            # [OPERATIVO]: Extrae los ángulos de Euler: Pitch (inclinación), Yaw (giro), Roll (rotación).
            p, y, r = face.pose
            
            # [TÉCNICO]: Estructura de datos normalizada para el consumo del resto de funciones del script.
            resultados.append({
                'bbox': bbox, 
                'embedding': face.normed_embedding, # Vector de 512 dimensiones normalizado.
                'res': (ancho, alto),
                'pose': (p, y, r)
            })
        return resultados

   