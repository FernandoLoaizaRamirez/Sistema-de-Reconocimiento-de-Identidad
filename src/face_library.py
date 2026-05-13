import os
import pickle
import numpy as np

# Máximo de embeddings por sujeto. Coincide con la constante del roadmap.
GALLERY_MAX_SIZE = 5


class FaceLibrary:
    """
    Gestión en RAM de la galería multi-template de embeddings biométricos.

    Estructura interna de face_db:
    {
        'trabajador_1': {
            'gallery': [
                {'embedding': np.ndarray(512,), 'score': float, 'pose_tag': str},
                ...  # hasta GALLERY_MAX_SIZE entradas
            ],
            'update_count': int
        },
        ...
    }
    """

    def __init__(self, path='data/embeddings'):
        self.path = path
        self.face_db = {}
        self.cargar_rostros()

    # ------------------------------------------------------------------
    # CARGA DESDE DISCO
    # ------------------------------------------------------------------

    def cargar_rostros(self):
        """
        Lee todos los .pkl del directorio y los sube a RAM.
        Soporta dos formatos:
          - v2 (nuevo):  pkl con clave 'gallery'  → carga directa
          - v1 (legacy): pkl con clave 'embedding' → envuelve en galería de 1 entrada
        """
        if not os.path.exists(self.path):
            print(f"⚠️  No se encontró {self.path}. Creándola...")
            os.makedirs(self.path, exist_ok=True)
            return

        archivos_pkl = [f for f in os.listdir(self.path) if f.endswith('_embedding.pkl')]
        cargados, errores = 0, 0

        for archivo in archivos_pkl:
            subject_id = archivo.replace('_embedding.pkl', '')
            ruta = os.path.join(self.path, archivo)
            try:
                with open(ruta, 'rb') as f:
                    data = pickle.load(f)

                if data.get('version') == 2:
                    # Formato v2: galería completa ya almacenada
                    self.face_db[subject_id] = {
                        'gallery': data['gallery'],
                        'update_count': data.get('update_count', 0)
                    }
                else:
                    # Formato v1 legacy: un solo embedding → convertir al vuelo
                    emb = data['embedding']
                    emb = emb / np.linalg.norm(emb)  # normalización defensiva
                    self.face_db[subject_id] = {
                        'gallery': [{
                            'embedding': emb,
                            'score': data.get('calidad', 0.0),
                            'pose_tag': 'frontal'  # asumimos frontal para registros legacy
                        }],
                        'update_count': 0
                    }
                cargados += 1

            except Exception as e:
                print(f"❌ Error cargando {archivo}: {e}")
                errores += 1

        print(f"✅ Galería en RAM: {cargados} sujetos cargados"
              f"{f', {errores} errores' if errores else ''}.")

    # ------------------------------------------------------------------
    # BÚSQUEDA (nueva responsabilidad de FaceLibrary)
    # ------------------------------------------------------------------

    def buscar(self, query_embedding: np.ndarray, umbral: float) -> tuple:
        """
        Busca el sujeto más cercano comparando query contra TODA la galería.

        La distancia de un sujeto es el MÍNIMO sobre sus embeddings —
        basta con que UNA entrada de la galería coincida para reconocerlo.

        Operación vectorizada por sujeto: O(N × G) comparaciones,
        donde G ≤ GALLERY_MAX_SIZE = 5.

        Returns:
            (encontrado: bool, subject_id: str | None, distancia: float)
        """
        if not self.face_db:
            return False, None, 2.0

        norm = np.linalg.norm(query_embedding)
        if norm == 0:
            return False, None, 2.0
        q = query_embedding / norm

        mejor_id, mejor_dist = None, np.inf

        for subject_id, record in self.face_db.items():
            gallery = record['gallery']
            if not gallery:
                continue

            # Matriz (G, 512) — todas las entradas del sujeto de una vez
            matrix = np.array([e['embedding'] for e in gallery], dtype=np.float32)
            dists = np.linalg.norm(matrix - q, axis=1)
            min_dist = dists.min()

            if min_dist < mejor_dist:
                mejor_dist = min_dist
                mejor_id = subject_id

        encontrado = mejor_dist < umbral
        return encontrado, mejor_id, float(mejor_dist)

    # ------------------------------------------------------------------
    # GESTIÓN DE GALERÍA
    # ------------------------------------------------------------------

    def agregar_entrada(self, subject_id: str, embedding: np.ndarray,
                        score: float, pose_tag: str) -> bool:
        """
        Inserta o reemplaza una entrada en la galería de un sujeto.

        Estrategia de inserción (en orden de prioridad):
          1. Galería con espacio libre            → insertar directamente.
          2. Pose no representada en la galería   → reemplazar el de menor score
                                                    (maximiza diversidad).
          3. Pose ya existente, score superior    → reemplazar el peor de esa pose.
          4. Ninguna condición cumplida           → no hay cambio, retorna False.

        Returns:
            True si la galería fue modificada.
        """
        emb = embedding / np.linalg.norm(embedding)
        nueva_entrada = {'embedding': emb, 'score': score, 'pose_tag': pose_tag}

        if subject_id not in self.face_db:
            # Sujeto nuevo: crear registro
            self.face_db[subject_id] = {'gallery': [nueva_entrada], 'update_count': 0}
            return True

        galeria = self.face_db[subject_id]['gallery']

        # Caso 1: espacio libre
        if len(galeria) < GALLERY_MAX_SIZE:
            galeria.append(nueva_entrada)
            self.face_db[subject_id]['update_count'] += 1
            return True

        poses_existentes = {e['pose_tag'] for e in galeria}

        # Caso 2: pose nueva → reemplazar el peor score global para diversificar
        if pose_tag not in poses_existentes:
            idx_peor = int(np.argmin([e['score'] for e in galeria]))
            galeria[idx_peor] = nueva_entrada
            self.face_db[subject_id]['update_count'] += 1
            return True

        # Caso 3: misma pose → reemplazar solo si mejora claramente
        MARGEN_UPGRADE = 50.0
        entradas_misma_pose = [(i, e) for i, e in enumerate(galeria)
                               if e['pose_tag'] == pose_tag]
        idx_peor_pose, peor_entry = min(entradas_misma_pose, key=lambda x: x[1]['score'])
        if score > peor_entry['score'] + MARGEN_UPGRADE:
            galeria[idx_peor_pose] = nueva_entrada
            self.face_db[subject_id]['update_count'] += 1
            return True

        return False

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------

    def obtener_nombres(self) -> list:
        return list(self.face_db.keys())

    def stats(self) -> dict:
        """Retorna métricas de salud de la galería para logging."""
        if not self.face_db:
            return {}
        sizes = [len(r['gallery']) for r in self.face_db.values()]
        scores = [e['score'] for r in self.face_db.values() for e in r['gallery']]
        return {
            'total_sujetos': len(sizes),
            'total_embeddings': sum(sizes),
            'galerias_llenas': sum(1 for s in sizes if s == GALLERY_MAX_SIZE),
            'score_medio': round(float(np.mean(scores)), 1),
            'score_min': round(float(np.min(scores)), 1),
            'score_max': round(float(np.max(scores)), 1),
        }