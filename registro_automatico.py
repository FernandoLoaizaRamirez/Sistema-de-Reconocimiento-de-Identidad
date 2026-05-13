import cv2
import numpy as np
import os
import pickle
import time
import threading

# ── CAMBIO 1 ──────────────────────────────────────────────────────────────────
# Se importa FaceLibrary. Ella reemplaza completamente a db_embeddings,
# recargar_db(), _es_registrado() y _actualizar_registro_existente().
# ──────────────────────────────────────────────────────────────────────────────
from src.face_library import FaceLibrary

# ==========================================================
# --- CONFIGURACIÓN TÉCNICA ---
# ==========================================================
FILTRO_YAW_ESTRICTO    = 15.0
FILTRO_PITCH_MIN       = -40.0
FILTRO_PITCH_MAX       = 25.0
FILTRO_NITIDEZ_MIN     = 280.0
FILTRO_BRIGHTNESS_MIN  = 85.0
FILTRO_BRIGHTNESS_MAX  = 220.0
FILTRO_TAMANO_MIN_PX   = 110

UMBRAL_RECONOCIMIENTO  = 1.05
TIEMPO_ESCANEO_SEGUNDOS = 0.8
COOLDOWN_ACTUALIZACION = 60
UMBRAL_SEGURIDAD_DRIFT = 0.85

COLOR_RECONOCIDO   = (0, 255, 0)
COLOR_ESCANEANDO   = (0, 255, 255)
COLOR_RECHAZADO    = (0, 0, 255)
COLOR_ACTUALIZANDO = (255, 128, 0)


# ── HELPER LIBRE ──────────────────────────────────────────────────────────────
# Separado de la clase porque no necesita estado: entrada de pose → etiqueta
# semántica que FaceLibrary usa para diversificar la galería.
# ──────────────────────────────────────────────────────────────────────────────
def _get_pose_tag(yaw: float, pitch: float) -> str:
    if abs(yaw) <= 5 and abs(pitch) <= 10:
        return 'frontal'
    if yaw > 5:
        return 'slight_right'
    if yaw < -5:
        return 'slight_left'
    return 'up' if pitch > 10 else 'down'


class CameraStream:
    """Hilo optimizado para capturar RTSP sin delay acumulado."""

    def __init__(self, source):
        self.stream = cv2.VideoCapture(source)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                continue
            self.ret, self.frame = ret, frame

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


class RegistradorEmpleados_Gold:

    def __init__(self, folder_db="data"):
        from src.face_engine import FaceEngine
        self.engine = FaceEngine()

        self.folder_embeddings = os.path.join(folder_db, "embeddings")
        self.folder_raw        = os.path.join(folder_db, "raw")

        # ── CAMBIO 2 ──────────────────────────────────────────────────────────
        # ELIMINADO: self.db_embeddings = {}
        # REEMPLAZADO POR: FaceLibrary, que gestiona carga, búsqueda y galería.
        # ──────────────────────────────────────────────────────────────────────
        self.library = FaceLibrary(path=self.folder_embeddings)

        # last_update_time se mantiene aquí porque es estado de sesión (cooldown
        # en RAM), no dato persistente que deba vivir en FaceLibrary.
        self.last_update_time = {}

        self.escaneando_persona_nueva = False
        self.tiempo_inicio_escaneo   = 0
        self.candidatos_registro     = []

        os.makedirs(self.folder_embeddings, exist_ok=True)
        os.makedirs(self.folder_raw, exist_ok=True)

        # ── CAMBIO 3 ──────────────────────────────────────────────────────────
        # recargar_db() ya no construye db_embeddings a mano: delega en
        # FaceLibrary.cargar_rostros(), que soporta v1 y v2 transparentemente.
        # ──────────────────────────────────────────────────────────────────────

    # --------------------------------------------------------------------------
    # RECARGA DE BASE DE DATOS
    # --------------------------------------------------------------------------

    def recargar_db(self):
        # ── CAMBIO 4 ──────────────────────────────────────────────────────────
        # ANTES: 10 líneas iterando pkl, normalizando embeddings manualmente.
        # AHORA: una llamada. FaceLibrary hace todo eso internamente.
        # ──────────────────────────────────────────────────────────────────────
        self.library.cargar_rostros()
        s = self.library.stats()
        print(f"\n--- 📂 DB RECARGADA: {s.get('total_sujetos', 0)} empleados | "
              f"{s.get('total_embeddings', 0)} embeddings totales ---")

    # --------------------------------------------------------------------------
    # CALIDAD DE IMAGEN
    # --------------------------------------------------------------------------

    def _get_quality(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[max(0, y1):y2, max(0, x1):x2]
        if roi.size == 0:
            return 0, 0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return gray.mean(), cv2.Laplacian(gray, cv2.CV_64F).var()

    # --------------------------------------------------------------------------
    # LOOP PRINCIPAL DE PROCESAMIENTO
    # --------------------------------------------------------------------------

    def procesar_fuente(self, source_path, modo_batch=False, usar_hilos=False):
        source_path = source_path.strip('"').strip("'")

        cam = CameraStream(source_path).start() if usar_hilos else cv2.VideoCapture(source_path)
        if usar_hilos:
            time.sleep(1)  # buffer warm-up

        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                break

            display_frame = frame.copy()
            rostros = self.engine.procesar_frame(frame)
            alto_video, ancho_video = frame.shape[:2]

            if self.escaneando_persona_nueva:
                if (time.time() - self.tiempo_inicio_escaneo) >= TIEMPO_ESCANEO_SEGUNDOS:
                    self._registrar_mejor_frame()
                    self.escaneando_persona_nueva = False

            for r in rostros:
                bbox      = r['bbox']
                embedding = r['embedding']
                x1, y1, x2, y2 = map(int, bbox)
                p_val, y_val, _ = r.get('pose', (0, 0, 0))
                brillo, nitidez = self._get_quality(frame, bbox)
                ancho_cara      = x2 - x1

                # ── CAMBIO 5 ──────────────────────────────────────────────────
                # ANTES: self._es_registrado(embedding) buscaba en db_embeddings
                #        con un embedding único por sujeto.
                # AHORA: self.library.buscar() compara contra TODA la galería
                #        (hasta 5 embeddings por sujeto) usando distancia L2
                #        vectorizada. Misma firma de retorno → cero cambios
                #        en la lógica que consume el resultado.
                # ──────────────────────────────────────────────────────────────
                es_reg, subject_id, dist = self.library.buscar(
                    embedding, UMBRAL_RECONOCIMIENTO
                )

                if es_reg:
                    self.escaneando_persona_nueva = False
                    score_actual = nitidez - abs(y_val) * 5

                    # ── CAMBIO 6 ──────────────────────────────────────────────
                    # ANTES: comparaba score_actual contra un único 'calidad'
                    #        almacenado y sobreescribía ese único embedding.
                    # AHORA: delega en library.agregar_entrada(), que decide si
                    #        insertar/reemplazar según la estrategia de galería
                    #        (diversidad de pose, margen de score, capacidad).
                    #        La validación anti-drift vive en _intentar_upgrade().
                    # ──────────────────────────────────────────────────────────
                    if ancho_cara >= FILTRO_TAMANO_MIN_PX:
                        upgraded = self._intentar_upgrade(
                            subject_id, frame, bbox, embedding,
                            score_actual, p_val, y_val
                        )
                        if upgraded:
                            color_box = COLOR_ACTUALIZANDO
                            label     = f"UPGRADING: {subject_id}"
                        else:
                            color_box = COLOR_RECONOCIDO
                            label     = f"{subject_id} ({dist:.2f})"
                    else:
                        color_box = COLOR_RECONOCIDO
                        label     = f"{subject_id} ({dist:.2f})"

                else:
                    # ── CAMBIO 7 ──────────────────────────────────────────────
                    # AÑADIDOS los filtros de Pitch y Brillo que estaban
                    # declarados como constantes pero nunca se evaluaban (DT-01,
                    # DT-02 del biometric_spec.md). Ahora el QC Gate es completo.
                    # ──────────────────────────────────────────────────────────
                    cara_cortada = (
                        y1 <= 15 or y2 >= alto_video - 15 or
                        x1 <= 15 or x2 >= ancho_video - 15
                    )
                    f_ancho   = ancho_cara >= FILTRO_TAMANO_MIN_PX
                    f_yaw     = abs(y_val) <= FILTRO_YAW_ESTRICTO
                    f_pitch   = FILTRO_PITCH_MIN <= p_val <= FILTRO_PITCH_MAX   # DT-01 ✅
                    f_nitidez = nitidez >= FILTRO_NITIDEZ_MIN
                    f_brillo  = FILTRO_BRIGHTNESS_MIN <= brillo <= FILTRO_BRIGHTNESS_MAX  # DT-02 ✅

                    if f_ancho and f_yaw and f_pitch and f_nitidez and f_brillo and not cara_cortada:
                        color_box = COLOR_ESCANEANDO
                        label     = "ESCANEANDO..."
                        if not self.escaneando_persona_nueva:
                            self.escaneando_persona_nueva = True
                            self.tiempo_inicio_escaneo    = time.time()
                            self.candidatos_registro      = []

                        self.candidatos_registro.append({
                            'frame'    : frame.copy(),
                            'bbox'     : bbox,
                            'embedding': embedding,
                            'nitidez'  : nitidez,
                            'y_val'    : y_val,
                            'p_val'    : p_val,
                            'score'    : nitidez - abs(y_val) * 5,
                        })
                    else:
                        color_box = COLOR_RECHAZADO
                        if not f_nitidez:
                            label = "BORROSO"
                        elif not f_yaw or not f_pitch:
                            label = "POSICION"
                        elif not f_brillo:
                            label = "ILUMINACION"
                        else:
                            label = "CORTADA"

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(display_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

            if not modo_batch:
                cv2.imshow("SISTEMA BIOMETRICO GOLD",
                           cv2.resize(display_frame, (1280, 720)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cam.stop() if usar_hilos else cam.release()
        if not modo_batch:
            cv2.destroyAllWindows()

    # --------------------------------------------------------------------------
    # REGISTRO DE SUJETO NUEVO
    # --------------------------------------------------------------------------

    def _registrar_mejor_frame(self):
        if not self.candidatos_registro:
            return

        mejor = max(self.candidatos_registro, key=lambda x: x['score'])

        # ── CAMBIO 8 ──────────────────────────────────────────────────────────
        # ANTES: _es_registrado() → si ya existe, abortar.
        # AHORA: library.buscar() con el mismo umbral. Misma semántica,
        #        pero compara contra la galería completa.
        # ──────────────────────────────────────────────────────────────────────
        ya_existe, _, _ = self.library.buscar(mejor['embedding'], UMBRAL_RECONOCIMIENTO)
        if ya_existe:
            return

        # ── CAMBIO 9 ──────────────────────────────────────────────────────────
        # ANTES: nuevo_id = f"trabajador_{len(self.db_embeddings) + 1}"
        #        → colisionaba al borrar registros (DT-04).
        # AHORA: ID basado en timestamp → único, sin colisiones, cronológico.
        # ──────────────────────────────────────────────────────────────────────
        nuevo_id  = f"trabajador_{int(time.time())}"
        pose_tag  = _get_pose_tag(mejor['y_val'], mejor['p_val'])

        # ── CAMBIO 10 ─────────────────────────────────────────────────────────
        # ANTES: _save_to_disk() → luego db_embeddings[nuevo_id] = {...}
        #        Dos operaciones separadas que podían quedar inconsistentes.
        # AHORA: primero RAM (library.agregar_entrada), luego disco atómico.
        #        Si el guardado falla, la entrada en RAM es consistente para
        #        la sesión actual; en el peor caso se pierde al reiniciar.
        # ──────────────────────────────────────────────────────────────────────
        self.library.agregar_entrada(
            nuevo_id, mejor['embedding'], mejor['score'], pose_tag
        )
        self._save_subject_to_disk(nuevo_id)
        self._save_crop(nuevo_id, mejor['frame'], mejor['bbox'], pose_tag)

        print(f"🏆 REGISTRO EXITOSO: {nuevo_id} | pose={pose_tag} | "
              f"score={mejor['score']:.1f}")

    # --------------------------------------------------------------------------
    # UPGRADE DE ENTRADA EXISTENTE
    # --------------------------------------------------------------------------

    def _intentar_upgrade(self, subject_id, frame, bbox,
                          embedding, score, p_val, y_val) -> bool:
        """
        Intenta mejorar la galería de un sujeto ya reconocido.

        Separado del loop principal por claridad y para que el anti-drift
        quede encapsulado aquí, no disperso en el bucle de frames.

        Returns:
            True si la galería fue efectivamente modificada.
        """
        # ── CAMBIO 11 ─────────────────────────────────────────────────────────
        # Cooldown idéntico al original, pero ahora protege la llamada a
        # library.agregar_entrada() en lugar de _save_to_disk() directamente.
        # ──────────────────────────────────────────────────────────────────────
        now = time.time()
        if now - self.last_update_time.get(subject_id, 0) < COOLDOWN_ACTUALIZACION:
            return False

        # ── CAMBIO 12 ─────────────────────────────────────────────────────────
        # Anti-drift: validar que el nuevo embedding pertenezca al mismo sujeto
        # antes de modificar la galería. En multi-template, comparamos contra
        # el best_embedding (el de mayor score) como ancla de identidad.
        # ──────────────────────────────────────────────────────────────────────
        record = self.library.face_db.get(subject_id)
        if record is None:
            return False

        emb_norm   = embedding / np.linalg.norm(embedding)
        best_emb   = max(record['gallery'], key=lambda e: e['score'])['embedding']
        dist_drift = np.linalg.norm(emb_norm - best_emb)

        if dist_drift > UMBRAL_SEGURIDAD_DRIFT:
            return False  # Posible impostor o cambio brusco de apariencia

        pose_tag = _get_pose_tag(y_val, p_val)

        # ── CAMBIO 13 ─────────────────────────────────────────────────────────
        # ANTES: sobreescribía el único embedding si score mejoraba en +50.
        # AHORA: library.agregar_entrada() aplica la estrategia de galería
        #        (diversidad de pose + margen de score). Si no hay mejora
        #        real, retorna False y no se toca nada.
        # ──────────────────────────────────────────────────────────────────────
        modificado = self.library.agregar_entrada(subject_id, embedding, score, pose_tag)

        if modificado:
            self._save_subject_to_disk(subject_id)
            self._save_crop(subject_id, frame, bbox, pose_tag)
            self.last_update_time[subject_id] = now
            print(f"🔄 MEJORA APLICADA: {subject_id} | pose={pose_tag} | "
                  f"score={score:.1f}")

        return modificado

    # --------------------------------------------------------------------------
    # PERSISTENCIA
    # --------------------------------------------------------------------------

    def _save_subject_to_disk(self, subject_id: str):
        """
        Serializa el registro completo (galería) de un sujeto a disco.

        ── CAMBIO 14 ─────────────────────────────────────────────────────────
        ANTES: _save_to_disk() guardaba {'embedding': ..., 'calidad': ...} —
               formato v1, un único embedding.
        AHORA: guarda {'version': 2, 'gallery': [...], ...} — formato v2
               con la galería completa. La escritura sigue siendo atómica
               (tmp → replace) para resistir cortes de corriente.
        ─────────────────────────────────────────────────────────────────────
        """
        record = self.library.face_db.get(subject_id)
        if record is None:
            return

        path_pkl = os.path.join(self.folder_embeddings, f"{subject_id}_embedding.pkl")
        payload  = {
            'version'     : 2,
            'gallery'     : record['gallery'],   # lista de dicts con embedding/score/pose_tag
            'update_count': record['update_count'],
            'saved_at'    : time.time(),
        }
        tmp = path_pkl + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path_pkl)  # Atómico en POSIX / Windows (Python ≥ 3.3)

        # Guardar crop de referencia visual (mejor frame de la galería)
        # — solo cuando se llama desde _registrar_mejor_frame con frame disponible.
        # El upgrade no actualiza el crop para no depender de frame en este método.

    def _save_crop(self, subject_id: str, frame, bbox, pose_tag: str):
        """Guarda el recorte facial de referencia visual (JPG) con tag de pose."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        crop = frame[max(0, y1 - 40):min(h, y2 + 40),
                     max(0, x1 - 40):min(w, x2 + 40)]
        
        filename = f"{subject_id}_{pose_tag}.jpg"
        cv2.imwrite(os.path.join(self.folder_raw, filename), crop)

    # --------------------------------------------------------------------------
    # MÉTODOS ELIMINADOS — referencia para revisión de código
    # --------------------------------------------------------------------------
    #
    # ❌ recargar_db()                  → sustituido por self.library.cargar_rostros()
    # ❌ _es_registrado()               → sustituido por self.library.buscar()
    # ❌ _actualizar_registro_existente()→ sustituido por _intentar_upgrade()
    #                                     + self.library.agregar_entrada()
    # ❌ _save_to_disk() (formato v1)   → sustituido por _save_subject_to_disk()
    #                                     (formato v2) + _save_crop()
    #
    # --------------------------------------------------------------------------


# ==========================================================
# --- MENÚ PRINCIPAL ---
# ==========================================================
if __name__ == "__main__":
    reg = RegistradorEmpleados_Gold()

    RTSP_110 = "rtsp://LoaizaStream:YfPvHbQX68zBCRJ@192.168.30.110:554/Streaming/Channels/101"
    RTSP_111 = "rtsp://LoaizaStream:YfPvHbQX68zBCRJ@192.168.30.111:554/Streaming/Channels/101"

    while True:
        print("\n" + "=" * 50)
        print("  SISTEMA DE REGISTRO AUTOMÁTICO (BIOMETRÍA GOLD)")
        print("=" * 50)

        # ── CAMBIO 15 ─────────────────────────────────────────────────────────
        # ANTES: len(reg.db_embeddings) → contaba sujetos desde dict interno.
        # AHORA: stats() de FaceLibrary da métricas enriquecidas sin exponer
        #        internals. Se muestra también total de embeddings en galería.
        # ──────────────────────────────────────────────────────────────────────
        s = reg.library.stats()
        print(f" Empleados: {s.get('total_sujetos', 0)} | "
              f"Embeddings: {s.get('total_embeddings', 0)} | "
              f"Galerías llenas: {s.get('galerias_llenas', 0)}")
        print("-" * 50)
        print("1. Cámara 110 (RTSP Hilos)")
        print("2. Cámara 111 (RTSP Hilos)")
        print("3. Analizar Video Local")
        print("4. Procesar Carpeta de Videos (Batch)")
        print("r. Recargar Base de Datos")
        print("q. Salir")
        print("-" * 50)

        opc = input("Seleccione una opción: ").lower()

        if opc == '1':
            reg.procesar_fuente(RTSP_110, usar_hilos=True)
        elif opc == '2':
            reg.procesar_fuente(RTSP_111, usar_hilos=True)
        elif opc == '3':
            ruta = input("Introduce la ruta del video: ").strip()
            reg.procesar_fuente(ruta, usar_hilos=False)
        elif opc == '4':
            ruta_folder = input("Ruta de la carpeta: ").strip('"').strip("'")
            if os.path.exists(ruta_folder):
                videos = [f for f in os.listdir(ruta_folder)
                          if f.lower().endswith(('.mp4', '.avi', '.mkv'))]
                print(f"🎥 Encontrados {len(videos)} videos. Iniciando proceso batch...")
                for v in videos:
                    print(f"👉 Procesando: {v}")
                    reg.procesar_fuente(os.path.join(ruta_folder, v),
                                        modo_batch=True, usar_hilos=False)
                reg.recargar_db()
            else:
                print("❌ Carpeta no encontrada.")
        elif opc == 'r':
            reg.recargar_db()
        elif opc == 'q':
            print("Saliendo del sistema...")
            break
        else:
            print("⚠️ Opción no válida.")
