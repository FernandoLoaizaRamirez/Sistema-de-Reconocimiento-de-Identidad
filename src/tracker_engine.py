import cv2
import numpy as np

class TrackerEngine:
    def __init__(self):
        self.esta_rastreando = False
        self.nombre_rastreado = "Desconocido"
        self.roi_hist = None
        self.track_window = None
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def iniciar_rastreo(self, frame, bbox, nombre):
        # Aseguramos que las coordenadas estén dentro de la imagen
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Recorte seguro (clamping)
        x = max(0, x1)
        y = max(0, y1)
        w = min(x2, w_frame) - x
        h = min(y2, h_frame) - y

        # Validar que el cuadro tenga un tamaño real
        if w <= 0 or h <= 0:
            self.esta_rastreando = False
            return

        # Intentar el recorte
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0: # Si por alguna razón sigue vacío, abortamos
            self.esta_rastreando = False
            return

        self.track_window = (x, y, w, h)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Filtro de color para piel
        mask = cv2.inRange(hsv_roi, np.array((0., 40., 30.)), np.array((180., 255., 255.)))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        
        self.nombre_rastreado = nombre
        self.esta_rastreando = True

    def actualizar(self, frame):
        if not self.esta_rastreando or self.roi_hist is None:
            return False, None, "Desconocido"

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        # MeanShift busca la densidad de color más cercana
        ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)
        
        x, y, w, h = self.track_window
        bbox_nuevo = [x, y, x + w, y + h]
        
        if w < 10 or h < 10:
            self.esta_rastreando = False
            return False, None, "Desconocido"

        return True, bbox_nuevo, self.nombre_rastreado

    def detener(self):
        self.esta_rastreando = False