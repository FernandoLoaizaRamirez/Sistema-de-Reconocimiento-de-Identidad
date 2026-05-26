# Sistema de Reconocimiento de Identidad Facial

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework: InsightFace](https://img.shields.io/badge/Framework-InsightFace-orange.svg)](https://insightface.ai/)

Un sistema avanzado de reconocimiento facial biométrico diseñado para la identificación de sujetos en entornos dinámicos con variabilidad de iluminación y pose. Basado en el modelo de última generación `buffalo_l` (InsightFace) y optimizado para inferencia en tiempo real mediante aceleración por hardware (CUDA).

## 🚀 Características Principales

*   **Reconocimiento en Tiempo Real:** Procesamiento de flujos de video (RTSP o locales) con baja latencia.
*   **Registro Biométrico Multi-ángulo:** Protocolo de registro de 6 poses para maximizar el recall ante variaciones de cabeza.
*   **Arquitectura Vectorizada:** Búsqueda de identidad mediante operaciones matriciales NumPy (Distancia L2) para máxima eficiencia.
*   **Validación Ground-Truth:** Herramientas integradas para evaluar el desempeño contra datos etiquetados (formato MOT).
*   **Robustez Ambiental:** Diseñado para manejar condiciones de iluminación variable y poses extremas (hasta 87° de yaw).
*   **Fallback Inteligente:** Soporte nativo para GPU NVIDIA (CUDA) con cambio automático a CPU si es necesario.

## 🏗️ Arquitectura del Sistema

El motor utiliza un pipeline de cuatro etapas optimizado:
1.  **Detección (SCRFD):** Localización de rostros con resolución adaptable (640px/1280px).
2.  **Alineación Facial:** Normalización geométrica mediante 5 landmarks faciales.
3.  **Extracción de Embeddings (ArcFace R100):** Generación de vectores de identidad de 512 dimensiones.
4.  **Comparación y Decisión:** Clasificación probabilística basada en umbrales de similitud calibrados.

## 🛠️ Instalación

### Requisitos Previos
*   Python 3.10 o superior.
*   Controladores NVIDIA y CUDA Toolkit (opcional, para aceleración GPU).

### Configuración del Entorno
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/FernandoLoaizaRamirez/Sistema-de-Reconocimiento-de-Identidad.git
   cd Sistema-de-Reconocimiento-de-Identidad
   ```

2. Instalar dependencias:
   *   **Windows:**
       ```bash
       pip install -r requirements_win.txt
       ```
   *   **Linux:**
       ```bash
       pip install -r requirements_linux.txt
       ```

## 📖 Uso

### 1. Registro de Personas
Utilice un video del sujeto para extraer los mejores embeddings desde 6 ángulos diferentes:
```bash
python registro_por_video.py --video "ruta/al/video.mp4"
```

### 2. Ejecución del Reconocedor
Inicie el sistema en tiempo real para procesar cámaras IP o videos locales:
```bash
python reconocedor.py
```
*   *Nota: Use `--det_size 1280` para mejorar la detección en cámaras distantes.*

### 3. Validación de Métricas
Compare el desempeño del sistema contra un archivo de Ground Truth (MOT):
```bash
python validador.py --video video.mp4 --gt ground_truth.txt --video_id 1
```

## 📊 Desempeño Operativo

*   **Precisión:** >98% en condiciones controladas de oficina.
*   **Recall:** >80% con cobertura de galería multi-pose completa.
*   **Velocidad de Inferencia:** ~6.6 FPS en hardware de gama media (RTX 4050).
*   **Límites de Detección:** Hasta 89° de rotación lateral (Yaw).

## 📂 Estructura del Repositorio

*   `reconocedor.py`: Script principal de inferencia en tiempo real.
*   `registro_por_video.py`: Herramienta de enrolamiento biométrico.
*   `validador.py`: Suite de pruebas y métricas de desempeño.
*   `src/face_engine.py`: Wrapper principal del motor de InsightFace.
*   `documentacion_tecnica.md`: Reporte detallado de evaluación y límites operativos.

## ⚖️ Licencia

Este proyecto está bajo la Licencia MIT. Consulte el archivo `LICENSE` para más detalles.
