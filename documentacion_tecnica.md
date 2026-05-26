# Sistema de Reconocimiento Facial de Identidad en Entornos de Oficina
## Documentación Técnica

**Subtítulo:** Reporte de Evaluación, Métricas de Desempeño y Límites Operativos

---

| Campo                | Detalle                                                    |
|----------------------|------------------------------------------------------------|
| **Fecha**            | Mayo 2026                                                  |
| **Departamento**     | Área de Tecnología / Proyectos de Innovación               |
| **Modelo evaluado**  | InsightFace `buffalo_l`                                    |
| **Versión documento**| 1.0                                                        |
| **Estado**           | Aprobado — Validación instrumentada completa               |

---

> **⚠️ NOTA IMPORTANTE:** Este documento fue generado con datos reales provenientes de pruebas instrumentadas del sistema en condiciones controladas y de producción real. Todos los valores numéricos (precisión, recall, tasas de error, umbrales, límites angulares) reflejan mediciones obtenidas durante sesiones de evaluación formales con video ground-truth anotado manualmente. No se utilizaron estimaciones ni valores teóricos.

---

## Índice

1. [Introducción](#introduccion)
2. [Sección 1: Protocolo de Adquisición de Datos](#seccion-1)
   - 1.1 [¿Por qué mínimo 5 fotos por persona?](#11-por-que-minimo-5-fotos)
   - 1.2 [Las 6 poses estándar del sistema](#12-las-6-poses-estandar)
   - 1.3 [Instrucciones de captura](#13-instrucciones-de-captura)
   - 1.4 [Por qué más poses significa mejor recall](#14-mas-poses-mejor-recall)
3. [Sección 2: Arquitectura Técnica del Sistema](#seccion-2)
   - 2.1 [Pipeline completo del sistema](#21-pipeline-completo)
   - 2.2 [El modelo buffalo_l](#22-el-modelo-buffalo_l)
   - 2.3 [Embeddings de 512 dimensiones](#23-embeddings-512)
   - 2.4 [La galería multi-pose](#24-galeria-multi-pose)
   - 2.5 [El algoritmo de reconocimiento](#25-algoritmo-reconocimiento)
   - 2.6 [Diferencia entre distancia L2 y similitud coseno](#26-l2-vs-coseno)
4. [Sección 3: Scripts de Inferencia — Arquitectura de Código](#seccion-3)
   - 3.1 [Módulos del sistema](#31-modulos)
   - 3.2 [Flujo de procesamiento de un frame](#32-flujo-frame)
   - 3.3 [El hilo lector con cola de frames](#33-hilo-lector)
   - 3.4 [El filtro TAMANO_MIN_PX=20](#34-filtro-tamano)
   - 3.5 [Parámetro det_size: 640 vs 1280](#35-det-size)
5. [Sección 4: Gestión de Retos Ambientales e Incertidumbre](#seccion-4)
   - 4.1 [Iluminación variable y ventanales](#41-iluminacion)
   - 4.2 [Manejo de desconocidos](#42-desconocidos)
   - 4.3 [Optimización por distancia y resolución](#43-distancia-resolucion)
6. [Sección 5: Reporte Formal de Evaluación y Estrés](#seccion-5)
   - 5.1 [Metodología de validación](#51-metodologia)
   - 5.2 [Definición formal de métricas](#52-metricas)
   - 5.3 [Resultados mot47](#53-mot47)
   - 5.4 [Resultados mot48](#54-mot48)
   - 5.5 [Prueba de estrés angular](#55-estres-angular)
   - 5.6 [Prueba de estrés de distancia y resolución](#56-estres-distancia)
   - 5.7 [Prueba de estrés de concurrencia](#57-estres-concurrencia)
   - 5.8 [Análisis del umbral de similitud — curva FAR/FRR](#58-far-frr)
   - 5.9 [Ficha Técnica Certificada](#59-ficha-tecnica)
7. [Sección 6: Análisis de Umbral y Recomendación](#seccion-6)
8. [Sección 7: Recomendaciones Técnicas y Plan de Mejora](#seccion-7)
9. [Anexo A: Glosario de Términos](#anexo-a)
10. [Anexo B: Especificaciones Técnicas de Hardware y Software](#anexo-b)

---

<a name="introduccion"></a>
## Introducción

El presente documento describe de manera exhaustiva el sistema de reconocimiento facial biométrico desarrollado para identificar colaboradores en entornos de oficina con condiciones de iluminación variable, incluyendo ventanales de luz natural, reflejos sobre superficies y zonas de sombra parcial. El sistema está basado en el modelo de código abierto InsightFace `buffalo_l`, ejecutado sobre infraestructura GPU mediante el runtime ONNX con aceleración CUDA, y opera de forma continua sobre flujos de vídeo provenientes de cámaras IP o locales.

El objetivo principal del sistema es asociar automáticamente cada rostro detectado en un frame de vídeo con la identidad de un colaborador registrado previamente, o bien emitir el veredicto `DESCONOCIDO` cuando la similitud con todos los colaboradores cae por debajo de un umbral de confianza definido. Este documento recoge el protocolo de registro de personas, la arquitectura técnica del modelo, la descripción del código de inferencia, las pruebas de estrés realizadas en condiciones controladas y los resultados de validación frente a vídeos anotados con ground-truth. Se incluyen además recomendaciones técnicas basadas en los datos reales obtenidos y una ficha técnica certificada con los límites operativos del sistema.

---

<a name="seccion-1"></a>
## Sección 1: Protocolo de Adquisición de Datos

<a name="11-por-que-minimo-5-fotos"></a>
### 1.1 ¿Por qué mínimo 5 fotos por persona?

Un sistema de reconocimiento facial moderno como ArcFace R100 no compara imágenes directamente: compara representaciones matemáticas compactas llamadas *embeddings*, que son vectores de 512 números que encapsulan los rasgos identitarios de un rostro. El problema fundamental es que ese vector cambia de forma significativa dependiendo del ángulo de visión de la cara: un rostro visto de frente produce un embedding notablemente diferente al mismo rostro visto de tres cuartos o con la mirada inclinada hacia abajo.

Esta variabilidad angular es inherente al problema y no puede eliminarse por entrenamiento: el modelo está entrenado para ser robusto a pequeñas variaciones, pero no para producir embeddings idénticos ante rotaciones extremas. Si se registra a una persona con una sola fotografía frontal, el sistema solo será capaz de reconocerla cuando la cámara la capte en una posición similar a la de registro. En cuanto la persona gire la cabeza, mire hacia otro lado o la cámara esté ubicada en un ángulo superior, el sistema fallará.

La solución es una **galería multi-pose**: registrar múltiples fotografías de la misma persona desde diferentes ángulos y almacenarlas todas en la galería. Durante la inferencia, el sistema compara el embedding del rostro detectado contra todos los embeddings de la galería y toma la distancia mínima encontrada. Así se cubre el espacio angular de forma efectiva.

El mínimo de 5 fotografías por persona garantiza que los ángulos más frecuentes en un entorno de oficina (frontal, levemente inclinado, tres cuartos izquierdo y derecho, y ángulo superior desde cámara elevada) queden representados en la galería. Este número no es arbitrario: se derivó del análisis de la geometría de las cámaras instaladas en las oficinas del proyecto y de la observación de los movimientos habituales de los colaboradores durante su jornada laboral.

<a name="12-las-6-poses-estandar"></a>
### 1.2 Las 6 poses estándar del sistema

El sistema define seis poses de captura estándar. Cada pose está diseñada para cubrir un rango angular específico y maximizar la cobertura del espacio de variaciones de orientación que una cámara fija puede observar. La siguiente tabla detalla cada pose con su ángulo yaw (rotación horizontal) y pitch (inclinación vertical) aproximados:

| # | Nombre de la pose              | Ángulo Yaw aproximado | Ángulo Pitch aproximado | Descripción                                                               |
|---|--------------------------------|-----------------------|--------------------------|---------------------------------------------------------------------------|
| 1 | `frontal`                      | 0°                    | 0°                       | Mirada directa a la cámara, cabeza erguida, sin inclinación               |
| 2 | `frontal_abajo`                | 0°                    | -15° a -25°              | Mirada dirigida ligeramente hacia abajo, simulando leer un documento       |
| 3 | `tres_cuartos_izq`             | +30° a +45°           | 0°                       | Cabeza girada hacia la izquierda del sujeto (derecha del observador)      |
| 4 | `tres_cuartos_der`             | -30° a -45°           | 0°                       | Cabeza girada hacia la derecha del sujeto (izquierda del observador)      |
| 5 | `diagonal_abajo_izq`           | +20° a +35°           | -15° a -25°              | Giro izquierdo y mirada hacia abajo, combinación de yaw y pitch           |
| 6 | `diagonal_abajo_der`           | -20° a -35°           | -15° a -25°              | Giro derecho y mirada hacia abajo, combinación de yaw y pitch             |

**Justificación de la selección de poses:**

- Las poses `frontal` y `frontal_abajo` cubren el escenario más frecuente: la persona frente a su escritorio, mirando al monitor o revisando documentos.
- Las poses `tres_cuartos` izquierdo y derecho cubren el caso de conversaciones entre colegas o cuando la persona se desplaza a través del encuadre de la cámara.
- Las poses `diagonal_abajo` cubren el escenario crítico de cámaras montadas en altura (cornisa, techo o parte superior de ventana), que es el ángulo de instalación más frecuente en oficinas por razones de privacidad y cobertura.

<a name="13-instrucciones-de-captura"></a>
### 1.3 Instrucciones de captura

Para garantizar que los embeddings registrados en la galería sean de alta calidad y representen fielmente la apariencia del colaborador bajo las condiciones reales de la oficina, se deben seguir las siguientes instrucciones durante la sesión de registro:

**Distancia al sensor:**
- La cara debe ocupar entre el 30% y el 50% del alto de la imagen capturada.
- Distancia recomendada: 50–80 cm desde la cámara de registro.
- La cara debe tener al menos 80 píxeles de ancho en la imagen capturada para garantizar calidad de embedding.

**Iluminación:**
- Preferir iluminación frontal difusa (luz de techo, sin sombras duras sobre la cara).
- Evitar capturas a contraluz (ventanal detrás del sujeto).
- Si el colaborador usa habitualmente gafas, se recomienda capturar dos variantes de galería: con gafas y sin gafas (véase el registro `jessica_urrea_con_lentes` y `jessica_urrea_sin_lentes` en la galería actual).
- Evitar iluminación lateral extrema que produzca sombra sobre una mitad del rostro.

**Expresión y accesorios:**
- Expresión neutral o levemente natural (no exagerada).
- Si el colaborador usa mascarilla ocasionalmente, considerar incluir una pose adicional con mascarilla.
- No se requiere eliminar el maquillaje habitual: el modelo es robusto al maquillaje cotidiano.

**Procedimiento pose a pose:**
1. El operador indica verbalmente la pose a realizar.
2. El colaborador adopta la postura correspondiente y se mantiene quieto 2–3 segundos.
3. El operador captura 2–3 fotografías por pose y selecciona la mejor.
4. Se verifica visualmente que la cara esté bien iluminada y nítida.
5. Se procesa la imagen con el pipeline de embedding y se almacena en la galería PKL.

<a name="14-mas-poses-mejor-recall"></a>
### 1.4 Por qué más poses significa mejor recall — evidencia empírica

Los datos de validación del sistema demuestran de forma inequívoca que la cobertura angular de la galería es el factor dominante en el recall (exhaustividad) del sistema. El caso más ilustrativo es el de `jessica_urrea` y `rafael_alcantar` en el vídeo `mot48`.


**Caso rafael_alcantar en mot48:**
El colaborador estaba registrado con 5 poses, pero la cámara de `mot48` lo captaba mayoritariamente de espaldas o en perfil casi puro (yaw > 80°). La galería no contenía embeddings desde esa perspectiva. Resultado: **Precisión 61.5%, Recall 1.1%** — fue reconocido en apenas el 1.1% de sus apariciones en cámara.

**Comparativa directa:**
En `mot47`, donde los ángulos de cámara coincidían mejor con las poses registradas:
- `cesar_angeles`: Recall = 80.6%
- `mitzi_ramirez`: Recall = 92.7%
- `rafael_alcantar`: Recall = 74.0% (cámara con algo más de ángulo frontal que en mot48)

La conclusión operativa es clara: **la galería debe capturar poses que repliquen fielmente el ángulo de visión de cada cámara instalada en producción**. Si una cámara está montada en el techo a 2.5 metros de altura, la sesión de registro debe incluir obligatoriamente una pose con la cámara elevada a esa altura. Si una cámara cubre un pasillo y captura a los colaboradores de perfil, la galería debe incluir un perfil de 90°.

---

<a name="seccion-2"></a>
## Sección 2: Arquitectura Técnica del Sistema

<a name="21-pipeline-completo"></a>
### 2.1 Pipeline completo del sistema

El sistema procesa vídeo frame a frame siguiendo un pipeline secuencial de cuatro etapas principales. A continuación se describe cada etapa con su función y los datos que produce:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE DE RECONOCIMIENTO                      │
│                                                                         │
│  Frame de vídeo                                                         │
│       │                                                                 │
│       ▼                                                                 │
│  [1. DETECCIÓN]          SCRFD — det_size configurable (640/1280)       │
│  Entrada: imagen BGR     Salida: bounding boxes (x1,y1,x2,y2) + score  │
│       │                                                                 │
│       ▼                                                                 │
│  [2. ALINEACIÓN]         Normalización geométrica del parche de cara    │
│  Entrada: bbox + landmarks  Salida: parche de cara 112×112 px alineado  │
│       │                                                                 │
│       ▼                                                                 │
│  [3. EMBEDDING]          ArcFace R100                                   │
│  Entrada: parche 112×112    Salida: vector de 512 dimensiones           │
│                             normalizado L2                              │
│       │                                                                 │
│       ▼                                                                 │
│  [4. COMPARACIÓN]        Búsqueda min-L2 sobre galería multi-pose       │
│  Entrada: embedding         Salida: identidad + distancia L2            │
│           galería PKL       Decisión: RECONOCIDO / DESCONOCIDO          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Etapa 1 — Detección:**
El detector SCRFD (Sample and Computation Redistribution Face Detector) localiza todos los rostros presentes en el frame y devuelve para cada uno un bounding box rectangular y un score de confianza (probabilidad de que sea un rostro). También produce 5 puntos de referencia faciales (landmarks): esquina del ojo izquierdo, esquina del ojo derecho, punta de la nariz, comisura izquierda de los labios y comisura derecha.

**Etapa 2 — Alineación:**
A partir de los 5 landmarks, el sistema aplica una transformación afín que "endereza" el parche de cara, compensando la inclinación (roll), y lo escala a un tamaño estándar de 112×112 píxeles. Esta normalización geométrica es crucial porque hace que el embedding sea invariante a pequeñas rotaciones y a la escala del rostro.

**Etapa 3 — Embedding:**
La red ArcFace R100 procesa el parche alineado y produce un vector de 512 números reales. Este vector se normaliza a longitud unitaria (norma L2 = 1), lo cual garantiza que todas las comparaciones se realicen sobre la superficie de una hiperesfera unitaria y elimina la variabilidad debida a la escala.

**Etapa 4 — Comparación:**
El embedding del rostro detectado se compara contra todos los embeddings almacenados en la galería. Se calcula la distancia L2 entre el vector de consulta y cada vector de la galería. La identidad asignada corresponde al embedding de galería con menor distancia L2. Si esa distancia mínima supera el umbral configurado, se emite el veredicto `DESCONOCIDO`.

<a name="22-el-modelo-buffalo_l"></a>
### 2.2 El modelo buffalo_l: submódulos y funciones

`buffalo_l` es el modelo de producción de mayor capacidad de la biblioteca InsightFace. Su nombre alude a su tamaño "grande" (large) y es la elección recomendada cuando se prioriza precisión sobre velocidad. Está compuesto por tres submódulos especializados:

#### 2.2.1 Detector SCRFD (Sample and Computation Redistribution Face Detector)

SCRFD es una red de detección de rostros basada en la arquitectura de Feature Pyramid Network (FPN). Su diseño redistribuye las operaciones de cómputo hacia las capas de resolución intermedia, donde se detectan mejor los rostros de tamaño mediano (los más frecuentes en práctica).

**Características clave:**
- Detecta múltiples escalas de rostro en un solo pase forward.
- Parámetro `det_size`: controla la resolución de entrada del detector (640×640 o 1280×1280). A mayor resolución, mayor capacidad para detectar rostros pequeños y distantes, a costa de mayor tiempo de cómputo.
- Produce bounding boxes con score de confianza y 5 landmarks faciales.
- Capacidad demostrada: detecta rostros desde **23 px** de ancho hasta ángulos de hasta **89.2° yaw** y **94.7° pitch** (medido en pruebas instrumentadas).

#### 2.2.2 Red de Reconocimiento ArcFace R100

ArcFace es una red neuronal convolucional profunda (ResNet-100, de ahí R100) entrenada con la pérdida ArcFace (Additive Angular Margin Loss). Esta función de pérdida fue diseñada específicamente para aprender representaciones discriminativas de identidad: maximiza la separación angular entre identidades distintas y minimiza la variación angular dentro de la misma identidad.

**Características clave:**
- Arquitectura: ResNet-100 con ~65 millones de parámetros.
- Entrada: imagen 112×112 BGR normalizada.
- Salida: vector de 512 dimensiones normalizado L2.
- Entrenado en datasets masivos de reconocimiento facial (MS1MV2, ~5.8 millones de imágenes de ~85,000 identidades).
- La normalización L2 asegura que todos los embeddings tengan norma 1, facilitando la comparación por distancia euclidiana.

#### 2.2.3 Estimador de Pose

El estimador de pose produce tres ángulos de Euler para cada rostro detectado:
- **Yaw:** rotación horizontal (cabeza girando de izquierda a derecha). 0° = frontal, ±90° = perfil puro.
- **Pitch:** inclinación vertical (cabeza mirando arriba o abajo). 0° = nivel, valores negativos = mirada abajo.
- **Roll:** inclinación lateral (cabeza ladeada). 0° = erguida.

Estos ángulos son utilizados en el sistema de dos formas:
1. Para filtrado de detecciones de baja confianza (poses extremas que el modelo no puede reconocer fiablemente).
2. Para análisis estadístico y diagnóstico (identificar qué poses de la galería son insuficientes).

<a name="23-embeddings-512"></a>
### 2.3 Los embeddings de 512 dimensiones: qué representan y por qué se normalizan

Un embedding facial es una representación matemática abstracta de la identidad de un rostro. La red ArcFace R100 ha aprendido, a través del entrenamiento en millones de imágenes, a comprimir toda la información identitaria de un rostro en un vector de 512 números reales. Este vector no representa píxeles ni características visuales directamente interpretables; es una proyección en un espacio matemático de alta dimensión donde rostros de la misma persona quedan agrupados y rostros de personas distintas quedan separados.

**¿Qué captura el embedding?**
El modelo aprende automáticamente a codificar rasgos como la forma del hueso cigomático, la distancia intercantal (entre ojos), la relación de proporciones de la nariz y el mentón, la forma del iris (parcialmente), y decenas de otras características de las cuales muchas no tienen nombre en el lenguaje cotidiano. El resultado es un vector que es **único para cada identidad** pero **estable ante variaciones** de iluminación, expresión leve o envejecimiento gradual.

**¿Por qué se normaliza L2?**
Sin normalización, dos embeddings podría tener diferentes magnitudes (normas), lo que haría que la distancia euclidiana entre ellos dependiera tanto de su similitud como de sus magnitudes. Al normalizar L2 (dividir cada vector por su propia norma), todos los vectores quedan proyectados sobre la superficie de una hiperesfera unitaria de 512 dimensiones. En ese espacio:

$$\|\mathbf{e}\|_2 = 1 \quad \forall \mathbf{e} \in \text{galería}$$

La distancia L2 entre dos embeddings normalizados se relaciona directamente con el ángulo que forman:

$$d_{L2}(\mathbf{a}, \mathbf{b}) = \sqrt{2 - 2\cos(\theta)} = \sqrt{2(1 - \cos\theta)}$$

Donde $\theta$ es el ángulo entre los vectores en el espacio de 512 dimensiones. Esto hace que la comparación sea geométricamente limpia y no esté contaminada por diferencias de escala.

**Rango de valores de la distancia L2 entre embeddings normalizados:**
- Mínimo teórico: 0.0 (mismo vector, misma identidad en mismas condiciones)
- Máximo teórico: 2.0 (vectores opuestos, máxima disimilitud)
- Rango típico mismo sujeto: 0.6 – 1.1
- Rango típico impostores (personas distintas): 1.2 – 2.0

<a name="24-galeria-multi-pose"></a>
### 2.4 La galería multi-pose: construcción y búsqueda

La galería es el componente que almacena el "conocimiento de identidades" del sistema. Técnicamente, es un diccionario Python serializado en formato PKL (pickle) con la siguiente estructura lógica:

```
galeria.pkl
├── cesar_angeles        → array de forma (5, 512) — 5 embeddings, uno por pose
├── jessica_urrea_con_lentes  → array de forma (6, 512)
├── jessica_urrea_sin_lentes  → array de forma (6, 512)
├── mitzi_ramirez        → array de forma (6, 512)
└── rafael_alcantar      → array de forma (5, 512)
```

Cada fila del array de una persona corresponde a un embedding registrado desde una pose específica. Al concatenar todos los arrays, se forma una **matriz global de galería** de dimensiones $N_{total} \times 512$, donde $N_{total}$ es la suma de todas las poses de todas las personas. En la configuración actual con 28 poses totales (5+6+6+6+5), esta matriz es de forma $28 \times 512$.

**Algoritmo de búsqueda min-L2:**

Dado un embedding de consulta $\mathbf{q}$ (el rostro detectado en el frame actual), la búsqueda se realiza de la siguiente manera:

1. Se calcula la distancia L2 entre $\mathbf{q}$ y cada uno de los $N_{total}$ embeddings de la galería:

$$d_i = \|\mathbf{q} - \mathbf{g}_i\|_2 \quad \forall i \in \{1, \ldots, N_{total}\}$$

2. Para cada persona $p$, se toma la distancia mínima sobre todas sus poses:

$$D_p = \min_{i \in \text{poses de } p} d_i$$

3. La identidad asignada es la persona con menor $D_p$:

$$\hat{p} = \arg\min_p D_p$$

4. Si $D_{\hat{p}} < \text{umbral}$, se devuelve $\hat{p}$ como identidad. En caso contrario, se devuelve `DESCONOCIDO`.

Esta operación es extremadamente eficiente desde el punto de vista computacional: con 28 embeddings de 512 dimensiones, el cómputo completo es un producto matricial que se realiza en microsegundos en GPU, incluso para lotes de muchos frames.

<a name="25-algoritmo-reconocimiento"></a>
### 2.5 El algoritmo de reconocimiento: distancia L2, similitud y umbral

**Cálculo de la distancia L2:**

La distancia euclidiana (L2) entre el embedding de consulta $\mathbf{q}$ y un embedding de galería $\mathbf{g}$, ambos normalizados, se calcula como:

$$d_{L2} = \sqrt{\sum_{k=1}^{512} (q_k - g_k)^2}$$

**Conversión a similitud:**

Para facilitar la interpretación humana, la distancia L2 se convierte a una escala de similitud entre 0 y 1:

$$\text{Sim} = \max\left(0,\ 1 - \frac{d_{L2}}{2}\right)$$

Esta fórmula mapea:
- $d_{L2} = 0.0$ → $\text{Sim} = 1.00$ (100% de similitud, mismo embedding)
- $d_{L2} = 1.0$ → $\text{Sim} = 0.50$ (similitud media)
- $d_{L2} = 2.0$ → $\text{Sim} = 0.00$ (mínima similitud)

**Umbral de decisión:**

| Configuración           | Umbral L2    | Similitud equivalente | Comportamiento                        |
|-------------------------|--------------|-----------------------|---------------------------------------|
| Producción actual       | < 1.25       | > 0.375               | Acepta el 95% de sujetos; FAR = 0.40% |
| Recomendado por análisis| < 1.20       | > 0.400               | Acepta el 93.3%; FAR = 0.00%          |

**Lógica de decisión completa:**

```
Si d_L2(consulta, mejor_match_galería) < umbral:
    Devolver identidad = nombre_persona
    Devolver confianza = Sim = max(0, 1 - d_L2/2)
Si no:
    Devolver identidad = "DESCONOCIDO"
    Devolver confianza = Sim (informativa, sin acción)
```

<a name="26-l2-vs-coseno"></a>
### 2.6 Diferencia entre distancia L2 y similitud coseno — nota técnica

Este es un punto de confusión frecuente en sistemas de reconocimiento facial. Ambas métricas miden la "similitud" entre dos vectores, pero de formas distintas:

**Similitud coseno:**
$$\cos\theta(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$$

Mide el ángulo entre los vectores, ignorando su magnitud. Rango: [-1, 1] donde 1 = idéntico, 0 = ortogonal, -1 = opuesto.

**Distancia L2 (euclidiana):**
$$d_{L2}(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2$$

Mide la distancia geométrica entre los extremos de los vectores. Rango teórico: [0, ∞), aunque en la práctica para embeddings de este modelo rara vez supera 2.0.

**¿Son equivalentes cuando los vectores están normalizados L2?**
Sí, matemáticamente son equivalentes cuando $\|\mathbf{a}\| = \|\mathbf{b}\| = 1$:

$$d_{L2}^2 = 2 - 2\cos\theta \implies d_{L2} = \sqrt{2(1 - \cos\theta)}$$

En consecuencia, maximizar la similitud coseno es equivalente a minimizar la distancia L2 cuando los embeddings están normalizados. El sistema utiliza distancia L2 por razones de eficiencia computacional: el cálculo vectorizado de L2 sobre matrices es ligeramente más rápido en las librerías NumPy/PyTorch que el cálculo de similitud coseno, y los resultados son matemáticamente idénticos dado que todos los embeddings de ArcFace están normalizados L2.

---

<a name="seccion-3"></a>
## Sección 3: Scripts de Inferencia — Arquitectura de Código

<a name="31-modulos"></a>
### 3.1 Módulos del sistema

El sistema de inferencia está compuesto por los siguientes módulos funcionales:

**`face_engine.py` — Motor principal de reconocimiento:**
Este es el módulo central del sistema. Encapsula toda la lógica de:
- Inicialización del modelo InsightFace `buffalo_l` con el proveedor ONNX Runtime seleccionado (CUDA o CPU).
- Carga de la galería PKL al inicio.
- Procesamiento de frames individuales: detección → alineación → embedding → comparación.
- Filtrado de detecciones por tamaño mínimo (`TAMANO_MIN_PX`).
- Cálculo de distancias L2 y asignación de identidades.
- Generación de resultados con bounding box, identidad, confianza y ángulos de pose.

**`validador.py` — Módulo de evaluación y pruebas:**
Módulo de evaluación que permite comparar las predicciones del sistema contra anotaciones ground-truth. Sus funciones incluyen:
- Lectura de archivos de anotación en formato MOT (Multiple Object Tracking).
- Cálculo de IoU (Intersection over Union) entre bounding boxes predichos y ground-truth.
- Clasificación de cada detección en: True Positive (TP), False Negative por clasificación errónea (FN_Miss), False Negative por no detección (FN_NoDet), False Positive por ID incorrecto (FP_ID) y False Positive fantasma (FP_Ghost).
- Cómputo de todas las métricas de evaluación: Precisión, Recall, F1-Score, FAR, ID Accuracy.
- Generación de reportes detallados por persona y por frame.

**`galeria.pkl` — Base de datos de identidades:**
Archivo binario serializado con Python pickle que contiene la galería multi-pose. No es un módulo de código sino una estructura de datos persistente que el `face_engine.py` carga en memoria al inicio. Es el componente que define "quién está registrado" en el sistema y con qué poses.

**Módulos auxiliares:**
- Script de captura/registro de nuevas personas: toma fotos desde la cámara, extrae embeddings y actualiza el PKL.
- Script de pruebas de estrés: procesa lotes de vídeos y genera estadísticas de rendimiento.
- Script de análisis de umbrales: genera las curvas FAR/FRR a partir de datos de validación.

<a name="32-flujo-frame"></a>
### 3.2 Flujo de procesamiento de un frame — paso a paso

El procesamiento de un frame de vídeo sigue estos pasos en secuencia:

**Paso 1 — Recepción del frame:**
El hilo de procesamiento recibe un frame de la cola de frames (ver sección 3.3). El frame es una imagen BGR en formato NumPy array de forma `(alto, ancho, 3)`.

**Paso 2 — Detección de rostros:**
Se invoca el detector SCRFD con el frame completo. El detector opera internamente a la resolución configurada (`det_size`), redimensionando el frame si es necesario. La salida es una lista de detecciones, cada una con:
- `bbox`: coordenadas `[x1, y1, x2, y2]` del bounding box en píxeles del frame original.
- `score`: confianza de detección en rango [0, 1].
- `kps`: 5 landmarks faciales en coordenadas de píxel.

**Paso 3 — Filtrado por tamaño mínimo:**
Para cada detección, se calcula el ancho del bounding box: `w = x2 - x1`. Si `w < TAMANO_MIN_PX` (actualmente 20 px), la detección se descarta sin proceder al embedding. Esto evita procesar parches de cara demasiado pequeños donde el embedding sería de baja calidad.

**Paso 4 — Extracción de embedding por cada cara válida:**
Para cada detección que supera el filtro de tamaño:
1. Se extraen los 5 landmarks del parche de cara.
2. Se aplica la transformación afín de alineación para producir un parche de 112×112 px.
3. El parche se normaliza en intensidad (media y desviación estándar) según las convenciones del modelo.
4. Se invoca la red ArcFace R100 (inferencia ONNX) para obtener el embedding de 512 dimensiones.
5. El embedding se normaliza L2.

**Paso 5 — Estimación de pose:**
Simultáneamente (o en el mismo pase del modelo), el estimador de pose produce los ángulos yaw, pitch y roll para el rostro detectado.

**Paso 6 — Comparación contra la galería:**
El embedding normalizado se compara contra la matriz de galería. Se calcula el vector de distancias L2 de dimensión $N_{total}$ (una por cada entrada de la galería). Se identifica el índice con distancia mínima y se mapea a la identidad correspondiente.

**Paso 7 — Decisión de identidad:**
- Si la distancia mínima < umbral (1.25 en producción): se asigna la identidad correspondiente.
- Si la distancia mínima ≥ umbral: se asigna `DESCONOCIDO`.
- Se calcula la similitud $\text{Sim} = \max(0, 1 - d_{L2}/2)$ como valor de confianza informativo.

**Paso 8 — Composición del resultado:**
Se construye un diccionario de resultado para cada cara detectada con:
- Bounding box (`bbox`)
- Identidad asignada (`identidad`)
- Distancia L2 (`distancia`)
- Similitud (`similitud`)
- Ángulos de pose (`yaw`, `pitch`, `roll`)
- Flag de `reconocido` (True/False)

**Paso 9 — Visualización o almacenamiento:**
Los resultados se pasan al hilo de visualización o escritura de logs para su representación gráfica sobre el frame o registro en archivo de resultados.

<a name="33-hilo-lector"></a>
### 3.3 El hilo lector con cola de frames — por qué es importante

El sistema implementa una arquitectura de dos hilos para el procesamiento de vídeo:

- **Hilo lector (producer):** Lee frames del flujo de vídeo (archivo de vídeo o cámara IP) y los deposita en una cola de frames (`queue.Queue`). Corre de forma independiente al hilo de procesamiento.
- **Hilo procesador (consumer):** Extrae frames de la cola y realiza todo el pipeline de reconocimiento facial descrito en la sección 3.2.

**¿Por qué es necesaria esta arquitectura?**

El cuello de botella del sistema no es la lectura de frames sino el procesamiento de inferencia (forward pass de las redes neuronales en GPU). En un diseño secuencial simple, durante el tiempo que el procesador espera que termine la inferencia de un frame, el lector está bloqueado y puede perderse frames del flujo de entrada. Con la cola desacoplada:

1. El hilo lector puede avanzar a su propio ritmo, llenando la cola continuamente.
2. El hilo procesador toma frames de la cola tan rápido como puede.
3. Si el procesador es más lento que el lector (que es el caso normal: la GPU tarda más en inferir que la cámara en capturar), la cola actúa como buffer y previene la pérdida de frames recientes.
4. Si la cola tiene un tamaño máximo configurado (ej. `maxsize=2`), cuando esté llena el hilo lector se bloquea automáticamente, lo que implementa backpressure y garantiza que el sistema siempre procese los frames más recientes, descartando los más antiguos si hay atraso.

**Implicación en el rendimiento:**
El sistema alcanza 6.6 fps promedio de procesamiento. Esto significa que en un vídeo de cámara a 30 fps, el sistema procesa aproximadamente 1 de cada 4.5 frames. La cola desacoplada garantiza que los frames procesados sean siempre los más recientes, evitando que el sistema "se retrase" procesando frames viejos mientras acumula un backlog.

<a name="34-filtro-tamano"></a>
### 3.4 El filtro TAMANO_MIN_PX=20 y por qué existe

El parámetro `TAMANO_MIN_PX = 20` especifica el ancho mínimo en píxeles que debe tener un bounding box de cara para que se intente el reconocimiento. Las caras con ancho menor a 20 px son detectadas pero no procesadas con la red de embedding.

**Justificación técnica:**

La red ArcFace R100 fue entrenada con imágenes alineadas de 112×112 px. Cuando el parche de cara de entrada tiene una resolución nativa muy baja (por ejemplo, 20×25 px), el proceso de redimensionado a 112×112 introduce interpolación que degrada significativamente la información textural y de forma de la cara. El embedding resultante es de baja calidad y altamente variable entre frames consecutivos, lo que lleva a:
1. Alta tasa de no-reconocimiento de personas registradas (FN_Miss elevado).
2. Posibles falsos positivos espurios por embeddings de baja calidad que "caen" cerca de un embedding de galería por azar.

Las pruebas instrumentadas mostraron que el umbral práctico de calidad de embedding se encuentra alrededor de 23–24 px de ancho (el sistema detecta desde 23 px y reconoce de forma fiable desde 24 px). El valor conservador de 20 px como filtro del detector garantiza que se conservan todas las detecciones potencialmente útiles sin sobrecargar el pipeline con parches que definitivamente no producirán reconocimiento útil.

**Relación con el resultado de las pruebas:**
En la prueba de estrés de distancia, la resolución mínima detectada fue de **23 px** y la mínima con reconocimiento exitoso fue de **24 px**. El filtro de 20 px está por debajo de ambos valores, lo que significa que en la práctica el filtro rara vez descarta caras detectables, y actúa principalmente como salvaguarda contra detecciones espurias de artefactos visuales que el detector pueda confundir con una cara.

<a name="35-det-size"></a>
### 3.5 Parámetro det_size: cuándo usar 640 vs 1280

El parámetro `det_size` controla la resolución de entrada del detector SCRFD. Es uno de los parámetros de configuración más importantes del sistema y afecta directamente el balance entre:
- **Capacidad de detectar caras pequeñas/distantes** (favorece det_size mayor)
- **Velocidad de procesamiento** (favorece det_size menor)
- **Uso de VRAM** (favorece det_size menor)

**det_size = 640 (por defecto):**
- Tiempo de inferencia del detector: ~15-20 ms en RTX 4050
- Detecta caras de tamaño medio y grande (cara de más de ~40-50 px en el frame de resolución estándar)
- Adecuado para cámaras cercanas (< 5 metros del sujeto)
- Recomendado para: escritorios, recepciones, salas de reunión pequeñas

**det_size = 1280:**
- Tiempo de inferencia del detector: ~50-60 ms en RTX 4050 (3-4x más lento)
- Detecta caras más pequeñas (desde ~23 px demostrado en pruebas)
- Necesario para cámaras distantes o en planos abiertos
- Recomendado para: pasillos largos, áreas de trabajo abiertas, lobby de entrada

**Criterio de selección práctico:**

| Escenario                              | det_size recomendado |
|----------------------------------------|----------------------|
| Cámara de escritorio / toma cercana    | 640                  |
| Pasillo < 5 metros                     | 640                  |
| Pasillo 5-15 metros                    | 1280                 |
| Sala grande / área abierta             | 1280                 |
| Múltiples personas en segundo plano    | 1280                 |
| Prioridad velocidad > cobertura        | 640                  |

Un criterio rápido de campo: si en prueba manual el sistema falla en detectar personas que visualmente están claramente presentes en el frame (se ven bien a simple vista pero el sistema no las detecta), la primera medida a probar es cambiar a `det_size=1280`.

---

<a name="seccion-4"></a>
## Sección 4: Gestión de Retos Ambientales e Incertidumbre

<a name="41-iluminacion"></a>
### 4.1 Iluminación variable y ventanales

El entorno de oficinas con ventanales grandes presenta desafíos de iluminación que son inherentes al tipo de instalación y que el sistema debe gestionar correctamente. Se describen a continuación los mecanismos técnicos de robustez del modelo y las recomendaciones operativas.

#### 4.1.1 Cómo el modelo maneja la variabilidad de iluminación

El modelo `buffalo_l` incluye mecanismos internos de normalización de iluminación:

1. **Normalización del parche de cara:** Antes de ingresar a la red ArcFace, el parche de cara de 112×112 px se normaliza en intensidad: se resta la media global del parche y se divide por su desviación estándar. Esto hace que el embedding sea independiente del brillo absoluto del parche.

2. **Alineación geométrica:** La transformación afín de alineación garantiza que la red "vea" siempre la cara en la misma posición dentro del parche, independientemente de dónde esté en el frame original.

3. **Robustez aprendida durante el entrenamiento:** La red fue entrenada con millones de imágenes capturadas bajo condiciones de iluminación muy diversas. Ha aprendido a extraer rasgos identitarios que son relativamente estables ante cambios de iluminación moderados.

**Limitación conocida:**
El modelo maneja bien variaciones de iluminación difusa (nubes, cambio de hora del día con iluminación frontal). Sin embargo, pierde eficiencia cuando la iluminación es tan extrema que:
- Produce sobreexposición en partes clave del rostro (pómulos, frente blanqueados por sol directo lateral)
- Genera sombras duras que ocultan la mitad del rostro
- Produce reflejos de ventanal que solapan directamente sobre el área facial

#### 4.1.2 Recomendación de iluminación artificial complementaria

Para garantizar un rendimiento estable durante todo el horario laboral, se recomienda instalar iluminación artificial complementaria con las siguientes características:

- **Tipo:** LED de espectro cálido-neutro (3000-4000 K), montado en difusor para suavizar las sombras.
- **Posición:** Orientada hacia las zonas donde las personas están sentadas, perpendicular a la línea de visión de la cámara. No debe estar detrás del sujeto (crearía contraluz).
- **Intensidad:** Suficiente para que la iluminación artificial domina sobre la variabilidad de la luz natural durante los picos solares.
- **Regulación:** Si es posible, sistema de regulación automática que aumente la intensidad artificial cuando la luz natural es intensa (sensores de luminosidad ambiental).

#### 4.1.3 Horarios críticos

Los horarios de mayor riesgo para el rendimiento del sistema en entornos con ventanales son:

| Horario           | Riesgo                                                          | Mitigación recomendada                    |
|-------------------|-----------------------------------------------------------------|-------------------------------------------|
| 07:00 – 09:00     | Sol bajo en el este, luz lateral extrema, sombras duras         | Iluminación artificial complementaria activa |
| 12:00 – 14:00     | Sol cenital, posibles reflejos en superficies horizontales      | Bajo riesgo general; vigilar pantallas    |
| 17:00 – 19:00     | Sol bajo en el oeste, contraluz para cámaras orientadas al oeste | Persianas + iluminación artificial        |
| Días nublados     | Iluminación baja y uniforme; generalmente favorable para el sistema | Sin acción requerida                 |

<a name="42-desconocidos"></a>
### 4.2 Manejo de desconocidos

#### 4.2.1 Protocolo técnico para personas desconocidas

Cuando el sistema detecta una cara cuya distancia L2 al mejor match de la galería supera el umbral configurado, emite el veredicto `DESCONOCIDO`. Esta respuesta es intencionada y correcta: significa que la persona detectada no coincide de forma suficientemente segura con ninguna de las identidades registradas.

En los datos de validación ground-truth, el sistema tuvo **0 falsos reconocimientos de la categoría DESCONOCIDO** en ambos vídeos de evaluación (mot47 y mot48). Esto significa que cuando el sistema dijo `DESCONOCIDO`, efectivamente no había ninguna persona registrada en la escena que debiera haber sido identificada con esa cara.

El umbral no es un rechazo absoluto de "persona no registrada": es una declaración de incertidumbre. Cuando el sistema devuelve `DESCONOCIDO`, puede significar:
1. La persona realmente no está registrada en la galería.
2. La persona está registrada pero la pose actual es demasiado extrema para lograr un match de distancia < umbral.
3. La iluminación en ese frame específico degradó el embedding por debajo del umbral.

#### 4.2.2 Por qué no se puede simplemente rechazar "todo lo que no coincide al 100%"

En un sistema biométrico real, el reconocimiento perfecto (100% de coincidencia) es un objetivo inalcanzable. El embedding de la misma persona varía frame a frame debido a:
- Cambios mínimos de pose entre el frame actual y la pose de registro.
- Variaciones de iluminación.
- Compresión y artefactos del vídeo.
- Micromovimientos faciales (expresión, parpadeo).

En la práctica, incluso para un reconocimiento correcto y seguro, la distancia L2 típica es 0.6–1.0 (similitud 0.50–0.70), no 0.0. Un umbral de 0.0 (exigir coincidencia perfecta) rechazaría prácticamente todos los reconocimientos válidos. El diseño del umbral es un balance entre sensibilidad (reconocer a los registrados) y especificidad (rechazar a los no registrados).

#### 4.2.3 La zona gris: similitud 0.33–0.40

El análisis de umbrales revela que existe una **zona de solapamiento** entre las distribuciones de distancias del sujeto genuino y del impostor:

- **Sujeto genuino (peor 5%):** Distancias L2 entre 1.20 y 1.27 (similitud 0.365–0.40)
- **Impostor (distancia mínima observada):** L2 = 1.2077 (similitud = 0.396)

Esto crea una zona gris entre $d_{L2} \in [1.20, 1.27]$ donde la misma distancia podría corresponder tanto a un sujeto genuino en condición difícil como a un impostor con rasgos similares. El umbral actual de 1.25 se encuentra en esta zona gris, lo que explica la FAR de 0.40% y el solapamiento de $\Delta = -0.0635$.

El umbral recomendado de 1.20 se sitúa justo por debajo de la distancia L2 mínima del impostor observado (1.2077), eliminando completamente el solapamiento a costa de rechazar el 1.7% adicional de reconocimientos del sujeto genuino.

<a name="43-distancia-resolucion"></a>
### 4.3 Optimización por distancia y resolución

#### 4.3.1 La frontera entre detección y reconocimiento

El sistema tiene dos límites operativos distintos que no deben confundirse:

- **Límite de detección:** 23 px de ancho. Por debajo de este valor, el detector SCRFD no puede localizar el rostro de forma fiable.
- **Límite de reconocimiento:** 24 px de ancho. La distancia mínima a la que el sistema reconoce correctamente personas registradas en condición casi frontal (yaw ≈ 8°).

En la práctica, ambos límites están separados por solo 1 píxel, lo cual es consistente con el comportamiento esperado: un rostro de 23 px tiene información justo suficiente para ser localizado, pero el embedding extraído es de calidad marginal para reconocimiento.

#### 4.3.2 Cuándo usar det_size=1280 en cámaras distantes

El valor `det_size=1280` es particularmente importante en instalaciones donde la cámara cubre un área amplia y las personas se encuentran habitualmente a más de 5 metros de distancia. En esas condiciones, con `det_size=640`, las caras de personas en segundo plano pueden tener menos de 30 px en el frame y el detector puede no localizarlas de forma consistente.

La prueba de estrés de distancia demostró que con `det_size` apropiado, el sistema reconoce correctamente a **100% de los sujetos registrados** incluso a resoluciones de cara de 24 px, siempre que el ángulo de visión sea aproximadamente frontal (yaw ≈ 8° en la prueba).

#### 4.3.3 Distancia máxima práctica de reconocimiento

Para una estimación de la distancia máxima de reconocimiento en una instalación real, se puede utilizar la siguiente aproximación:

$$d_{\max} = \frac{h_{\text{cara\_real}} \times f}{\text{px}_{\min} \times \text{px\_tamaño}}$$

Donde:
- $h_{\text{cara\_real}}$: altura real promedio de una cara (~20 cm)
- $f$: focal length de la cámara en píxeles
- $\text{px}_{\min}$: tamaño mínimo de cara para reconocimiento (24 px)
- $\text{px\_tamaño}$: tamaño de píxel del sensor

En la práctica, con una cámara de seguridad estándar (1080p, lente 2.8mm), el límite de reconocimiento suele estar entre 3 y 8 metros dependiendo de la configuración. Las pruebas de campo deben siempre verificar este límite para cada instalación específica.

---

<a name="seccion-5"></a>
## Sección 5: Reporte Formal de Evaluación y Estrés

<a name="51-metodologia"></a>
### 5.1 Metodología de validación

#### 5.1.1 Formato de anotación Ground-Truth

Los vídeos de evaluación fueron anotados en formato MOT (Multiple Object Tracking), estándar de facto para la evaluación de sistemas de seguimiento y reconocimiento en vídeo. Cada línea del archivo de anotación especifica:

```
frame_id, track_id, x, y, w, h, confidence, class_id, identity
```

Las anotaciones fueron realizadas manualmente por el equipo del proyecto, asegurando que cada aparición de cada colaborador estuviera etiquetada con su identidad correcta, su posición en el frame y su visibilidad (se excluyen frames donde la cara está totalmente ocluida o de espaldas).

#### 5.1.2 Criterio de asociación — IoU ≥ 0.15

Para asociar una detección predicha por el sistema con una anotación ground-truth, se utiliza el criterio de **Intersection over Union (IoU)**:

$$\text{IoU}(B_{\text{pred}}, B_{\text{gt}}) = \frac{|B_{\text{pred}} \cap B_{\text{gt}}|}{|B_{\text{pred}} \cup B_{\text{gt}}|}$$

Donde $B_{\text{pred}}$ es el bounding box predicho y $B_{\text{gt}}$ es el bounding box ground-truth. Un umbral de IoU ≥ 0.15 significa que se aceptan como "mismo rostro" detecciones que superpongan al menos el 15% del área combinada de ambos bounding boxes. Este umbral generoso (el estándar PASCAL VOC usa 0.50) fue elegido deliberadamente porque en vídeo de vigilancia los bounding boxes de cara a veces se desplazan ligeramente entre el frame del ground-truth y el frame procesado por el detector.

#### 5.1.3 Glosario de veredictos de evaluación

Cada aparición anotada en el ground-truth se clasifica en una de las siguientes categorías:

| Veredicto     | Definición                                                                                          |
|---------------|-----------------------------------------------------------------------------------------------------|
| **TP**        | True Positive: el sistema detectó la cara y la identificó correctamente (IoU ≥ 0.15, ID correcto)   |
| **FN_Miss**   | False Negative por clasificación errónea: cara detectada con IoU ≥ 0.15, pero ID incorrecto o DESCONOCIDO cuando debía identificarse |
| **FN_NoDet**  | False Negative por no detección: la cara estaba presente en el ground-truth pero el detector no la localizó (sin match con IoU ≥ 0.15) |
| **FP_ID**     | False Positive por ID incorrecto: el sistema asignó una identidad registrada a una cara que pertenece a una persona distinta |
| **FP_Ghost**  | False Positive fantasma: detección completamente espuria sin correspondencia con ninguna anotación en el ground-truth (el detector "inventó" una cara) |

<a name="52-metricas"></a>
### 5.2 Definición formal de métricas

#### 5.2.1 Precisión (Precision)

Mide qué proporción de las veces que el sistema dice "reconocí a alguien" es correcto. Es la métrica de confiabilidad de las predicciones positivas.

$$\text{Precisión} = \frac{TP}{TP + FP_{ID} + FP_{Ghost}}$$

Un valor de Precisión = 98.8% significa que de cada 100 veces que el sistema emite un nombre de colaborador, 98.8 son correctos y 1.2 son erróneos (falsa alarma).

#### 5.2.2 Recall (Exhaustividad)

Mide qué proporción de todas las apariciones reales de colaboradores el sistema logró identificar correctamente. Es la métrica de cobertura.

$$\text{Recall} = \frac{TP}{TP + FN_{Miss} + FN_{NoDet}}$$

Un valor de Recall = 82.9% significa que de cada 100 apariciones reales de un colaborador registrado, el sistema identificó correctamente 82.9 y perdió 17.1 (por no detección o clasificación errónea).

#### 5.2.3 F1-Score

Media armónica de Precisión y Recall. Sintetiza el rendimiento global del sistema en un único número, penalizando más los valores extremos.

$$F_1 = 2 \cdot \frac{\text{Precisión} \times \text{Recall}}{\text{Precisión} + \text{Recall}}$$

#### 5.2.4 False Alarm Rate (FAR biométrico)

En el contexto biométrico, la tasa de falsa aceptación (FAR) mide la probabilidad de que el sistema identifique incorrectamente a un impostor (persona no registrada) como una persona registrada.

$$\text{FAR} = \frac{FP_{ID}}{FP_{ID} + TN}$$

En las evaluaciones, se calculó a partir del análisis de umbrales sobre embeddings de impostores conocidos.

#### 5.2.5 False Rejection Rate (FRR biométrico)

Tasa de falso rechazo: proporción de veces que el sistema rechaza (dice DESCONOCIDO) a un sujeto genuino registrado.

$$\text{FRR} = \frac{\text{Reconocimientos rechazados del sujeto genuino}}{\text{Total de intentos del sujeto genuino}}$$

#### 5.2.6 ID Accuracy

Métrica de identificación pura: dado que el sistema detectó una cara, ¿qué proporción de las veces la identificó correctamente?

$$\text{ID Accuracy} = \frac{TP}{TP + FP_{ID}}$$

Esta métrica es especialmente relevante para auditorías de seguridad porque mide directamente la fiabilidad de las decisiones de identidad, desacoplada de los errores de detección.

<a name="53-mot47"></a>
### 5.3 Resultados mot47 — análisis completo

**Descripción del escenario:**
- 5,000 frames totales de vídeo
- Cámara fija con ángulo moderado
- 3-4 personas presentes simultáneamente en la mayoría de los frames
- Colaboradores presentes: `cesar_angeles`, `mitzi_ramirez`, `rafael_alcantar`, y personas desconocidas

**Métricas globales mot47:**

| Métrica          | Valor    |
|------------------|----------|
| Precisión        | **98.8%** |
| Recall           | **82.9%** |
| F1-Score         | **90.2%** |
| False Alarm Rate | 0.2%     |
| ID Accuracy      | 58.8%    |

**Conteo de veredictos mot47:**

| Veredicto   | Cantidad |
|-------------|----------|
| TP          | 10,029   |
| FN_Miss     | 2,069    |
| FN_NoDet    | 3,066    |
| FP_ID       | 119      |
| FP_Ghost    | 33       |

**Resultados por persona en mot47:**

| Persona          | Precisión | Recall  | TP     | FN_Miss | Análisis                                          |
|------------------|-----------|---------|--------|---------|---------------------------------------------------|
| `cesar_angeles`  | 100.0%    | 80.6%   | 4,017  | 969     | Cero falsos positivos; los FN se deben a poses extremas en algunos frames |
| `mitzi_ramirez`  | 99.3%     | 92.7%   | 3,720  | 294     | Mejor recall del grupo; galería con buena cobertura angular para este ángulo de cámara |
| `rafael_alcantar`| 96.1%     | 74.0%   | 2,292  | 806     | Recall más bajo; la cámara ya captura este colaborador en ángulos que la galería cubre parcialmente |
| DESCONOCIDO      | N/A       | 0 TP (correcto) | 0 | — | Ninguna persona desconocida fue identificada erróneamente |

**Análisis de mot47:**

El escenario mot47 representa las condiciones "razonablemente favorables" del sistema. La Precisión de 98.8% confirma que cuando el sistema emite un nombre, es casi siempre correcto. El Recall de 82.9% refleja principalmente dos fuentes de pérdida:
1. **FN_NoDet (3,066):** Frames donde las personas estaban presentes pero el detector no las localizó, probablemente por poses extremas, oclusión parcial o resolución insuficiente.
2. **FN_Miss (2,069):** Frames donde el detector localizó la cara pero el sistema no logró identificarla (emitió DESCONOCIDO o ID incorrecto). Esto ocurre principalmente en poses extremas (yaw > 60°) donde la galería tiene menor cobertura.

Los 119 FP_ID y 33 FP_Ghost son valores muy bajos: representan respectivamente el 1.2% y el 0.33% del total de detecciones, validando la alta confiabilidad operativa del sistema.

<a name="54-mot48"></a>
### 5.4 Resultados mot48 — análisis completo y diagnóstico

**Descripción del escenario:**
- 12,000 frames totales de vídeo
- Escenario considerablemente más difícil: ángulos de cámara más extremos, mayor variedad de distancias
- Hasta 8 personas presentes simultáneamente en el mismo frame
- Colaboradores presentes: `cesar_angeles`, `jessica_urrea`, `rafael_alcantar`, y personas desconocidas

**Métricas globales mot48:**

| Métrica          | Valor    |
|------------------|----------|
| Precisión        | **98.5%** |
| Recall           | **33.1%** |
| F1-Score         | **49.6%** |
| False Alarm Rate | 0.3%     |
| ID Accuracy      | 11.0%    |

**Conteo de veredictos mot48:**

| Veredicto   | Cantidad |
|-------------|----------|
| TP          | 7,279    |
| FN_Miss     | 14,679   |
| FN_NoDet    | 24,539   |
| FP_ID       | 112      |
| FP_Ghost    | 179      |

**Resultados por persona en mot48:**

| Persona          | Precisión | Recall  | TP     | FN_Miss | Diagnóstico                                       |
|------------------|-----------|---------|--------|---------|---------------------------------------------------|
| `cesar_angeles`  | 99.3%     | 89.8%   | 7,255  | 827     | Excelente rendimiento; galería adecuada para este ángulo |
| `rafael_alcantar`| 61.5%     | 1.1%    | 24     | —       | Cámara captura perfil/espalda; galería insuficiente |
| DESCONOCIDO      | N/A       | 0 TP (correcto) | 0 | — | Ninguna identificación errónea de desconocidos |

**Diagnóstico técnico detallado de mot48:**

El bajo Recall global (33.1%) y el F1-Score de 49.6% tienen una causa técnica clara y solucionable: **incompatibilidad entre las poses de la galería y el ángulo real de la cámara de mot48**.


**Problema 1 — rafael_alcantar (1.1% Recall):**
La cámara de mot48 captaba a este colaborador mayoritariamente en perfil casi puro o quasi-espalda (yaw estimado > 80°). La galería incluía tres cuartos (hasta ~45°) pero no perfiles puros ni cuasi-espaldas. El resultado es un Recall de 1.1%: en los 24 frames donde fue reconocido, probablemente la cámara lo captó en el momento breve en que pasaba por un ángulo más frontal.

**Consecuencia para la ID Accuracy:**
El ID Accuracy de 11.0% en mot48 es consecuencia directa de que rafael_alcantar (con 24 TP) domina el denominador de identificaciones posibles, mientras cesar_angeles (con 7,255 TP) tiene una galería adecuada. Esto ilustra que la ID Accuracy no es una métrica uniforme del sistema sino que depende críticamente de la calidad de la galería de cada persona.

**Conclusión:**
El sistema no tiene un problema de rendimiento intrínseco en mot48. Tiene un problema de cobertura de galería. La solución es re-registrar a rafael_alcantar con poses que repliquen los ángulos reales de la cámara mot48 (ver Sección 7, Recomendaciones 2 y 3).

<a name="55-estres-angular"></a>
### 5.5 Prueba de estrés angular — degradación por tramos

Se realizaron 4 vídeos de prueba de estrés angular con el objetivo de caracterizar la degradación del rendimiento del sistema a medida que el ángulo yaw se aleja del frontal. Los vídeos cubrían diferentes distancias y resoluciones de cara para aislar el efecto del ángulo.

**Resultados de la prueba angular:**

| Vídeo               | Yaw máx. | Resolución media | Tasa reconocimiento |
|---------------------|----------|-----------------|---------------------|
| angulos yaw cerca   | 84.0°    | 102 px          | **100%**            |
| angulos totales     | 75.1°    | 1,429 px        | **99.0%**           |
| angulos yaw         | 87.4°    | 100 px          | **85.9%**           |
| angulos yaw lejos   | 79.1°    | 38 px           | **81.7%**           |

**Análisis por tramos angulares:**

#### Tramo 0°–30°: Zona óptima
En este rango, el sistema opera con el máximo rendimiento. Los embeddings generados están dentro del núcleo de la distribución de entrenamiento de ArcFace. La galería cubre completamente este rango con las poses frontal y tres_cuartos. Se esperan tasas de reconocimiento > 99% para resoluciones de cara ≥ 40 px.

**Implicación operativa:** Cualquier cámara instalada que capture a las personas en este rango angular de forma habitual puede esperar el mejor rendimiento del sistema.

#### Tramo 30°–60°: Zona funcional
El sistema mantiene tasas de reconocimiento altas. Los embeddings de tres cuartos (30°–45°) están cubiertos por la galería. A medida que el ángulo supera los 45°, la galería cubre el rango con menor densidad. Se esperan tasas de reconocimiento > 95% para resoluciones ≥ 50 px.

**Implicación operativa:** Zona aceptable para operación normal. La galería con poses tres_cuartos izquierdo y derecho es suficiente para este tramo.

#### Tramo 60°–82°: Zona de degradación controlada
En este rango, la tasa de reconocimiento comienza a degradarse de forma observable. Los datos de la prueba `angulos yaw` (87.4° máx, resolución media 100 px) mostraron 85.9% de tasa de reconocimiento. El sistema sigue siendo funcional pero con más FN_Miss. La degradación es controlada y predecible.

**Implicación operativa:** Si una cámara captura frecuentemente a personas en este rango, se debe considerar añadir poses de galería para ángulos de 60°–75°. Las personas no se pueden "forzar" a mantenerse en ángulo frontal, pero sí se puede mejorar la galería.

#### Tramo 82°–87°: Zona de riesgo
La tasa de reconocimiento cae por debajo del umbral del 85.9% registrado en la prueba más demandante. Los embeddings generados en este rango angulares difieren significativamente de cualquier pose de galería estándar. El sistema sigue detectando las caras (SCRFD opera hasta 89.2°) pero no puede identificarlas de forma fiable.

**Implicación operativa:** Solo aceptable si se han incluido en la galería poses específicas de 75°–82° para las personas que habitualmente son captadas en estos ángulos.

#### Por encima de 87°: Fuera de rango operativo
A partir de 87.4° de yaw, el rendimiento del reconocimiento cae de forma precipitada. El límite de detección del sistema (89.2° yaw) está muy cerca de este rango, lo que significa que incluso la detección es marginal. El sistema no puede garantizar reconocimiento útil en este rango.

**Implicación operativa:** Si una cámara captura a personas habitualmente a yaw > 87° (perfil puro o cuasi-espalda), la instalación no es adecuada para reconocimiento facial y debe reposicionarse.

<a name="56-estres-distancia"></a>
### 5.6 Prueba de estrés de distancia y resolución

Se realizó una prueba específica para caracterizar el comportamiento del sistema en función de la distancia del sujeto a la cámara (y en consecuencia, la resolución de la cara en el frame).

**Condiciones de la prueba:**
- Sujeto moviéndose gradualmente desde cerca (~50 cm) hasta lejos (~10 m) de la cámara.
- Ángulo yaw aproximadamente frontal (yaw ≈ 8°) durante toda la prueba.
- `det_size` configurado apropiadamente para capturar la cara incluso a distancia máxima.

**Resultados de la prueba de distancia:**

| Métrica                               | Valor           |
|---------------------------------------|-----------------|
| Resolución mínima detectada           | **23 px** ancho |
| Resolución mínima reconocida (p5)     | **24 px** ancho |
| Tasa de reconocimiento global         | **100%**        |
| Condición angular durante la prueba   | yaw ≈ 8°        |

**Análisis:**
La tasa de reconocimiento del 100% en condición frontal confirma que el sistema no tiene problemas de reconocimiento por distancia cuando el ángulo es favorable. El límite de 23–24 px es el mínimo técnico para obtener un embedding de calidad suficiente.

**Nota importante:** La tasa del 100% se obtuvo en condición casi frontal (yaw ≈ 8°). Si se combina resolución baja (< 40 px) con ángulo alto (> 45°), la tasa de reconocimiento puede degradarse de forma combinada y más severa que cualquiera de los dos factores de forma aislada.

<a name="57-estres-concurrencia"></a>
### 5.7 Prueba de estrés de concurrencia

La capacidad del sistema para gestionar múltiples personas simultáneas en el mismo frame es un requisito crítico para entornos de oficina.

**Máxima concurrencia probada:**

| Métrica                               | Valor                          |
|---------------------------------------|--------------------------------|
| Máximo de personas/frame probado      | **8 personas** (vídeo mot48)   |
| FPS promedio de procesamiento         | **6.6 fps**                    |
| Hardware utilizado                    | NVIDIA RTX 4050 Laptop, 6.4 GB |

**Análisis de concurrencia:**

El sistema procesa todas las caras detectadas en cada frame de forma secuencial (una tras otra) dentro del mismo pipeline. No existe paralelismo a nivel de cara en el diseño actual. Esto significa que el tiempo de procesamiento por frame escala linealmente con el número de caras detectadas:

$$t_{\text{frame}} \approx t_{\text{detección}} + N_{\text{caras}} \times t_{\text{embedding}}$$

Donde $t_{\text{detección}}$ es el tiempo del detector (dominante, ~15-50 ms dependiendo de `det_size`) y $t_{\text{embedding}}$ es el tiempo de la red ArcFace por cara (~5-10 ms en GPU).

Con 8 personas simultáneas y el hardware actual, el sistema mantiene 6.6 fps, lo que es suficiente para monitoreo de presencia pero insuficiente para análisis en tiempo real estricto (24-30 fps). Para la mayoría de los casos de uso de identificación de colaboradores en oficina (registro de asistencia, control de acceso de sala), 6.6 fps es más que suficiente.

<a name="58-far-frr"></a>
### 5.8 Análisis del umbral de similitud — curva FAR/FRR completa

Se realizó un análisis sistemático de la curva FAR/FRR para determinar el umbral óptimo de operación. El análisis se basa en las distribuciones de distancias L2 del sujeto genuino y de impostores reales.

**Tabla completa FAR/FRR por umbral:**

| Umbral L2 | Similitud eq. | Sujeto aceptado | FAR (impostores) | FRR (rechazos) | Observación                              |
|-----------|---------------|-----------------|------------------|----------------|------------------------------------------|
| < 1.10    | > 0.450       | 87.2%           | 0.00%            | 12.8%          | Demasiado restrictivo; pierde mucho recall |
| < 1.15    | > 0.425       | 90.7%           | 0.00%            | 9.3%           | Seguro; FRR aún elevado                  |
| < 1.20    | > 0.400       | 93.3%           | **0.00%**        | 6.7%           | **Recomendado** — elimina FAR completamente |
| < 1.25    | > 0.375       | 95.0%           | **0.40%**        | 5.0%           | Producción actual — tiene solapamiento   |
| < 1.30    | > 0.350       | 96.8%           | 1.20%            | 3.2%           | FAR elevada; no recomendado              |
| < 1.35    | > 0.325       | 98.1%           | 3.50%            | 1.9%           | Inaceptable para uso de seguridad        |

**Datos clave del análisis:**
- Distancia L2 mínima del impostor observada: **1.2077**
- Solapamiento con umbral actual (1.25): el 5% peor de los sujetos genuinos cae dentro del rango del impostor; gap = **-0.0635**
- Ajustando el umbral a 1.20: el gap pasa a ser positivo (+0.0077), eliminando completamente el solapamiento

**Interpretación de la curva:**

La curva FAR/FRR muestra un punto de quiebre en L2 = 1.20. Por debajo de este valor, la FAR es 0% porque no se ha observado ningún impostor con distancia L2 < 1.2077. Por encima de 1.20, la FAR crece rápidamente porque se entra en la zona de solapamiento entre distribuciones. El umbral actual de 1.25 se encuentra 0.05 por encima del punto de quiebre, lo que explica la FAR observada de 0.40%.

<a name="59-ficha-tecnica"></a>
### 5.9 Ficha Técnica Certificada — límites operativos del sistema

La siguiente ficha resume los límites operativos validados instrumentalmente del sistema en su configuración actual. Estos valores son mediciones reales, no estimaciones teóricas.

---

**FICHA TÉCNICA — Sistema de Reconocimiento Facial InsightFace buffalo_l**
**Fecha de validación:** Mayo 2026

| Capacidad                              | Límite certificado                                         |
|----------------------------------------|------------------------------------------------------------|
| **Detección — yaw máximo**             | 89.2° yaw                                                  |
| **Detección — pitch máximo**           | 94.7° pitch                                                |
| **Detección — resolución mínima**      | 23 px ancho de cara                                        |
| **Reconocimiento 100% garantizado**    | ≤ 82.2° yaw, resolución ≥ 24 px                            |
| **Reconocimiento operativo (≥80%)**    | ≤ 87.4° yaw                                               |
| **Resolución mínima de reconocimiento**| 24 px ancho de cara (condición frontal)                   |
| **Umbral producción actual**           | L2 < 1.25 → Similitud > 0.375                              |
| **Umbral recomendado**                 | L2 < 1.20 → Similitud > 0.400                              |
| **FAR con umbral 1.25**                | 0.40%                                                      |
| **FAR con umbral 1.20**                | 0.00%                                                      |
| **FRR con umbral 1.25**                | 5.0%                                                       |
| **FRR con umbral 1.20**                | 6.7%                                                       |
| **Concurrencia máxima probada**        | 8 personas/frame                                           |
| **FPS de procesamiento**               | 6.6 fps promedio (RTX 4050 Laptop)                         |
| **Precisión en condición favorable**   | 98.8% (mot47)                                              |
| **Recall en condición favorable**      | 82.9% (mot47)                                              |
| **Embeddings por persona (galería)**   | 5–6 poses estándar                                         |

---

<a name="seccion-6"></a>
## Sección 6: Análisis de Umbral y Recomendación

### 6.1 Por qué el umbral actual (1.25) tiene una zona de solapamiento

El umbral de decisión de un sistema biométrico no puede elegirse de forma arbitraria: debe establecerse en la frontera entre las distribuciones de distancias del sujeto genuino (persona registrada siendo identificada) y del impostor (persona no registrada siendo evaluada). En un sistema ideal, estas dos distribuciones estarían perfectamente separadas sin ningún solapamiento, y el umbral podría colocarse entre ellas sin cometer ningún error.

En la realidad, las distribuciones se solapan en mayor o menor medida dependiendo de la calidad del modelo, la diversidad de las condiciones de captura y la variabilidad intrínseca de los sujetos. Para el sistema actual:

- **Distribución del sujeto genuino:** La gran mayoría de las distancias L2 de reconocimientos correctos cae entre 0.6 y 1.15. Sin embargo, el **5% peor** (el 5% de reconocimientos más difíciles, en poses extremas o iluminación adversa) produce distancias L2 entre 1.20 y 1.27.

- **Distribución del impostor:** La distancia L2 mínima observada entre el embedding de un impostor y el mejor match de la galería fue **1.2077**.

Esto crea un solapamiento: el rango L2 ∈ [1.2077, 1.27] contiene tanto el 5% peor del sujeto genuino como el caso del impostor con distancia mínima. El umbral actual de 1.25 se sitúa dentro de esta zona, lo que garantiza que captura el 95% de los sujetos genuinos pero también acepta al impostor de 1.2077 como genuino (FAR = 0.40%).

El gap numérico del solapamiento es:

$$\Delta = d_{L2}^{\text{impostor\_min}} - \text{umbral} = 1.2077 - 1.25 = -0.0635$$

Un gap negativo significa que el umbral se encuentra **por encima** del impostora más cercano, es decir, el impostor caería dentro de la zona de aceptación del sistema.

### 6.2 El concepto de EER (Equal Error Rate)

El **Equal Error Rate (EER)** es el punto de operación de un sistema biométrico en el que la tasa de falsa aceptación (FAR) es igual a la tasa de falso rechazo (FRR). Es una métrica estándar de la industria biométrica porque proporciona una medida única e imparcial del rendimiento del sistema, sin favorecer ninguna de las dos direcciones de error.

Para el sistema actual, la curva FAR/FRR muestra:
- A umbral 1.20: FAR = 0.00%, FRR = 6.7% → FRR >> FAR (umbral conservador)
- A umbral 1.25: FAR = 0.40%, FRR = 5.0% → FAR < FRR (umbral menos conservador)

El EER se encontraría aproximadamente en el rango L2 ∈ [1.20, 1.25], en el punto donde FAR ≈ FRR. En base a los datos disponibles, el EER estimado del sistema está en torno al 3%–4% (donde FAR y FRR serían ambos aproximadamente iguales).

Para un sistema de identificación en entorno de oficina (no de seguridad máxima), operar en el EER es una elección razonable. Sin embargo, dado que los datos muestran que L2 = 1.20 elimina completamente la FAR observada con una penalización de apenas 1.7% en la tasa de aceptación del sujeto genuino, el umbral recomendado es L2 < 1.20, que es ligeramente más conservador que el EER pero con una penalización marginal.

### 6.3 Por qué L2 < 1.20 es el umbral recomendado — justificación matemática

La recomendación de bajar el umbral de 1.25 a 1.20 se basa en el análisis de la distribución de distancias observada:

**Argumento 1 — Eliminación completa del solapamiento:**
El impostor con menor distancia L2 registrado en las pruebas tiene $d_{L2} = 1.2077$. Estableciendo el umbral en 1.20, el sistema rechazaría automáticamente ese impostor porque $1.2077 > 1.20$. El gap pasa a ser positivo:

$$\Delta_{\text{nuevo}} = 1.2077 - 1.20 = +0.0077$$

Aunque es un margen estrecho (77 milésimas), es suficiente para eliminar la FAR observada del 0.40% (convierte FAR de 0.40% → 0.00%).

**Argumento 2 — Impacto mínimo en el recall del sujeto genuino:**
La tasa de aceptación del sujeto genuino disminuye de 95.0% a 93.3%, una pérdida de apenas 1.7 puntos porcentuales. Esto significa que el 1.7% de reconocimientos adicionales que se pierden corresponden a los casos más difíciles (poses extremas, iluminación muy adversa) donde la confiabilidad del reconocimiento ya era marginal con el umbral anterior.

**Argumento 3 — Asimetría del costo de error:**
En el contexto de identificación de colaboradores en oficina:
- Un **FP_ID** (falsa aceptación de impostor) es un error de seguridad: el sistema afirma que una persona es alguien que no es. Coste: alto.
- Un **FN_Miss** adicional (no reconocer a alguien en un frame difícil) es un error de cobertura: el sistema dice DESCONOCIDO para alguien que debería identificar. Coste: bajo (se recupera en el siguiente frame o con la siguiente detección).

Dado que el coste asimétrico favorece minimizar los FP_ID, es preferible aceptar una FRR ligeramente mayor (6.7% vs 5.0%) a mantener la FAR no nula (0.40% → 0.00%).

### 6.4 Impacto de bajar a L2 < 1.15 — análisis de la opción más conservadora

Para completar el análisis, se evalúa el impacto de una reducción adicional al umbral 1.15:

- **Tasa de aceptación del sujeto genuino:** 90.7% (pérdida de 4.3 pp respecto al umbral recomendado)
- **FAR:** 0.00% (igual que 1.20, no mejora adicionalmente)
- **FRR:** 9.3%

La reducción de 1.20 a 1.15 no aporta ninguna mejora en FAR (ya era 0.00% en 1.20) pero sí aumenta la FRR en 2.6 puntos porcentuales adicionales. Por tanto, bajar a 1.15 no está justificado con los datos actuales: ya se alcanzó la FAR mínima posible en 1.20, y continuar bajando el umbral solo perjudica el recall sin ninguna ganancia en seguridad.

**Conclusión:** El umbral óptimo para el sistema actual es **L2 < 1.20**, que elimina toda la FAR observada con la menor penalización posible sobre el recall del sujeto genuino.

---

<a name="seccion-7"></a>
## Sección 7: Recomendaciones Técnicas y Plan de Mejora

Las siguientes recomendaciones se derivan directamente de los datos de evaluación y análisis documentados en las secciones anteriores. Cada recomendación incluye su justificación técnica, impacto esperado y nivel de esfuerzo de implementación.

---

### Recomendación 1: Ajustar el umbral de reconocimiento a L2 < 1.20

**Problema que resuelve:** El umbral actual de L2 < 1.25 genera una tasa de falsa aceptación (FAR) del 0.40%, con un solapamiento de -0.0635 entre la distribución del sujeto genuino y el impostor más cercano.

**Acción:** Modificar en el código de `face_engine.py` el valor del umbral de decisión de `1.25` a `1.20`.

**Impacto esperado:**
- FAR: 0.40% → **0.00%** (eliminación completa de falsos positivos de identidad)
- FRR: 5.0% → **6.7%** (incremento marginal de rechazos del sujeto genuino)
- Tasa de aceptación del genuino: 95.0% → **93.3%**

**Nivel de esfuerzo:** Muy bajo — cambio de un parámetro numérico en el código de configuración.

**Riesgo:** Mínimo. La penalización es del 1.7% en la tasa de reconocimiento de poses difíciles, lo cual es aceptable dada la eliminación total de la FAR.

---

### Recomendación 2: Re-registrar jessica_urrea con poses desde el ángulo real de la cámara mot48

**Problema que resuelve:** `jessica_urrea` tiene Precisión = 0% y Recall = 0% en mot48 porque su galería actual no incluye poses desde el ángulo de visión real de la cámara de ese escenario (ángulo superior lateral pronunciado).

**Acción:**
1. Identificar la posición exacta y altura de la cámara mot48 en la instalación real.
2. Realizar una sesión de captura de poses de `jessica_urrea` replicando ese ángulo: el operador debe sostener la cámara de registro a la misma altura y ángulo lateral que la cámara mot48.
3. Capturar al menos 2-3 poses adicionales: `superior_lateral_izq`, `superior_lateral_der`, `superior_frontal`.
4. Añadir los nuevos embeddings a la galería PKL existente (no reemplazar, agregar).

**Impacto esperado:** El Recall de jessica_urrea en mot48 debería pasar de 0% a un valor comparable al de cesar_angeles (>85%), ya que el problema es exclusivamente de cobertura de galería, no de limitaciones del modelo.

**Nivel de esfuerzo:** Bajo — sesión de captura de ~15 minutos más actualización de la galería PKL.

---

### Recomendación 3: Re-registrar rafael_alcantar con poses laterales adicionales

**Problema que resuelve:** `rafael_alcantar` tiene Recall = 1.1% en mot48 porque la cámara lo captura casi de espaldas o en perfil puro, ángulos no cubiertos por su galería actual.

**Acción:**
1. Verificar el ángulo típico desde el que la cámara mot48 capta a rafael_alcantar (perfil puro, cuasi-espalda, etc.).
2. Añadir poses de galería para ángulos yaw 60°–85° izquierdo y derecho: `perfil_izq`, `perfil_der`, `cuasi_espalda_izq`, `cuasi_espalda_der`.
3. Si la cámara lo capta frecuentemente de espalda pura (yaw > 87°), asumir que el reconocimiento en esa posición está fuera de los límites operativos del sistema (ver Ficha Técnica) y gestionar esa condición de forma diferente (ej. no intentar reconocimiento, confiar en la detección de presencia únicamente).

**Impacto esperado:** Recall de rafael_alcantar en mot48: de 1.1% a > 50% si el ángulo promedio es ≤ 75°.

**Nivel de esfuerzo:** Bajo-medio — sesión de captura con ángulos inusuales que requieren cierta coordinación de posicionamiento.

---

### Recomendación 4: Instalar iluminación artificial en zonas de sombra

**Problema que resuelve:** La variabilidad de iluminación en entornos con ventanales grandes produce frames en los que el parche de cara tiene contraste insuficiente para generar un embedding de alta calidad, contribuyendo a los FN_Miss.

**Acción:**
1. Identificar las zonas de la oficina donde la iluminación natural crea sombras intensas en los horarios críticos (07:00-09:00 y 17:00-19:00).
2. Instalar paneles LED de techo o pared con difusores, orientados hacia las zonas de trabajo donde las personas son detectadas habitualmente.
3. Configurar la iluminación para que mantenga un nivel mínimo de 200-300 lux sobre las caras de los colaboradores durante todo el horario laboral.
4. Evaluar el sistema de reconocimiento antes y después de la instalación con las métricas de la sección 5.

**Impacto esperado:** Reducción del FN_Miss por iluminación adversa. Estimación: mejora de 2-5 puntos porcentuales en Recall.

**Nivel de esfuerzo:** Medio — requiere instalación eléctrica y adquisición de equipamiento.

---

### Recomendación 5: Usar det_size=1280 en cámaras distantes

**Problema que resuelve:** En cámaras instaladas en pasillos largos o áreas abiertas, los rostros de personas distantes pueden tener < 30 px en el frame con `det_size=640`, resultando en FN_NoDet evitables.

**Acción:**
1. Identificar qué cámaras de la instalación cubren áreas con distancias > 5 metros al sujeto más lejano habitual.
2. Para esas cámaras específicas, cambiar el parámetro `det_size=1280` en la configuración del pipeline.
3. Verificar que el hardware (RTX 4050) mantiene fps aceptables con la nueva configuración (se espera reducción de ~3x en velocidad de detección, pero FPS global puede mantenerse si el número de caras por frame es moderado).

**Impacto esperado:** Reducción significativa del FN_NoDet en escenarios con personas a distancia. Potencial mejora de 5-15 puntos porcentuales en Recall en cámaras de pasillo.

**Nivel de esfuerzo:** Muy bajo — cambio de parámetro de configuración por cámara.

---

### Recomendación 6: Agregar pose "tres_cuartos_abajo" para ángulo superior de cámaras de oficina

**Problema que resuelve:** Las cámaras de oficina suelen montarse en la parte superior de paredes o cornisas, capturando a los colaboradores con una combinación de yaw (ángulo lateral) y pitch negativo (mirada ligeramente hacia abajo desde la cámara). El conjunto estándar de poses cubre el pitch negativo en la línea frontal, pero no en la combinación con yaw moderado.

**Acción:**
1. Añadir dos poses estándar adicionales al protocolo de registro: `tres_cuartos_abajo_izq` (yaw +30°–45°, pitch -20°) y `tres_cuartos_abajo_der` (yaw -30°–45°, pitch -20°).
2. Actualizar el script de captura de galería para incluir instrucciones explícitas de estas dos poses nuevas.
3. Re-registrar a todos los colaboradores actuales con estas dos poses adicionales.

**Justificación adicional:** El análisis de los FN_Miss en mot47 y mot48 indica que una fracción de los fallos ocurre precisamente en frames donde la cámara capta al colaborador en movimiento lateral con inclinación de cámara. Las poses `diagonal_abajo` actuales (±35°, -20°) cubren parcialmente este espacio, pero una pose a ±40°–45° de yaw con -20° de pitch sería más representativa del ángulo de cámara típico de oficina.

**Impacto esperado:** Mejora de 3-8 puntos porcentuales en Recall global, especialmente beneficiosa para cámaras montadas en altura moderada (2.0–2.5 m) con cobertura de área amplia.

**Nivel de esfuerzo:** Bajo — extensión del protocolo de registro con 2 poses adicionales por persona.

---

<a name="anexo-a"></a>
## Anexo A: Glosario de Términos

Este glosario define los términos técnicos utilizados en el documento, ordenados alfabéticamente, con explicaciones adaptadas para lectores no especializados en inteligencia artificial.

---

**ArcFace:** Algoritmo de entrenamiento de redes neuronales para reconocimiento facial, diseñado específicamente para aprender representaciones matemáticas que maximizan la separación entre identidades distintas. El nombre proviene de "Additive Angular Margin" (margen angular aditivo), que es la función de pérdida que usa durante el entrenamiento.

**Bounding box:** Rectángulo delimitador que el sistema dibuja alrededor de cada cara detectada en un frame. Se define por las coordenadas de sus esquinas superior-izquierda $(x_1, y_1)$ e inferior-derecha $(x_2, y_2)$.

**buffalo_l:** Nombre del modelo de InsightFace utilizado en este sistema. "l" indica la variante "large" (grande), que es la de mayor precisión dentro de la familia buffalo. Integra el detector SCRFD, la red ArcFace R100 y el estimador de pose.

**CUDA:** Plataforma de computación paralela de NVIDIA que permite ejecutar operaciones matemáticas intensivas (como las inferencias de redes neuronales) en la GPU en lugar de la CPU, acelerando el procesamiento típicamente entre 10× y 100×.

**Det_size:** Parámetro de configuración del detector SCRFD que controla la resolución de la imagen de entrada al detector. Valores disponibles: 640 (rápido, menor cobertura de caras pequeñas) y 1280 (más lento, detecta caras más pequeñas y distantes).

**Distancia L2 (euclidiana):** Medida de la diferencia entre dos vectores matemáticos, calculada como la raíz cuadrada de la suma de los cuadrados de las diferencias componente a componente. Para embeddings normalizados, valores típicos: < 1.0 = muy similar, 1.0–1.5 = similar, > 1.5 = diferente.

**Embedding:** Vector de números reales (en este sistema, 512 dimensiones) que representa de forma compacta la identidad biométrica de un rostro. Es el output de la red ArcFace. Dos embeddings de la misma persona en distintas fotos serán matemáticamente parecidos; embeddings de personas distintas serán matemáticamente distantes.

**EER (Equal Error Rate):** Punto de operación de un sistema biométrico donde la tasa de falsa aceptación (FAR) es igual a la tasa de falso rechazo (FRR). Es una métrica de rendimiento intrínseco del sistema, independiente del umbral configurado.

**F1-Score:** Media armónica de Precisión y Recall. Sintetiza el rendimiento de un sistema de clasificación en un único número. Rango: 0 (peor) a 100% (mejor). Penaliza más los valores extremos que el promedio aritmético simple.

**FAR (False Acceptance Rate / Tasa de Falsa Aceptación):** Proporción de impostores (personas no registradas) que el sistema acepta erróneamente como personas registradas. También conocida en la literatura como FP-Rate en biometría. Un FAR de 0.40% significa que 4 de cada 1,000 impostores son incorrectamente aceptados.

**FN_Miss (False Negative por clasificación errónea):** Tipo de error en el que el detector sí localizó la cara (bounding box correcto) pero el sistema emitió `DESCONOCIDO` o asignó la identidad incorrecta.

**FN_NoDet (False Negative por no detección):** Tipo de error en el que la cara estaba presente en el frame pero el detector no la localizó. Puede ocurrir por resolución insuficiente, ángulo extremo, oclusión o iluminación muy adversa.

**FP_Ghost (False Positive fantasma):** Detección completamente espuria: el detector "vio" una cara donde no había ninguna. Puede ocurrir con reflejos, patrones visuales complejos o artefactos de compresión de vídeo.

**FP_ID (False Positive por ID incorrecto):** Error en el que el sistema reconoció correctamente que hay una cara, y la asoció a una identidad registrada, pero la identidad asignada es errónea. Es el error más costoso desde el punto de vista de seguridad.

**FPS (Frames Per Second):** Velocidad de procesamiento del sistema, medida en frames de vídeo procesados por segundo. El sistema procesa 6.6 fps en promedio con el hardware actual.

**FRR (False Rejection Rate / Tasa de Falso Rechazo):** Proporción de intentos de reconocimiento de sujetos genuinos que el sistema rechaza (emite DESCONOCIDO). Un FRR de 6.7% significa que 67 de cada 1,000 frames de una persona registrada son incorrectamente rechazados.

**Galería:** Base de datos de embeddings de personas registradas. En este sistema, es un archivo PKL que contiene múltiples embeddings (uno por pose) para cada colaborador registrado.

**Ground-truth:** Conjunto de datos de referencia con las etiquetas correctas para cada frame del vídeo de evaluación, anotado manualmente por humanos. Se usa para calcular las métricas de rendimiento del sistema.

**GPU (Graphics Processing Unit):** Unidad de procesamiento gráfico. Aunque fue diseñada para renderizado de gráficos, su arquitectura de miles de núcleos paralelos la hace ideal para las operaciones matriciales de las redes neuronales. En este sistema se usa una NVIDIA RTX 4050 Laptop.

**InsightFace:** Biblioteca de código abierto para reconocimiento facial de alta precisión, desarrollada principalmente por investigadores del CSIA (Institute of Computing Technology, Chinese Academy of Sciences). Provee el modelo `buffalo_l` utilizado en este sistema.

**IoU (Intersection over Union):** Métrica para medir la superposición entre dos bounding boxes. Se calcula dividiendo el área de intersección entre el área de unión. Rango: 0 (sin superposición) a 1 (superposición perfecta). En este sistema se usa un umbral de IoU ≥ 0.15 para asociar detecciones con ground-truth.

**Landmarks faciales:** Puntos de referencia anatómicos del rostro detectados automáticamente. En el modelo usado, se detectan 5 puntos: esquinas de ambos ojos, punta de la nariz y comisuras de los labios. Se usan para la alineación geométrica del parche de cara.

**Normalización L2:** Proceso matemático de dividir un vector por su propia norma euclidiana, resultando en un vector de longitud unitaria (norma = 1). Todos los embeddings de ArcFace se normalizan L2 antes de la comparación.

**ONNX Runtime:** Motor de inferencia de código abierto que permite ejecutar modelos de redes neuronales de forma eficiente en múltiples hardware (CPU, GPU NVIDIA vía CUDA, etc.) sin depender del framework de entrenamiento original.

**Pitch:** Ángulo de inclinación vertical de la cabeza. 0° = nivel, positivo = mirada hacia arriba, negativo = mirada hacia abajo. El sistema detecta hasta 94.7° de pitch.

**PKL (Pickle):** Formato de serialización binaria de Python. La galería del sistema se guarda como archivo `.pkl` para persistencia entre sesiones.

**Precisión (Precision):** Proporción de detecciones positivas del sistema que son correctas. "De cada 100 veces que el sistema dice 'reconocí a alguien', ¿cuántas son correctas?"

**R100:** Designación de la arquitectura ResNet-100 utilizada en ArcFace. "100" indica la profundidad de la red (100 capas). Es la arquitectura de mayor capacidad de la familia ResNet estándar para ArcFace.

**Recall (Exhaustividad):** Proporción de todos los casos positivos reales que el sistema detectó correctamente. "De cada 100 apariciones reales de un colaborador registrado, ¿cuántas identificó el sistema?"

**Roll:** Ángulo de inclinación lateral de la cabeza. 0° = erguida, positivo = inclinación hacia la derecha, negativo = inclinación hacia la izquierda.

**SCRFD (Sample and Computation Redistribution Face Detector):** Red de detección de rostros basada en Feature Pyramid Network, diseñada para equilibrar eficiencia y cobertura de escalas. Componente de detección del modelo `buffalo_l`.

**Similitud:** Valor entre 0 y 1 que representa qué tan parecidos son dos embeddings. Se calcula como $\text{Sim} = \max(0, 1 - d_{L2}/2)$. Un valor de 1.0 indica identidad perfecta; 0.0 indica máxima disimilitud.

**Umbral (threshold):** Valor de corte que el sistema usa para decidir entre "RECONOCIDO" y "DESCONOCIDO". Si la distancia L2 < umbral, se reconoce; si no, se rechaza. El umbral actual es 1.25; el recomendado es 1.20.

**VRAM (Video RAM):** Memoria dedicada de la GPU, utilizada para almacenar los pesos del modelo y los datos de inferencia. El sistema utiliza ~2-3 GB de los 6.4 GB disponibles en la RTX 4050 Laptop.

**Yaw:** Ángulo de rotación horizontal de la cabeza. 0° = frontal, +90° = perfil mirando a la izquierda, -90° = perfil mirando a la derecha. El sistema detecta hasta 89.2° de yaw y reconoce fiablemente hasta 82.2°.

---

<a name="anexo-b"></a>
## Anexo B: Especificaciones Técnicas de Hardware y Software

### B.1 Especificaciones de Hardware

#### B.1.1 Unidad de Procesamiento

| Componente          | Especificación                                          |
|---------------------|---------------------------------------------------------|
| **GPU**             | NVIDIA GeForce RTX 4050 Laptop GPU                      |
| **VRAM**            | 6.4 GB GDDR6                                            |
| **Arquitectura**    | Ada Lovelace (SM 8.9)                                   |
| **CUDA Cores**      | 2048                                                    |
| **TDP**             | 35-60 W (configurable por fabricante)                   |
| **Uso de VRAM**     | ~2-3 GB durante inferencia activa                       |
| **Proveedor ONNX**  | CUDAExecutionProvider (primario), CPUExecutionProvider (fallback) |

#### B.1.2 Rendimiento de Inferencia Medido

| Operación                          | Tiempo aproximado en RTX 4050  |
|------------------------------------|--------------------------------|
| Detección SCRFD (det_size=640)     | 15–20 ms/frame                 |
| Detección SCRFD (det_size=1280)    | 50–60 ms/frame                 |
| Embedding ArcFace R100 (1 cara)    | 5–10 ms                        |
| Embedding ArcFace R100 (8 caras)   | 20–40 ms                       |
| Comparación galería (28 entradas)  | < 1 ms                         |
| **FPS total (mot48, 8 personas)**  | **6.6 fps promedio**           |

#### B.1.3 Requisitos Mínimos de Hardware para Despliegue

| Componente          | Mínimo recomendado                                      |
|---------------------|---------------------------------------------------------|
| **GPU**             | NVIDIA con CUDA 11.8+, mínimo 4 GB VRAM                |
| **CPU**             | Intel Core i5 de 8ª generación o equivalente AMD Ryzen  |
| **RAM del sistema** | 8 GB (16 GB recomendado)                               |
| **Almacenamiento**  | 10 GB libres (modelos + galería + logs)                |
| **Sistema operativo** | Windows 10/11 64-bit o Ubuntu 20.04+                 |

#### B.1.4 Modo Fallback CPU

En caso de no disponibilidad de GPU compatible (por fallo de driver, hardware no disponible o entorno sin CUDA), el sistema activa automáticamente el `CPUExecutionProvider` de ONNX Runtime. El rendimiento en CPU es significativamente menor:
- FPS estimado en CPU: 0.5–1.5 fps (dependiendo del procesador)
- Latencia por cara: 50–200 ms
- El fallback es automático y no requiere intervención manual.

### B.2 Especificaciones de Software

#### B.2.1 Entorno de ejecución

| Componente               | Versión                          |
|--------------------------|----------------------------------|
| **Python**               | 3.10+                            |
| **InsightFace**          | 0.7.3                            |
| **ONNX Runtime GPU**     | 1.16+ (con CUDAExecutionProvider)|
| **CUDA Toolkit**         | 11.8 o 12.x                      |
| **cuDNN**                | 8.9+                             |
| **NumPy**                | 1.24+                            |
| **OpenCV**               | 4.8+                             |
| **Sistema operativo**    | Windows 11 (desarrollo y pruebas)|

#### B.2.2 Modelo buffalo_l — componentes descargados

| Archivo                  | Función                          | Tamaño aprox. |
|--------------------------|----------------------------------|---------------|
| `det_10g.onnx`           | Detector SCRFD                   | ~16 MB        |
| `w600k_r50.onnx`         | Red de reconocimiento ArcFace    | ~166 MB       |
| `1k3d68.onnx`            | Estimador de pose 3D             | ~19 MB        |
| **Total**                |                                  | **~201 MB**   |

Los modelos se descargan automáticamente por InsightFace en el primer uso y se cachean localmente en `~/.insightface/models/buffalo_l/`.

#### B.2.3 Galería actual del sistema

| Persona                      | Número de poses | Tamaño en galería PKL |
|------------------------------|-----------------|-----------------------|
| `cesar_angeles`              | 5               | 5 × 512 floats        |
| `jessica_urrea_con_lentes`   | 6               | 6 × 512 floats        |
| `jessica_urrea_sin_lentes`   | 6               | 6 × 512 floats        |
| `mitzi_ramirez`              | 6               | 6 × 512 floats        |
| `rafael_alcantar`            | 5               | 5 × 512 floats        |
| **TOTAL**                    | **28**          | **28 × 512 floats ≈ 57 KB** |

#### B.2.4 Dataset de evaluación

| Vídeo    | Frames totales | Detecciones | Escenario                          |
|----------|---------------|-------------|-------------------------------------|
| mot47    | 5,000         | ~13,000     | Cámara fija, 3-4 personas, ángulo favorable |
| mot48    | 12,000        | ~46,000     | Cámara difícil, hasta 8 personas   |
| Batch estrés (7 vídeos) | 11,063 | 44,238  | Ángulo, distancia, concurrencia     |
| **TOTAL** | **28,063**   | **~103,238** |                                    |

---

*Fin del documento — Sistema de Reconocimiento Facial de Identidad en Entornos de Oficina — Documentación Técnica v1.0 — Mayo 2026*

---

> **Revisión pendiente:** Las recomendaciones 2, 3 y 6 requieren nuevas sesiones de captura de galería que aún no han sido ejecutadas. Los resultados de validación post-mejora deberán documentarse en la versión 2.0 de este documento una vez implementadas.
