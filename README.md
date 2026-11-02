# Dactilograph

Reconocimiento en tiempo real del **alfabeto dactilológico de la Lengua de Signos Española (LSE)** mediante visión artificial.

Miniproyecto de la asignatura de Informática Gráfica.

## La idea

Una aplicación que, a través de la webcam, reconoce la letra del alfabeto dactilológico LSE que el usuario está haciendo con la mano y la muestra en pantalla.

No es un traductor de lengua de signos: la LSE funciona por signos con gramática propia. Este proyecto reconoce únicamente el **deletreo manual**, en el que cada letra corresponde a una postura de la mano o, en algunas letras, a una postura con movimiento.

Referencia del alfabeto: https://signame.es/alfabeto-dactilologico-lse/ (30 signos realizados con una sola mano: A–Z con Ñ, más CH, LL y RR).

## Cómo funciona

```
Webcam (OpenCV) → MediaPipe (21 puntos de la mano) → normalización → clasificador → letra en pantalla
```

1. **Captura:** OpenCV obtiene los fotogramas de la webcam.
2. **Detección:** MediaPipe extrae automáticamente los 21 puntos clave de la mano (x, y, z).
3. **Normalización:** los puntos se trasladan respecto a la muñeca, se escalan según el tamaño de la mano y se refleja la mano izquierda, para que el resultado no dependa de la posición ni de la distancia a la cámara. El resultado es un vector de 63 números.
4. **Clasificación de letras estáticas:** un modelo pequeño y local, entrenado mediante aprendizaje supervisado, recibe esos 63 números y devuelve la letra más probable con su confianza.
5. **Clasificación de letras con movimiento:** se guarda el último segundo y medio de fotogramas y un segundo modelo clasifica esa ventana a partir de la trayectoria de la muñeca y del índice y de la forma media de la mano. Una letra solo se acepta si se repite en varios fotogramas seguidos.
6. **Resultado:** se muestra el vídeo con el esqueleto de la mano y la letra detectada.

## Aprendizaje supervisado

El modelo no recibe imágenes, sino los puntos de la mano ya extraídos por MediaPipe, lo que convierte el problema en una clasificación sencilla.

Los datos de entrenamiento son propios: mientras el usuario mantiene una letra delante de la cámara, se guardan en un CSV los puntos normalizados de cada fotograma junto con la letra correspondiente como etiqueta.

```
persona,letra,x0,y0,z0,...,x20,y20,z20
jose,A,0.00,0.00,0.00,...
```

Con esos datos se entrena un clasificador clásico (k-NN, SVM o similar), que se guarda y se usa después en la aplicación en tiempo real.

Para las letras con movimiento, cada muestra es una secuencia de unos 1,5 s. `data/sequences.csv` guarda una fila por fotograma con los puntos de MediaPipe sin normalizar, de modo que las características se pueden recalcular sin volver a grabar. Normalizar cada fotograma por separado borraría el desplazamiento de la mano, así que la trayectoria se mide desde la muñeca del primer fotograma.

Los ejemplos de "ninguna letra con movimiento" se generan al entrenar a partir de las grabaciones de letras estáticas. También se pueden grabar con la etiqueta `-`.

## Aumento de datos con transformaciones afines

Los scripts de entrenamiento generan automáticamente copias de cada muestra con una transformación afín aleatoria sobre los puntos de la mano. Cada copia combina en una sola matriz una rotación de hasta ±15° alrededor de la muñeca, una escala distinta en X e Y de hasta ±10 % y una inclinación de hasta ±0,1. La traslación y la escala uniforme no aportan variedad por sí solas, porque la normalización las anula. En una secuencia, todos los fotogramas comparten la misma matriz para no inventar movimiento.

Las copias se añaden solo al entrenamiento de cada pliegue. Cada modelo se evalúa con y sin aumento, tanto con las muestras de prueba tal cual como con esas muestras transformadas, y se guarda el que mejor resultado da en conjunto. `--copias` indica cuántas copias se generan por muestra, y con `--copias 0` no se usa aumento.

`uv run scripts/plot_augmentation.py` dibuja una mano junto a sus transformaciones y las matrices usadas.

## Alcance

- **Letras estáticas (21):** A, B, C, CH, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U.
- **Letras con movimiento (9):** J, LL, Ñ, RR, V, W, X, Y y Z.
- **Investigación:** comparar distintos clasificadores y formas de representar la mano, y analizar qué letras se confunden entre sí y cómo funciona con personas distintas a las del entrenamiento.

## Tecnologías

- **Python**
- **OpenCV** — captura y visualización
- **MediaPipe** (Tasks API, `HandLandmarker`) — detección de los puntos de la mano
- **NumPy / pandas** — normalización y datos
- **scikit-learn** — clasificadores y métricas
- **matplotlib** — gráficas de resultados
- **uv** — entorno y dependencias · **ruff** — linter y formato · **pytest** — tests

## Instalación

Requiere [uv](https://docs.astral.sh/uv/), que instala por sí mismo la versión de Python adecuada.

```bash
uv sync                              # crea el entorno e instala las dependencias
uv run scripts/download_model.py     # descarga el modelo de MediaPipe a models/
uv run scripts/check_setup.py        # comprueba librerías, modelo y webcam
uv run scripts/train.py              # entrena el clasificador de letras estáticas
uv run scripts/train_dynamic.py      # entrena el clasificador de letras con movimiento
```

Los modelos se guardan en `models/`, que no se versiona, así que hay que generarlos tras clonar el repositorio.

## Uso

```bash
uv run scripts/recognize.py                            # reconocimiento en tiempo real
uv run scripts/play.py                                 # juego de burbujas para practicar
uv run scripts/collect_data.py --persona <nombre>      # graba letras estáticas
uv run scripts/collect_sequences.py --persona <nombre> # graba letras con movimiento
```

Tras grabar datos nuevos, hay que volver a entrenar el modelo correspondiente.

## Juego de burbujas

Para practicar el alfabeto, `scripts/play.py` hace subir burbujas con letras. Una burbuja explota al hacer su letra: las estáticas hay que mantenerlas un momento y las de movimiento se reconocen al terminar el gesto. Si una burbuja llega arriba se pierde una vida, y la partida termina al perder las tres. Cada cinco burbujas se sube de nivel, y las burbujas suben más rápido y aparecen más a menudo. Si una burbuja tarda en explotar, se muestra en una esquina cómo se hace su letra, con la mano media del dataset. Con `r` se empieza otra partida.

## Desarrollo

```bash
uv run ruff check .      # linter
uv run ruff format .     # formato
uv run pytest            # tests
```
