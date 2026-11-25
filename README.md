# Dactilograph

Reconocimiento en tiempo real del **alfabeto dactilológico de la Lengua de Signos Española (LSE)** mediante visión artificial, con una webcam normal, sin GPU y sin conexión.

## La idea

Una aplicación que, a través de la webcam, reconoce la letra del alfabeto dactilológico LSE que el usuario está haciendo con la mano y la muestra en pantalla.

No es un traductor de lengua de signos: la LSE funciona por signos con gramática propia. Este proyecto reconoce únicamente el **deletreo manual**, en el que cada letra corresponde a una postura de la mano o, en algunas letras, a una postura con movimiento.

Referencia del alfabeto: [Fundación CNSE, Dactilología](https://fundacioncnse.org/educa/bancolse/dactilologico.php) (30 signos realizados con una sola mano: A–Z con Ñ, más CH, LL y RR).

## Cómo funciona

```
Webcam (OpenCV) → MediaPipe (21 puntos de la mano) → normalización → clasificador → letra en pantalla
```

1. **Captura:** OpenCV obtiene los fotogramas de la webcam.
2. **Detección:** MediaPipe extrae automáticamente los 21 puntos clave de la mano (x, y, z).
3. **Normalización:** las coordenadas se pasan a píxeles cuadrados, se trasladan respecto a la muñeca, se escalan por el tamaño de la palma (la distancia entre la muñeca y la base del dedo corazón, que no cambia entre el puño y la mano abierta) y se refleja la mano izquierda. Así el resultado no depende de la posición ni de la distancia a la cámara. El resultado es un vector de 63 números. La rotación no se normaliza a propósito: algunas letras, como la G y la H frente a la I y la U, se distinguen sobre todo por la orientación de la mano.
4. **Clasificación de letras estáticas:** una SVM recibe esos 63 números y devuelve la letra más probable con su seguridad. Una letra solo cuenta si se mantiene 0,4 s con una seguridad de al menos 0,6.
5. **Clasificación de letras con movimiento:** se guarda el último segundo y medio de fotogramas y un k-NN clasifica esa ventana a partir de la trayectoria de la muñeca y del índice y de la forma media de la mano. Una letra solo se acepta si sale igual en 3 clasificaciones seguidas, y después se espera 1 s antes de admitir otra.
6. **Resultado:** se muestra el vídeo con el esqueleto de la mano y la letra detectada.

## Aprendizaje supervisado

El modelo no recibe imágenes, sino los puntos de la mano ya extraídos por MediaPipe, lo que convierte el problema en una clasificación sencilla.

Los datos de entrenamiento son propios: mientras el usuario mantiene una letra delante de la cámara, se guardan en un CSV los puntos normalizados de cada fotograma junto con la letra correspondiente como etiqueta.

```
persona,letra,x0,y0,z0,...,x20,y20,z20
jose,A,0.00,0.00,0.00,...
```

Cada letra estática se grabó dos veces con dos cámaras distintas, 100 fotogramas por grabación. Las letras con movimiento tienen 3 secuencias por letra. Todo el dataset es de una sola persona.

| | Letras | Grabaciones | Fotogramas |
|---|---:|---:|---:|
| Estáticas | 21 | 42 | 4 200 |
| Con movimiento | 9 | 27 | 1 202 |

Para las letras con movimiento, cada muestra es una secuencia de unos 1,5 s, unos 45 fotogramas a 30 FPS. `data/sequences.csv` guarda una fila por fotograma con los puntos de MediaPipe sin normalizar, de modo que las características se pueden recalcular sin volver a grabar. Normalizar cada fotograma por separado borraría el desplazamiento de la mano, así que la trayectoria se mide desde la muñeca del primer fotograma.

Cada secuencia se describe con 143 números: la trayectoria de la muñeca y del índice en el plano, medida en palmas e interpolada a 20 instantes equiespaciados (80 valores), y la forma media de la mano en la ventana (63 valores). Al interpolar no importan la velocidad del gesto, los FPS de la cámara ni los fotogramas en los que se perdió la mano.

Los ejemplos de "ninguna letra con movimiento" se generan al entrenar a partir de las grabaciones de letras estáticas, colocando la mano con un pequeño temblor aleatorio para que no aprenda una quietud perfecta. También se pueden grabar con la etiqueta `-`.

## Evaluación y resultados

Mezclar fotogramas al azar entre entrenamiento y prueba da cifras engañosas: los 100 fotogramas de una grabación son casi idénticos, y un k-NN llegaba así al 99,8 %. Por eso cada grabación entera va a un solo pliegue y los modelos se evalúan siempre con grabaciones que no han visto.

| Letras estáticas | Tal cual | Mano girada |
|---|---:|---:|
| k-NN (k = 5) | 76,9 % | 73,0 % |
| k-NN + aumento | 78,6 % | 77,6 % |
| SVM (RBF) | 81,5 % | 75,4 % |
| **SVM + aumento** (elegido) | **79,2 %** | **79,3 %** |

"Mano girada" aplica una transformación afín aleatoria a cada fotograma de prueba. La SVM sin aumento acierta más con la mano tal cual, pero cae en cuanto gira un poco; con aumento se mantiene, así que es la que usa la aplicación. Va calibrada para que devuelva probabilidades, que la aplicación muestra como seguridad.

Para las letras con movimiento se elige un **k-NN con k = 1 y aumento**: F1 medio de 0,69 con secuencias no vistas y el 95,2 % de los negativos bien reconocidos. Una SVM descartaba casi todos los negativos como letras, y un k mayor haría que los negativos ganaran por número a las 3 muestras de cada letra.

## Aumento de datos con transformaciones afines

Los scripts de entrenamiento generan automáticamente copias de cada muestra con una transformación afín aleatoria sobre los puntos de la mano. Cada copia combina en una sola matriz una rotación de hasta ±15° alrededor de la muñeca, una escala distinta en X e Y de hasta ±10 % y una inclinación de hasta ±0,1. La traslación y la escala uniforme no aportan variedad por sí solas, porque la normalización las anula. En una secuencia, todos los fotogramas comparten la misma matriz para no inventar movimiento.

Las copias se añaden solo al entrenamiento de cada pliegue. Cada modelo se evalúa con y sin aumento, tanto con las muestras de prueba tal cual como con esas muestras transformadas, y se guarda el que mejor resultado da en conjunto. `--copias` indica cuántas copias se generan por muestra, y con `--copias 0` no se usa aumento.

`uv run scripts/plot_augmentation.py` dibuja una mano junto a sus transformaciones y las matrices usadas.

## Rendimiento

En portátiles con poca potencia, o sin cargador, los FPS de la cámara pueden bajar a unos 14. Como la trayectoria se interpola siempre a 20 instantes, la ventana de 1,5 s solo necesita 12 fotogramas con mano, unos 8 FPS: quitando fotogramas a las secuencias grabadas, la clasificación no empeora. La cámara se lee en un hilo aparte que guarda solo el último fotograma con su instante de captura, para no procesar imágenes atrasadas. La tecla `f` de la aplicación muestra los FPS.

## Alcance

- **Letras estáticas (21):** A, B, C, CH, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U.
- **Letras con movimiento (9):** J, LL, Ñ, RR, V, W, X, Y y Z.
- **Limitaciones:** el dataset es de una sola persona, así que no se ha medido cómo funciona con otras, y las letras con movimiento tienen pocas secuencias por letra. La aplicación sigue una sola mano, que basta para el alfabeto LSE.

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

Los modelos se guardan en `models/`, que no se versiona, así que hay que generarlos tras clonar el repositorio. La aplicación los descarga y entrena ella misma la primera vez si no los encuentra.

## Uso

```bash
uv run scripts/app.py                                  # aplicación completa
uv run scripts/recognize.py                            # reconocimiento en tiempo real
uv run scripts/play.py                                 # juego de burbujas para practicar
uv run scripts/collect_data.py --persona <nombre>      # graba letras estáticas
uv run scripts/collect_sequences.py --persona <nombre> # graba letras con movimiento
```

Tras grabar datos nuevos, hay que volver a entrenar el modelo correspondiente.

## Aplicación

`scripts/app.py` reúne todo en una sola ventana con aspecto de pizarra de aula, pensada para usarse con una mano mientras la otra signa: cada acción tiene un botón y una tecla.

- **Aprender** (`1`): recorre las 30 letras. Muestra la letra, si es estática o con movimiento y la mano de referencia; al hacerla bien la rodea a tiza y la marca como aprendida. Espacio pasa a la siguiente y retroceso vuelve a la anterior, o se puede elegir cualquier letra en la fila de arriba.
- **Reconocer** (`2`): muestra la letra que ve el modelo, su seguridad y las últimas letras confirmadas.
- **Jugar** (`3`): el juego de burbujas, con récord.

`Esc` vuelve al menú, `f` muestra los fotogramas por segundo de la cámara y `q` sale. Las letras aprendidas y el récord se guardan en `~/.dactilograph/progreso.json`.

## Juego de burbujas

Para practicar el alfabeto, `scripts/play.py` hace subir burbujas con letras. Una burbuja explota al hacer su letra: las estáticas hay que mantenerlas un momento y las de movimiento se reconocen al terminar el gesto. Si una burbuja llega arriba se pierde una vida, y la partida termina al perder las tres. Cada cinco burbujas se sube de nivel, y las burbujas suben más rápido y aparecen más a menudo. Si una burbuja tarda en explotar, se muestra en una esquina cómo se hace su letra, con la mano media del dataset. Con `r` se empieza otra partida.

## Desarrollo

```bash
uv run ruff check .      # linter
uv run ruff format .     # formato
uv run pytest            # tests
```
