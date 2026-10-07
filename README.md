# Dactilograph

Reconocimiento en tiempo real del **alfabeto dactilológico de la Lengua de Signos Española (LSE)** mediante visión artificial.

Miniproyecto de la asignatura de Informática Gráfica.

## La idea

Una aplicación que, a través de la webcam, reconoce la letra del alfabeto dactilológico LSE que el usuario está haciendo con la mano y la muestra en pantalla, formando texto letra a letra.

No es un traductor de lengua de signos: la LSE funciona por signos con gramática propia. Este proyecto reconoce únicamente el **deletreo manual**, en el que cada letra corresponde a una postura de la mano.

Referencia del alfabeto: https://signame.es/alfabeto-dactilologico-lse/ (30 signos realizados con una sola mano: A–Z con Ñ, más CH, LL y RR).

## Cómo funciona

```
Webcam (OpenCV) → MediaPipe (21 puntos de la mano) → normalización → clasificador → letra en pantalla
```

1. **Captura:** OpenCV obtiene los fotogramas de la webcam.
2. **Detección:** MediaPipe extrae automáticamente los 21 puntos clave de la mano (x, y, z).
3. **Normalización:** los puntos se trasladan respecto a la muñeca, se escalan según el tamaño de la mano y se refleja la mano izquierda, para que el resultado no dependa de la posición ni de la distancia a la cámara. El resultado es un vector de 63 números.
4. **Clasificación:** un modelo pequeño y local, entrenado mediante aprendizaje supervisado, recibe esos 63 números y devuelve la letra más probable con su confianza.
5. **Estabilización:** una letra solo se confirma si se mantiene como la más probable durante un breve intervalo, evitando el parpadeo entre letras.
6. **Resultado:** se muestra el vídeo con el esqueleto de la mano, la letra detectada y el texto que se va formando.

## Aprendizaje supervisado

El modelo no recibe imágenes, sino los puntos de la mano ya extraídos por MediaPipe, lo que convierte el problema en una clasificación sencilla.

Los datos de entrenamiento son propios: mientras el usuario mantiene una letra delante de la cámara, se guardan en un CSV los puntos normalizados de cada fotograma junto con la letra correspondiente como etiqueta.

```
persona,letra,x0,y0,z0,...,x20,y20,z20
jose,A,0.00,0.00,0.00,...
```

Con esos datos se entrena un clasificador clásico (k-NN, SVM o similar), que se guarda y se usa después en la aplicación en tiempo real.

## Alcance

- **Núcleo:** las letras **estáticas** del alfabeto.
- **Ampliación posible:** las letras que requieren movimiento (por ejemplo, la Ñ es la N con movimiento).
- **Investigación:** comparar distintos clasificadores y formas de representar la mano, y analizar qué letras se confunden entre sí y cómo funciona con personas distintas a las del entrenamiento.

## Tecnologías

- **Python**
- **OpenCV** — captura y visualización
- **MediaPipe** (Tasks API, `HandLandmarker`) — detección de los puntos de la mano
- **NumPy / pandas** — normalización y datos
- **scikit-learn** — clasificadores y métricas
