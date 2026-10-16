# AIR PIANO
# Estudiante: Karol Stefany Saavedra Suarez
Piano virtual que se toca **en el aire** con las manos frente a la cámara. Detecta los 10 dedos en tiempo real y reproduce la nota del dedo que baja. Tiene modo libre y dos canciones guiadas: *Jingle Bells* y la *Oda a la alegría* de Beethoven.

##  Instalación

```bash
pip install opencv-python mediapipe numpy pygame
python air_piano_v3.py
```

Estructura necesaria (cada nota puede ser `.wav` o `.mp3`):

```
air_piano_v3.py
sonidos/
├── C4  D4  E4  F4  G4
└── A4  B4  C5  D5  E5
```

## Uso

| Tecla | Acción |
|-------|--------|
| `1` / `2` / `3` | Modo libre / Jingle Bells / Beethoven |
| `M` | Menú |
| `R` | Reiniciar canción |
| `F` | Pantalla completa |
| `+` / `-` | Sensibilidad |
| `ESC` | Salir |

Baja el dedo para tocar. El **pulgar** se dobla hacia la palma.

En las canciones, el dedo que toca ahora muestra una flecha y el cartel **BAJA** (o **DOBLA** para el pulgar). La barra superior indica la nota, la mano, el dedo y las próximas notas.

##  Notas por dedo

| Mano | Meñique | Anular | Medio | Índice | Pulgar |
|------|:-------:|:------:|:-----:|:------:|:------:|
| Izquierda | C4 | D4 | E4 | F4 | G4 |
| Derecha | E5 | D5 | C5 | B4 | A4 |

Las canciones se transponen a sol-la-si-do-re para usar ambas manos.

##  Cómo funciona

1. **Captura:** OpenCV lee la cámara y voltea la imagen (espejo).
2. **Manos:** MediaPipe Hands entrega 21 puntos por mano; se usan las 5 puntas de dedos.
3. **Señal por dedo:** se mide cuánto baja la punta respecto a su nudillo, dividido por el tamaño de la palma (así no depende de la distancia a la cámara). El pulgar se mide por su distancia a la base del meñique, porque se dobla de lado.
4. **Pulsación:** la señal se suaviza y, si sube más que el umbral en una ventana de 0.22 s, cuenta como toque. Un dedo debe subir antes de poder tocar otra vez.
5. **Dedos vecinos:** índice, medio, anular y meñique de una mano compiten; solo suena el que más se movió, y los vecinos se bloquean 0.18 s.
6. **Modos:** en modo libre suena cualquier dedo; en canción solo avanza con la nota esperada.
7. **Pantalla:** barra de información fuera de la imagen (no tapa la cara), piano abajo y marcadores sobre los dedos. Audio con pygame.

##  Cómo se hizo

Se desarrolló por versiones: piano base con Jingle Bells → ventana de tiempo, normalización por palma y detección del pulgar → menú de modos, Beethoven, 10 dedos y señales visuales → aislamiento de dedos vecinos.

**Tecnologías:** Python, OpenCV, MediaPipe, NumPy.

## 🔧 Ajustes principales

Al inicio de `air_piano_v3.py`:

| Parámetro | Valor | Efecto |
|-----------|:-----:|--------|
| `UMBRAL_DEDO` / `UMBRAL_PULGAR` | 0.16 / 0.12 | Menor = más sensible |
| `CONFIRMACION_S` | 0.06 | Menor = menos retraso |
| `BLOQUEO_VECINOS_S` | 0.18 | Mayor = menos toques de vecinos |
| `ACORDES_MISMA_MANO` | `False` | `True` permite dos dedos vecinos a la vez |

## Problemas comunes

- **No detecta toques:** más luz y sube la sensibilidad con `+`.
- **Suena el dedo vecino:** sube `BLOQUEO_VECINOS_S` a 0.25.
- **No suena una nota:** revisa el nombre del archivo en `sonidos/`.

## Actividad de reconocimiento de números

**Enlace:** [[Actividad detección de números]](https://github.com/karolsaavedra/Numeros_CienciaDatos.git)(#)

**Descripción:** _(breve descripción de la actividad)_
