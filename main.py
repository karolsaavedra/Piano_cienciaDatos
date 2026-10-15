import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time
from collections import deque

# ==========================================
# 1. CONFIGURACION
# ==========================================

CARPETA_SONIDOS = "sonidos"

# Teclado ordenado de grave a agudo, de izquierda a derecha:
#   Mano IZQUIERDA: menique, anular, medio, indice, pulgar -> C4 D4 E4 F4 G4
#   Mano DERECHA:   pulgar, indice, medio, anular, menique -> A4 B4 C5 D5 E5
# (son los mismos 10 archivos de sonido que ya tienes)
NOTAS = [
    "C4", "D4", "E4", "F4", "G4",
    "A4", "B4", "C5", "D5", "E5"
]

NOMBRES_DEDOS = [
    "Mano IZQUIERDA - Menique",
    "Mano IZQUIERDA - Anular",
    "Mano IZQUIERDA - Medio",
    "Mano IZQUIERDA - Indice",
    "Mano IZQUIERDA - Pulgar",
    "Mano DERECHA - Pulgar",
    "Mano DERECHA - Indice",
    "Mano DERECHA - Medio",
    "Mano DERECHA - Anular",
    "Mano DERECHA - Menique",
]

NUM_TECLAS = len(NOTAS)

# ---- SENSIBILIDAD ----
# MENOR umbral = MAS sensible. Tambien puedes cambiarla en vivo
# con las teclas + y - mientras el programa corre.
UMBRAL_DEDO = 0.16      # indice, medio, anular, menique
UMBRAL_PULGAR = 0.12    # pulgar (se mide al flexionarlo hacia la palma)
VENTANA_S = 0.22        # tiempo en el que se mide el movimiento (seg)
SUAVIZADO = 0.65        # 0 a 1: mas alto = reacciona mas rapido
HISTERESIS = 0.45
REARME_MAX_S = 0.45

# ---- AISLAMIENTO DE DEDOS (evita que un dedo "arrastre" al vecino) ----
# Al bajar el menique, el anular se mueve solo. Indice, medio, anular y
# menique de una misma mano compiten: solo suena el que MAS se movio.
CONFIRMACION_S = 0.06       # espera corta para ver cual dedo se mueve mas
BLOQUEO_VECINOS_S = 0.18    # tras un toque, los vecinos no pueden disparar
ACORDES_MISMA_MANO = False  # True = permite 2 dedos de una mano a la vez
RATIO_ACORDE = 0.85         # (solo si ACORDES_MISMA_MANO) cercania al mayor
SESGO_GUIA = 0.6            # en canciones, favorece el dedo esperado si se
                            # movio al menos 60% que el mayor (0 = apagado)

PULGARES = [4, 5]                       # el pulgar se mueve independiente
GRUPOS_ACOPLADOS = [[0, 1, 2, 3], [6, 7, 8, 9]]   # dedos que se arrastran
# El menique y el anular son mas cortos y se mueven menos: se compensa.
ESCALA_RANKING = [1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.2]

TIEMPO_ENTRE_NOTAS = 0.12
DURACION_TECLA = 0.25

PUNTAS_DEDOS = [4, 8, 12, 16, 20]

sensibilidad = 1.0      # 0.5 (poco sensible) a 2.0 (muy sensible)

historial = [deque(maxlen=12) for _ in range(NUM_TECLAS)]
senal_suave = [None] * NUM_TECLAS
dedos_armados = [True] * NUM_TECLAS
pico_senal = [0.0] * NUM_TECLAS
momento_desarme = [0.0] * NUM_TECLAS

estado_grupo = [
    {
        "pendiente_t": None, "elegibles": set(), "picos": {},
        "bloqueo_hasta": 0.0, "ganador": None
    }
    for _ in GRUPOS_ACOPLADOS
]

ultima_nota = [0.0] * NUM_TECLAS
teclas_activas = [0.0] * NUM_TECLAS

# ==========================================
# CANCIONES
# ==========================================
# Las dos canciones usan solo 5 notas (do-re-mi-fa-sol). Para que se toquen
# con AMBAS manos se suben a "sol-la-si-do-re" (misma melodia, otra altura):
# el sol lo toca el pulgar izquierdo y el resto la mano derecha.

TRANSPORTE = {
    "C4": "G4", "D4": "A4", "E4": "B4", "F4": "C5", "G4": "D5"
}


def transponer(lista):
    return [TRANSPORTE[n] for n in lista]


JINGLE_BELLS_BASE = [
    "E4", "E4", "E4",
    "E4", "E4", "E4",
    "E4", "G4", "C4", "D4", "E4",
    "F4", "F4", "F4", "F4", "F4",
    "E4", "E4", "E4", "E4",
    "D4", "D4", "E4", "D4", "G4",
    "E4", "E4", "E4",
    "E4", "E4", "E4",
    "E4", "G4", "C4", "D4", "E4",
    "F4", "F4", "F4", "F4", "F4",
    "E4", "E4", "E4", "E4",
    "G4", "G4", "F4", "D4", "C4"
]

# Beethoven - Himno a la alegria (Oda a la alegria)
ODA_ALEGRIA_BASE = [
    "E4", "E4", "F4", "G4", "G4", "F4", "E4", "D4",
    "C4", "C4", "D4", "E4", "E4", "D4", "D4",
    "E4", "E4", "F4", "G4", "G4", "F4", "E4", "D4",
    "C4", "C4", "D4", "E4", "D4", "C4", "C4"
]

CANCIONES = {
    "jingle": {
        "nombre": "JINGLE BELLS",
        "notas": transponer(JINGLE_BELLS_BASE)
    },
    "beethoven": {
        "nombre": "BEETHOVEN - ODA A LA ALEGRIA",
        "notas": transponer(ODA_ALEGRIA_BASE)
    },
}

# Estado de modo: "menu", "libre", "jingle" o "beethoven"
modo = "menu"
cancion_actual = []
paso_actual = 0
mensaje_error = ""
mensaje_hasta = 0.0

# Altura de la barra superior (queda FUERA de la imagen de la camara,
# asi no tapa tu cara).
BARRA = 150

# Paleta BGR
FONDO = (19, 17, 31)
BLANCO = (246, 245, 255)
VIOLETA = (190, 90, 255)
CIAN = (235, 225, 90)
GRIS = (170, 169, 190)
OSCURO = (29, 26, 45)
ROJO = (90, 90, 245)

COLORES_TECLAS = [
    (90, 90, 245),
    (60, 150, 255),
    (50, 205, 250),
    (110, 215, 110),
    (200, 215, 70),
    (255, 165, 70),
    (245, 100, 130),
    (225, 95, 190),
    (200, 90, 245),
    (165, 120, 255),
]

MOSTRAR_TECLAS_NEGRAS = True
NEGRAS_ENTRE = [
    True, True, False, True, True,
    True, False, True, True
]

# ==========================================
# 2. AUDIO
# ==========================================

pygame.mixer.init()
sonidos = {}

for i, nota in enumerate(NOTAS):
    rutas = [
        os.path.join(CARPETA_SONIDOS, nota + ".wav"),
        os.path.join(CARPETA_SONIDOS, nota + ".mp3")
    ]

    archivo = next((r for r in rutas if os.path.isfile(r)), None)

    if archivo:
        try:
            sonidos[i] = pygame.mixer.Sound(archivo)
            print(f"Sonido cargado: {nota}")
        except pygame.error as error:
            print(f"No se pudo cargar {archivo}: {error}")
    else:
        print(f"Falta el sonido de {nota}")


def reproducir_nota(indice):
    ahora = time.time()

    if ahora - ultima_nota[indice] < TIEMPO_ENTRE_NOTAS:
        return False

    if indice not in sonidos:
        print(f"No hay archivo de sonido para {NOTAS[indice]}")
        return False

    sonidos[indice].play()
    ultima_nota[indice] = ahora
    teclas_activas[indice] = ahora + DURACION_TECLA
    return True


# ==========================================
# 3. MEDIAPIPE
# ==========================================

mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils

detector = mp_manos.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ==========================================
# 4. FUNCIONES VISUALES
# ==========================================

def caja_redondeada(img, x1, y1, x2, y2, color, radio=16):
    alto, ancho = img.shape[:2]

    x1 = max(0, min(x1, ancho - 1))
    x2 = max(0, min(x2, ancho - 1))
    y1 = max(0, min(y1, alto - 1))
    y2 = max(0, min(y2, alto - 1))

    if x2 <= x1 or y2 <= y1:
        return

    radio = max(1, min(radio, (x2 - x1) // 2, (y2 - y1) // 2))

    cv2.rectangle(
        img, (x1 + radio, y1), (x2 - radio, y2),
        color, -1, cv2.LINE_AA
    )
    cv2.rectangle(
        img, (x1, y1 + radio), (x2, y2 - radio),
        color, -1, cv2.LINE_AA
    )

    for x, y in [
        (x1 + radio, y1 + radio),
        (x2 - radio, y1 + radio),
        (x1 + radio, y2 - radio),
        (x2 - radio, y2 - radio)
    ]:
        cv2.circle(img, (x, y), radio, color, -1, cv2.LINE_AA)


def texto(img, contenido, posicion, escala, color, grosor=1):
    cv2.putText(
        img, contenido, posicion,
        cv2.FONT_HERSHEY_SIMPLEX,
        escala, color, grosor, cv2.LINE_AA
    )


def texto_borde(img, contenido, posicion, escala, color, grosor=1):
    """Texto con contorno negro para que se lea sobre la camara."""
    texto(img, contenido, posicion, escala, (0, 0, 0), grosor + 3)
    texto(img, contenido, posicion, escala, color, grosor)


def panel_transparente(img, x1, y1, x2, y2, color, alpha=0.86):
    capa = img.copy()
    caja_redondeada(capa, x1, y1, x2, y2, color, 20)
    cv2.addWeighted(capa, alpha, img, 1 - alpha, 0, img)


def _mascara_redondeada(w, h, radio):
    radio = max(1, min(radio, w // 2, h // 2))
    m = np.zeros((h, w), np.uint8)

    cv2.rectangle(m, (radio, 0), (w - 1 - radio, h - 1), 255, -1)
    cv2.rectangle(m, (0, radio), (w - 1, h - 1 - radio), 255, -1)

    for cx, cy in [
        (radio, radio),
        (w - 1 - radio, radio),
        (radio, h - 1 - radio),
        (w - 1 - radio, h - 1 - radio)
    ]:
        cv2.circle(m, (cx, cy), radio, 255, -1, cv2.LINE_AA)

    return m.astype(np.float32) / 255.0


def _mezclar(img, x1, y1, x2, y2, capa, mascara):
    alto, ancho = img.shape[:2]

    ax1, ay1 = max(0, x1), max(0, y1)
    ax2, ay2 = min(ancho, x2), min(alto, y2)

    if ax2 <= ax1 or ay2 <= ay1:
        return

    ox, oy = ax1 - x1, ay1 - y1
    h, w = ay2 - ay1, ax2 - ax1

    capa = capa[oy:oy + h, ox:ox + w]
    m = mascara[oy:oy + h, ox:ox + w, None]
    roi = img[ay1:ay2, ax1:ax2].astype(np.float32)

    img[ay1:ay2, ax1:ax2] = (
        capa * m + roi * (1 - m)
    ).astype(np.uint8)


def caja_degradado(
    img, x1, y1, x2, y2,
    c_sup, c_inf, radio=12, alpha=1.0
):
    w, h = x2 - x1, y2 - y1

    if w <= 2 or h <= 2:
        return

    t = np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1, 1)

    grad = (
        np.array(c_sup, np.float32) * (1 - t)
        + np.array(c_inf, np.float32) * t
    )
    grad = np.broadcast_to(grad, (h, w, 3)).copy()

    _mezclar(
        img, x1, y1, x2, y2,
        grad, _mascara_redondeada(w, h, radio) * alpha
    )


# ==========================================
# 5. OBTENER PUNTAS Y SENAL DE CADA DEDO
# ==========================================

def calcular_senal(mano, punta, ancho, alto):
    """
    Devuelve un numero que SUBE cuando el dedo se presiona.

    - Indice/medio/anular/menique: altura de la punta respecto a su
      nudillo, dividida por el tamano de la palma.
    - Pulgar: se mide la distancia de la punta a la base del menique
      (el pulgar se dobla hacia la palma, no hacia abajo).
    """
    def p(i):
        lm = mano.landmark[i]
        return np.array([lm.x * ancho, lm.y * alto], dtype=np.float32)

    palma = max(1.0, float(np.linalg.norm(p(0) - p(9))))

    if punta == 4:
        distancia = float(np.linalg.norm(p(4) - p(17)))
        return -distancia / palma

    mcp = punta - 3
    return float(p(punta)[1] - p(mcp)[1]) / palma


def obtener_puntas(resultados, ancho, alto):
    manos_detectadas = {}

    if resultados.multi_hand_landmarks:
        clasificaciones = resultados.multi_handedness or []

        for i, mano in enumerate(resultados.multi_hand_landmarks):
            if i >= len(clasificaciones):
                continue

            lado = clasificaciones[i].classification[0].label
            manos_detectadas[lado] = mano

    posiciones = []

    # Mano izquierda: menique, anular, medio, indice, pulgar
    if "Left" in manos_detectadas:
        mano = manos_detectadas["Left"]

        for punta in reversed(PUNTAS_DEDOS):
            punto = mano.landmark[punta]
            posiciones.append({
                "x": int(punto.x * ancho),
                "y": int(punto.y * alto),
                "senal": calcular_senal(mano, punta, ancho, alto),
                "pulgar": punta == 4
            })
    else:
        posiciones.extend([None] * 5)

    # Mano derecha: pulgar, indice, medio, anular, menique
    if "Right" in manos_detectadas:
        mano = manos_detectadas["Right"]

        for punta in PUNTAS_DEDOS:
            punto = mano.landmark[punta]
            posiciones.append({
                "x": int(punto.x * ancho),
                "y": int(punto.y * alto),
                "senal": calcular_senal(mano, punta, ancho, alto),
                "pulgar": punta == 4
            })
    else:
        posiciones.extend([None] * 5)

    return posiciones


# ==========================================
# 6. ESTADO Y MODOS
# ==========================================

def reiniciar_detector():
    for i in range(NUM_TECLAS):
        historial[i].clear()
        senal_suave[i] = None
        dedos_armados[i] = True
        pico_senal[i] = 0.0

    for st in estado_grupo:
        st["pendiente_t"] = None
        st["elegibles"] = set()
        st["picos"] = {}
        st["bloqueo_hasta"] = 0.0
        st["ganador"] = None


def cambiar_modo(nuevo):
    global modo, cancion_actual, paso_actual, mensaje_hasta

    modo = nuevo
    paso_actual = 0
    mensaje_hasta = 0.0

    if nuevo in CANCIONES:
        cancion_actual = CANCIONES[nuevo]["notas"]
    else:
        cancion_actual = []

    reiniciar_detector()
    print(f"Modo: {nuevo}")


def indice_objetivo():
    """Indice del dedo/tecla que se debe tocar ahora (o None)."""
    if modo in CANCIONES and paso_actual < len(cancion_actual):
        return NOTAS.index(cancion_actual[paso_actual])
    return None


def nombre_del_modo():
    if modo == "libre":
        return "MODO LIBRE"
    if modo in CANCIONES:
        return CANCIONES[modo]["nombre"]
    return "MENU"


# ==========================================
# 7. BARRA SUPERIOR (FUERA DE LA CAMARA)
# ==========================================

def dibujar_fichas_notas(lienzo, ancho, x_inicio, y):
    """Fichas con las proximas notas. La primera es la actual."""
    siguientes = cancion_actual[paso_actual:paso_actual + 10]
    x = x_inicio
    w, h = 62, 40

    for k, nota in enumerate(siguientes):
        if x + w > ancho - 30:
            break

        if k == 0:
            caja_redondeada(lienzo, x, y, x + w, y + h, CIAN, 10)
            color_texto = OSCURO
        else:
            caja_redondeada(lienzo, x, y, x + w, y + h, (65, 60, 80), 10)
            color_texto = BLANCO if k < 4 else GRIS

        (tw, th), _ = cv2.getTextSize(
            nota, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        texto(
            lienzo, nota,
            (x + (w - tw) // 2, y + (h + th) // 2),
            0.6, color_texto, 2
        )
        x += w + 10


def dibujar_leyenda_libre(lienzo, ancho, x_inicio, y):
    """10 fichas de colores: una por dedo, con su nota."""
    w, h, sep = 56, 38, 8
    total = NUM_TECLAS * w + (NUM_TECLAS - 1) * sep

    if x_inicio + total > ancho - 20:
        x_inicio = max(20, ancho - 20 - total)

    texto(lienzo, "IZQUIERDA", (x_inicio, y - 8), 0.45, GRIS, 1)
    texto(
        lienzo, "DERECHA",
        (x_inicio + 5 * (w + sep), y - 8), 0.45, GRIS, 1
    )

    for i, nota in enumerate(NOTAS):
        x = x_inicio + i * (w + sep)
        caja_redondeada(lienzo, x, y, x + w, y + h, COLORES_TECLAS[i], 10)

        (tw, th), _ = cv2.getTextSize(
            nota, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        texto(
            lienzo, nota,
            (x + (w - tw) // 2, y + (h + th) // 2),
            0.55, BLANCO, 2
        )


def dibujar_encabezado(lienzo, manos_detectadas):
    alto, ancho = lienzo.shape[:2]

    cv2.rectangle(lienzo, (0, 0), (ancho, BARRA), OSCURO, -1)
    cv2.line(
        lienzo, (0, BARRA), (ancho, BARRA),
        (70, 70, 74), 2, cv2.LINE_AA
    )

    texto(
        lienzo, f"AIR PIANO | {nombre_del_modo()}",
        (30, 36), 0.8, BLANCO, 2
    )

    texto(
        lienzo, f"Sensibilidad: {int(sensibilidad * 100)}%  (+ / -)",
        (max(30, ancho - 330), 36), 0.5, GRIS, 1
    )

    texto(
        lienzo,
        "1 Libre | 2 Jingle Bells | 3 Beethoven | M Menu | "
        "R Reiniciar | F Pantalla | ESC Salir",
        (30, BARRA - 12), 0.45, GRIS, 1
    )

    x_derecha = 470

    if modo == "libre":
        texto(
            lienzo, "Toca cualquier tecla",
            (30, 78), 0.8, BLANCO, 2
        )
        texto(
            lienzo, "con los 10 dedos",
            (30, 108), 0.65, CIAN, 2
        )
        dibujar_leyenda_libre(lienzo, ancho, x_derecha, 70)

    elif modo in CANCIONES:
        total = len(cancion_actual)

        if paso_actual < total:
            nota = cancion_actual[paso_actual]
            indice = NOTAS.index(nota)

            texto(
                lienzo, f"PASO {paso_actual + 1} DE {total}",
                (30, 66), 0.55, CIAN, 2
            )

            # Barra de progreso
            x1, y1, x2, y2 = 215, 54, 420, 66
            caja_redondeada(lienzo, x1, y1, x2, y2, (65, 60, 80), 6)
            ancho_prog = int((x2 - x1) * paso_actual / total)
            if ancho_prog > 0:
                caja_redondeada(
                    lienzo, x1, y1, x1 + ancho_prog, y2, CIAN, 6
                )

            texto(
                lienzo, f"TOCA AHORA: {nota}",
                (30, 98), 0.95, BLANCO, 2
            )
            texto(
                lienzo, NOMBRES_DEDOS[indice],
                (30, 122), 0.55, COLORES_TECLAS[indice], 2
            )

            texto(
                lienzo, "PROXIMAS NOTAS",
                (x_derecha, 56), 0.45, GRIS, 1
            )
            dibujar_fichas_notas(lienzo, ancho, x_derecha, 64)

            if time.time() < mensaje_hasta:
                texto(
                    lienzo, mensaje_error,
                    (x_derecha, 128), 0.65, ROJO, 2
                )
            elif not manos_detectadas:
                texto(
                    lienzo, "Muestra tus manos frente a la camara",
                    (x_derecha, 128), 0.6, GRIS, 1
                )
            else:
                texto(
                    lienzo, "Baja el dedo con la flecha (pulgar: doblalo)",
                    (x_derecha, 128), 0.55, GRIS, 1
                )
        else:
            texto(
                lienzo, "CANCION COMPLETADA!",
                (30, 85), 0.9, CIAN, 2
            )
            texto(
                lienzo, "R = repetir | M = menu",
                (30, 115), 0.6, BLANCO, 1
            )

    else:
        texto(
            lienzo, "Elige un modo para empezar",
            (30, 85), 0.7, BLANCO, 2
        )


def dibujar_menu(lienzo):
    alto, ancho = lienzo.shape[:2]

    pw, ph = 680, 380
    x1 = (ancho - pw) // 2
    y1 = BARRA + max(20, (alto - BARRA - ph) // 3)

    panel_transparente(lienzo, x1, y1, x1 + pw, y1 + ph, OSCURO, 0.94)

    texto(lienzo, "ELIGE UN MODO", (x1 + 40, y1 + 58), 1.0, BLANCO, 2)

    opciones = [
        ("1", "MODO LIBRE", "Toca cualquier tecla con los 10 dedos"),
        ("2", "JINGLE BELLS", "Sigue la guia paso a paso"),
        ("3", "BEETHOVEN", "Oda a la alegria, paso a paso"),
    ]

    for k, (num, titulo, desc) in enumerate(opciones):
        yy = y1 + 100 + k * 84

        caja_redondeada(
            lienzo, x1 + 40, yy, x1 + 104, yy + 62,
            COLORES_TECLAS[k * 3 + 1], 12
        )
        texto(lienzo, num, (x1 + 62, yy + 42), 1.0, BLANCO, 2)

        texto(lienzo, titulo, (x1 + 130, yy + 28), 0.85, BLANCO, 2)
        texto(lienzo, desc, (x1 + 130, yy + 52), 0.55, GRIS, 1)

    texto(
        lienzo, "Presiona el numero en tu teclado",
        (x1 + 40, y1 + ph - 22), 0.55, CIAN, 1
    )


# ==========================================
# 8. PIANO VISUAL
# ==========================================

def dibujar_piano(imagen, ancho, alto):
    margen = 10
    separacion = 3
    alto_teclas = int(alto * 0.22)
    y1 = alto - alto_teclas - 10
    y2 = alto - 10

    cv2.rectangle(
        imagen, (0, y1 - 8), (ancho, alto),
        (38, 38, 40), -1
    )
    cv2.line(
        imagen, (0, y1 - 8), (ancho, y1 - 8),
        (70, 70, 74), 1, cv2.LINE_AA
    )

    ancho_util = (
        ancho - 2 * margen - (NUM_TECLAS - 1) * separacion
    )
    ancho_tecla = ancho_util // NUM_TECLAS

    ahora = time.time()
    xs = []
    objetivo = indice_objetivo()

    for i, nota in enumerate(NOTAS):
        x1 = margen + i * (ancho_tecla + separacion)
        x2 = x1 + ancho_tecla
        xs.append((x1, x2))

        activa = ahora < teclas_activas[i]
        color = COLORES_TECLAS[i % len(COLORES_TECLAS)]

        if activa:
            claro = tuple(
                min(255, int(c * 0.55 + 110)) for c in color
            )
            c_sup, c_inf = claro, color
            color_texto = BLANCO
        else:
            c_sup, c_inf = (255, 255, 255), (232, 232, 238)
            color_texto = (140, 135, 150)

        caja_degradado(
            imagen, x1, y1, x2, y2,
            c_sup, c_inf, 9
        )

        if not activa:
            cv2.rectangle(
                imagen,
                (x1 + 14, y2 - 40),
                (x2 - 14, y2 - 37),
                color, -1, cv2.LINE_AA
            )

        # Resalta la tecla que se debe tocar (parpadea)
        if i == objetivo and not activa:
            if int(ahora * 3) % 2 == 0:
                cv2.rectangle(
                    imagen, (x1, y1), (x2, y2),
                    CIAN, 4, cv2.LINE_AA
                )

        escala = 0.5
        grosor = 1

        (tw, _), _ = cv2.getTextSize(
            nota, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor
        )

        texto(
            imagen, nota,
            (x1 + (x2 - x1 - tw) // 2, y2 - 14),
            escala, color_texto, grosor
        )

    if MOSTRAR_TECLAS_NEGRAS:
        ancho_negra = int(ancho_tecla * 0.56)
        alto_negra = int((y2 - y1) * 0.60)

        for i, hay in enumerate(NEGRAS_ENTRE):
            if not hay:
                continue

            cx = (xs[i][1] + xs[i + 1][0]) // 2
            bx1 = cx - ancho_negra // 2
            bx2 = cx + ancho_negra // 2

            caja_degradado(
                imagen, bx1 - 1, y1, bx2 + 1,
                y1 + alto_negra + 1,
                (10, 10, 10), (10, 10, 10), 6
            )

            caja_degradado(
                imagen, bx1, y1, bx2,
                y1 + alto_negra,
                (72, 72, 76), (24, 24, 26), 6
            )

            cv2.line(
                imagen, (bx1 + 6, y1 + 3),
                (bx2 - 6, y1 + 3),
                (110, 110, 116), 1, cv2.LINE_AA
            )


# ==========================================
# 9. MARCADORES DE LOS DEDOS
# ==========================================

def dibujar_marcadores(lienzo, puntas, desplazamiento_y):
    """
    Cada dedo muestra su nota con el color de su tecla.
    El dedo que hay que bajar (en las canciones) tiene un anillo
    pulsante, una flecha animada hacia abajo y un cartel.
    """
    ahora = time.time()
    objetivo = indice_objetivo()

    for i, p in enumerate(puntas):
        if p is None:
            continue

        x = p["x"]
        y = p["y"] + desplazamiento_y
        color = COLORES_TECLAS[i]
        activa = ahora < teclas_activas[i]
        es_objetivo = (i == objetivo)

        if activa:
            cv2.circle(lienzo, (x, y), 15, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(lienzo, (x, y), 10, color, 2, cv2.LINE_AA)

        cv2.circle(lienzo, (x, y), 4, BLANCO, -1, cv2.LINE_AA)

        # Nota encima de cada dedo
        nota = NOTAS[i]
        (tw, _), _ = cv2.getTextSize(
            nota, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        texto_borde(
            lienzo, nota, (x - tw // 2, y - 20), 0.5,
            BLANCO if es_objetivo else color, 1
        )

        if es_objetivo:
            pulso = int(6 + 6 * abs(np.sin(ahora * 5)))
            cv2.circle(
                lienzo, (x, y), 18 + pulso, CIAN, 3, cv2.LINE_AA
            )

            base = y - 105 + int(10 * np.sin(ahora * 6))
            base = max(base, 12)

            cv2.arrowedLine(
                lienzo, (x, base), (x, base + 48),
                CIAN, 6, cv2.LINE_AA, tipLength=0.5
            )

            cartel = "DOBLA" if p["pulgar"] else "BAJA"
            (cw, _), _ = cv2.getTextSize(
                cartel, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            texto_borde(
                lienzo, cartel, (x - cw // 2, base - 8),
                0.7, CIAN, 2
            )


# ==========================================
# 10. DETECCION DE PULSACIONES
# ==========================================

def umbral_de(i):
    base = UMBRAL_PULGAR if i in PULGARES else UMBRAL_DEDO
    return base / sensibilidad


def actualizar_dedo(i, senal, ahora):
    """Suaviza la senal del dedo y devuelve cuanto bajo en la ventana."""
    if senal_suave[i] is None:
        senal_suave[i] = senal
    else:
        senal_suave[i] = (
            SUAVIZADO * senal + (1 - SUAVIZADO) * senal_suave[i]
        )

    s = senal_suave[i]
    historial[i].append((ahora, s))

    recientes = [v for t, v in historial[i] if ahora - t <= VENTANA_S]
    minimo = min(recientes) if recientes else s
    return s - minimo


def desarmar(i, ahora):
    dedos_armados[i] = False
    pico_senal[i] = senal_suave[i] if senal_suave[i] is not None else 0.0
    momento_desarme[i] = ahora


def revisar_rearme(i, ahora):
    """Un dedo vuelve a estar listo cuando sube un poco."""
    if dedos_armados[i] or senal_suave[i] is None:
        return

    s = senal_suave[i]
    pico_senal[i] = max(pico_senal[i], s)

    if pico_senal[i] - s >= umbral_de(i) * HISTERESIS:
        dedos_armados[i] = True
    elif ahora - momento_desarme[i] > REARME_MAX_S:
        dedos_armados[i] = True


def olvidar_movimiento(i, ahora):
    """Borra el historial reciente: ese movimiento ya fue atribuido."""
    if senal_suave[i] is not None:
        historial[i].clear()
        historial[i].append((ahora, senal_suave[i]))


def decidir_ganador(elegibles, picos):
    """Entre los dedos acoplados que se movieron, elige el que mas bajo."""
    puntajes = {
        i: picos.get(i, 0.0) * ESCALA_RANKING[i] for i in elegibles
    }
    mejor = max(puntajes, key=puntajes.get)

    # En las canciones se favorece el dedo que toca ahora
    objetivo = indice_objetivo()
    if (
        SESGO_GUIA
        and objetivo in puntajes
        and puntajes[objetivo] >= SESGO_GUIA * puntajes[mejor]
    ):
        mejor = objetivo

    jugar = [mejor]

    if ACORDES_MISMA_MANO:
        for i, v in puntajes.items():
            if i != mejor and v >= RATIO_ACORDE * puntajes[mejor]:
                jugar.append(i)

    return jugar, puntajes


def detectar_notas(resultados, ancho, alto):
    global paso_actual, mensaje_error, mensaje_hasta

    puntas = obtener_puntas(resultados, ancho, alto)
    ahora = time.time()

    subidas = [0.0] * NUM_TECLAS
    candidatos = {}

    for i, p in enumerate(puntas):
        if p is None:
            senal_suave[i] = None
            historial[i].clear()
            dedos_armados[i] = True
            continue

        subidas[i] = actualizar_dedo(i, p["senal"], ahora)
        revisar_rearme(i, ahora)

    # Pulgares: se mueven solos, se detectan directo
    for i in PULGARES:
        if puntas[i] is None:
            continue

        if dedos_armados[i] and subidas[i] >= umbral_de(i):
            desarmar(i, ahora)
            candidatos[i] = subidas[i]

    # Indice, medio, anular y menique de cada mano: compiten entre si
    for g, grupo in enumerate(GRUPOS_ACOPLADOS):
        st = estado_grupo[g]
        presentes = [i for i in grupo if puntas[i] is not None]

        if not presentes:
            st["pendiente_t"] = None
            continue

        cruzan = [
            i for i in presentes
            if dedos_armados[i] and subidas[i] >= umbral_de(i)
        ]

        nuevos = []
        for i in cruzan:
            en_bloqueo = (
                st["pendiente_t"] is None
                and ahora < st["bloqueo_hasta"]
                and i != st["ganador"]
            )
            if en_bloqueo:
                # Un vecino del dedo que acaba de tocar: era arrastre.
                desarmar(i, ahora)
            else:
                nuevos.append(i)

        if nuevos and st["pendiente_t"] is None:
            st["pendiente_t"] = ahora
            st["elegibles"] = set()
            st["picos"] = {}

        if st["pendiente_t"] is not None:
            for i in nuevos:
                st["elegibles"].add(i)
                desarmar(i, ahora)

            for i in presentes:
                st["picos"][i] = max(st["picos"].get(i, 0.0), subidas[i])

            esperado = ahora - st["pendiente_t"] >= CONFIRMACION_S

            if esperado and st["elegibles"]:
                jugar, puntajes = decidir_ganador(
                    st["elegibles"], st["picos"]
                )

                for i in grupo:
                    if i not in jugar and puntas[i] is not None:
                        olvidar_movimiento(i, ahora)

                for i in jugar:
                    candidatos[i] = puntajes[i]

                st["ganador"] = jugar[0]
                st["bloqueo_hasta"] = ahora + BLOQUEO_VECINOS_S
                st["pendiente_t"] = None

    if not candidatos:
        return puntas

    # MODO LIBRE: suena cada dedo ganador
    if modo == "libre":
        for i in sorted(candidatos):
            reproducir_nota(i)

    # MODO CANCION: avanza solo con la nota esperada
    elif modo in CANCIONES and paso_actual < len(cancion_actual):
        nota_esperada = cancion_actual[paso_actual]
        indice_esperado = NOTAS.index(nota_esperada)

        if indice_esperado in candidatos:
            if reproducir_nota(indice_esperado):
                paso_actual += 1
                mensaje_hasta = 0.0
                print(
                    f"Correcto: {nota_esperada} | "
                    f"Paso {paso_actual}/{len(cancion_actual)}"
                )
        else:
            fuerte = max(candidatos, key=candidatos.get)
            mensaje_error = (
                f"Esa fue {NOTAS[fuerte]}, toca {nota_esperada}"
            )
            mensaje_hasta = ahora + 1.2

    return puntas


# ==========================================
# 11. EJECUCION PRINCIPAL
# ==========================================

def main():
    global sensibilidad, paso_actual

    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        print("Error: no se pudo abrir la camara del PC.")
        detector.close()
        pygame.mixer.quit()
        return

    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    nombre_ventana = "AIR PIANO | Instrumento virtual"

    # KEEPRATIO evita que la imagen se estire o se corte al cambiar
    # el tamano de la ventana o al usar pantalla completa.
    cv2.namedWindow(
        nombre_ventana, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    )

    ventana_ajustada = False
    pantalla_completa = False

    print("\nAIR PIANO iniciado.")
    print("1 = Libre | 2 = Jingle Bells | 3 = Beethoven | M = Menu")
    print("R = reiniciar | F = pantalla completa | +/- = sensibilidad")
    print("ESC = salir\n")

    try:
        while True:
            correcto, imagen = camara.read()

            if not correcto:
                print("Error leyendo la imagen de la camara.")
                break

            imagen = cv2.flip(imagen, 1)
            alto, ancho = imagen.shape[:2]

            rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            resultados = detector.process(rgb)

            hay_manos = bool(resultados.multi_hand_landmarks)

            if hay_manos:
                for mano in resultados.multi_hand_landmarks:
                    mp_dibujo.draw_landmarks(
                        imagen,
                        mano,
                        mp_manos.HAND_CONNECTIONS,
                        mp_dibujo.DrawingSpec(
                            color=CIAN, thickness=2, circle_radius=2
                        ),
                        mp_dibujo.DrawingSpec(
                            color=VIOLETA, thickness=2
                        )
                    )

            puntas = detectar_notas(resultados, ancho, alto)

            # Lienzo: barra superior + imagen completa de la camara.
            # La barra queda fuera de la imagen, asi no tapa la cara.
            lienzo = np.zeros((alto + BARRA, ancho, 3), np.uint8)
            lienzo[BARRA:] = imagen

            dibujar_encabezado(lienzo, hay_manos)
            dibujar_piano(lienzo, ancho, alto + BARRA)
            dibujar_marcadores(lienzo, puntas, BARRA)

            if modo == "menu":
                dibujar_menu(lienzo)

            if not ventana_ajustada:
                ancho_v = min(ancho, 1280)
                alto_v = int(ancho_v * (alto + BARRA) / ancho)
                cv2.resizeWindow(nombre_ventana, ancho_v, alto_v)
                ventana_ajustada = True

            cv2.imshow(nombre_ventana, lienzo)

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == 27:
                break
            elif tecla == ord("1"):
                cambiar_modo("libre")
            elif tecla == ord("2"):
                cambiar_modo("jingle")
            elif tecla == ord("3"):
                cambiar_modo("beethoven")
            elif tecla in (ord("m"), ord("M")):
                cambiar_modo("menu")
            elif tecla in (ord("r"), ord("R")):
                paso_actual = 0
                reiniciar_detector()
                print("Reiniciado.")
            elif tecla in (ord("+"), ord("=")):
                sensibilidad = min(2.0, round(sensibilidad + 0.1, 2))
            elif tecla in (ord("-"), ord("_")):
                sensibilidad = max(0.5, round(sensibilidad - 0.1, 2))
            elif tecla in (ord("f"), ord("F")):
                pantalla_completa = not pantalla_completa
                cv2.setWindowProperty(
                    nombre_ventana,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN
                    if pantalla_completa else cv2.WINDOW_NORMAL
                )

    finally:
        camara.release()
        cv2.destroyAllWindows()
        detector.close()
        pygame.mixer.quit()
        print("AIR PIANO cerrado.")


if __name__ == "__main__":
    main()