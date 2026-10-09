import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time

# ==========================================
# 1. CONFIGURACION
# ==========================================

CARPETA_SONIDOS = "sonidos"

NOTAS = [
    "E5", "D5", "C5", "B4", "A4",
    "C4", "D4", "E4", "F4", "G4"
]

NUM_TECLAS = len(NOTAS)

UMBRAL_MOVIMIENTO = 0.055
SUAVIZADO = 0.75
TIEMPO_ENTRE_NOTAS = 0.30
DURACION_TECLA = 0.25

PUNTAS_DEDOS = [4, 8, 12, 16, 20]

posiciones_anteriores = [None] * NUM_TECLAS
ultima_nota = [0.0] * NUM_TECLAS
teclas_activas = [0.0] * NUM_TECLAS

# Paleta visual (BGR para OpenCV)
FONDO = (19, 17, 31)
BLANCO = (246, 245, 255)
VIOLETA = (190, 90, 255)
CIAN = (235, 225, 90)
GRIS = (170, 169, 190)
OSCURO = (29, 26, 45)

# Un color por tecla (BGR), se ilumina al tocarla
COLORES_TECLAS = [
    (90, 90, 245),    # rojo coral
    (60, 150, 255),   # naranja
    (50, 205, 250),   # amarillo
    (110, 215, 110),  # verde
    (200, 215, 70),   # turquesa
    (255, 165, 70),   # azul
    (245, 100, 130),  # índigo
    (225, 95, 190),   # violeta
    (200, 90, 245),   # magenta
    (165, 120, 255),  # rosa
]

# Teclas negras decorativas (solo visuales, no suenan ni cambian la lógica)
MOSTRAR_TECLAS_NEGRAS = True
# Entre qué teclas va una negra (False donde no existe, como entre B-C o E-F)
NEGRAS_ENTRE = [True, True, False, True, False, True, True, False, True]


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
        return

    if indice in sonidos:
        sonidos[indice].play()

    ultima_nota[indice] = ahora
    teclas_activas[indice] = ahora + DURACION_TECLA


# ==========================================
# 3. MEDIAPIPE
# ==========================================

mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils

detector = mp_manos.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ==========================================
# 4. DIBUJO DE ELEMENTOS REDONDEADOS
# ==========================================

def caja_redondeada(img, x1, y1, x2, y2, color, radio=16):
    """Dibuja una caja rellena con esquinas redondeadas."""

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


def panel_transparente(img, x1, y1, x2, y2, color, alpha=0.86):
    capa = img.copy()
    caja_redondeada(capa, x1, y1, x2, y2, color, 20)
    cv2.addWeighted(capa, alpha, img, 1 - alpha, 0, img)

# ==========================================
# HELPERS VISUALES NUEVOS
# ==========================================

def _mascara_redondeada(w, h, radio):
    """Máscara float 0..1 con esquinas redondeadas y bordes suaves."""
    radio = max(1, min(radio, w // 2, h // 2))
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (radio, 0), (w - 1 - radio, h - 1), 255, -1)
    cv2.rectangle(m, (0, radio), (w - 1, h - 1 - radio), 255, -1)
    for cx, cy in [(radio, radio), (w - 1 - radio, radio),
                   (radio, h - 1 - radio), (w - 1 - radio, h - 1 - radio)]:
        cv2.circle(m, (cx, cy), radio, 255, -1, cv2.LINE_AA)
    return m.astype(np.float32) / 255.0


def _mezclar(img, x1, y1, x2, y2, capa, mascara):
    """Mezcla una capa (h,w,3) sobre img usando una máscara, con recorte seguro."""
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

    img[ay1:ay2, ax1:ax2] = (capa * m + roi * (1 - m)).astype(np.uint8)


def caja_degradado(img, x1, y1, x2, y2, c_sup, c_inf, radio=12, alpha=1.0):
    """Caja redondeada con degradado vertical."""
    w, h = x2 - x1, y2 - y1
    if w <= 2 or h <= 2:
        return

    t = np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1, 1)
    grad = (np.array(c_sup, np.float32) * (1 - t)
            + np.array(c_inf, np.float32) * t)
    grad = np.broadcast_to(grad, (h, w, 3))

    _mezclar(img, x1, y1, x2, y2, grad, _mascara_redondeada(w, h, radio) * alpha)


def resplandor(img, x1, y1, x2, y2, color, expansion=18, intensidad=0.6, radio=14):
    """Glow suave alrededor de una caja."""
    gx1, gy1 = x1 - expansion, y1 - expansion
    gx2, gy2 = x2 + expansion, y2 + expansion
    w, h = gx2 - gx1, gy2 - gy1

    m = np.zeros((h, w), np.float32)
    m[expansion:h - expansion, expansion:w - expansion] = _mascara_redondeada(
        x2 - x1, y2 - y1, radio
    )
    m = cv2.GaussianBlur(m, (0, 0), expansion / 2.2) * intensidad

    capa = np.empty((h, w, 3), np.float32)
    capa[:] = color
    _mezclar(img, gx1, gy1, gx2, gy2, capa, m)


def sombra(img, x1, y1, x2, y2, radio=12, desplazamiento=6, intensidad=0.55):
    """Sombra difuminada debajo de una caja."""
    resplandor(
        img, x1, y1 + desplazamiento, x2, y2 + desplazamiento,
        (5, 4, 12), expansion=10, intensidad=intensidad, radio=radio
    )


# ==========================================
# 5. OBTENER LAS PUNTAS DE LOS DEDOS
# ==========================================

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

    # Mano izquierda: meñique, anular, medio, indice, pulgar
    if "Left" in manos_detectadas:
        mano = manos_detectadas["Left"]

        for punta in reversed(PUNTAS_DEDOS):
            punto = mano.landmark[punta]
            posiciones.append((
                int(punto.x * ancho),
                int(punto.y * alto),
                punto.y
            ))
    else:
        posiciones.extend([None] * 5)

    # Mano derecha: pulgar, indice, medio, anular, meñique
    if "Right" in manos_detectadas:
        mano = manos_detectadas["Right"]

        for punta in PUNTAS_DEDOS:
            punto = mano.landmark[punta]
            posiciones.append((
                int(punto.x * ancho),
                int(punto.y * alto),
                punto.y
            ))
    else:
        posiciones.extend([None] * 5)

    return posiciones


# ==========================================
# 6. INTERFAZ SUPERIOR
# ==========================================

def dibujar_encabezado(imagen, manos_detectadas):
    pass

# ==========================================
# 7. PIANO VISUAL PREMIUM
# ==========================================

def dibujar_piano(imagen, ancho, alto):
    margen = 10
    separacion = 3
    alto_teclas = int(alto * 0.30)
    y1 = alto - alto_teclas - 12
    y2 = alto - 12

    # Base oscura detrás de las teclas (rectángulo plano, nítido)
    cv2.rectangle(imagen, (0, y1 - 10), (ancho, alto), (38, 38, 40), -1)
    cv2.line(imagen, (0, y1 - 10), (ancho, y1 - 10), (70, 70, 74), 1, cv2.LINE_AA)

    ancho_util = ancho - 2 * margen - (NUM_TECLAS - 1) * separacion
    ancho_tecla = ancho_util // NUM_TECLAS
    ahora = time.time()

    xs = []

    # ---------- Teclas blancas ----------
    for i, nota in enumerate(NOTAS):
        x1 = margen + i * (ancho_tecla + separacion)
        x2 = x1 + ancho_tecla
        xs.append((x1, x2))

        activa = ahora < teclas_activas[i]
        color = COLORES_TECLAS[i % len(COLORES_TECLAS)]

        if activa:
            claro = tuple(min(255, int(c * 0.55 + 110)) for c in color)
            c_sup, c_inf = claro, color
            color_texto = BLANCO
        else:
            c_sup, c_inf = (255, 255, 255), (232, 232, 238)
            color_texto = (140, 135, 150)

        caja_degradado(imagen, x1, y1, x2, y2, c_sup, c_inf, 9)

        # Franja de color en la base (siempre visible, da identidad a cada tecla)
        if not activa:
            cv2.rectangle(
                imagen, (x1 + 14, y2 - 40), (x2 - 14, y2 - 37),
                color, -1, cv2.LINE_AA
            )

        # Nota pequeña
        escala, grosor = 0.36, 1
        (tw, _), _ = cv2.getTextSize(
            nota, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor
        )
        texto(
            imagen, nota,
            (x1 + (x2 - x1 - tw) // 2, y2 - 16),
            escala, color_texto, grosor
        )

    # ---------- Teclas negras decorativas ----------
    if MOSTRAR_TECLAS_NEGRAS:
        ancho_negra = int(ancho_tecla * 0.56)
        alto_negra = int((y2 - y1) * 0.60)

        for i, hay in enumerate(NEGRAS_ENTRE):
            if not hay:
                continue

            cx = (xs[i][1] + xs[i + 1][0]) // 2
            bx1 = cx - ancho_negra // 2
            bx2 = cx + ancho_negra // 2

            # Contorno fino + cuerpo con degradado
            caja_degradado(
                imagen, bx1 - 1, y1, bx2 + 1, y1 + alto_negra + 1,
                (10, 10, 10), (10, 10, 10), 6
            )
            caja_degradado(
                imagen, bx1, y1, bx2, y1 + alto_negra,
                (72, 72, 76), (24, 24, 26), 6
            )
            # Brillo fino en el borde superior
            cv2.line(
                imagen, (bx1 + 6, y1 + 3), (bx2 - 6, y1 + 3),
                (110, 110, 116), 1, cv2.LINE_AA
            )

# ==========================================
# 8. DETECTAR MOVIMIENTOS
# ==========================================

def dibujar_punta(imagen, x, y, activa, nota):
    color = COLORES_TECLAS[NOTAS.index(nota) % len(COLORES_TECLAS)] if activa else CIAN

    cv2.circle(imagen, (x, y), 10, color, 2, cv2.LINE_AA)
    cv2.circle(imagen, (x, y), 4, BLANCO, -1, cv2.LINE_AA)

    if activa:
        texto(imagen, nota, (x + 14, y - 10), 0.4, BLANCO, 1)



def detectar_notas(resultados, ancho, alto, imagen):
    global posiciones_anteriores

    puntas = obtener_puntas(resultados, ancho, alto)
    ahora = time.time()

    candidatos = []

    for i, posicion in enumerate(puntas):
        if posicion is None:
            posiciones_anteriores[i] = None
            continue

        x, y, y_normalizada = posicion
        anterior = posiciones_anteriores[i]

        if anterior is not None:
            y_suavizada = (
                SUAVIZADO * y_normalizada
                + (1 - SUAVIZADO) * anterior
            )

            desplazamiento = y_suavizada - anterior

            # Solo considerar movimientos claros hacia abajo
            if desplazamiento >= UMBRAL_MOVIMIENTO:
                candidatos.append((desplazamiento, i))

            posiciones_anteriores[i] = y_suavizada
        else:
            posiciones_anteriores[i] = y_normalizada

    # Elegir un único dedo: el que tenga el movimiento más marcado
    if candidatos:
        desplazamiento_maximo, indice = max(candidatos)

        # Reproducir solo una nota por intervalo
        if ahora - ultima_nota[indice] >= TIEMPO_ENTRE_NOTAS:
            reproducir_nota(indice)

    # Dibujar todos los dedos sin alterar su detección visual
    for i, posicion in enumerate(puntas):
        if posicion is None:
            continue

        x, y, _ = posicion
        activa = ahora < teclas_activas[i]
        dibujar_punta(imagen, x, y, activa, NOTAS[i])



# ==========================================
# 9. EJECUCION
# ==========================================

def main():
    camara = cv2.VideoCapture(0)

     # Solicitar una resolución adecuada para la cámara
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Crear la ventana del piano en pantalla completa
    nombre_ventana = "AIR PIANO | Instrumento virtual"

    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)

    cv2.setWindowProperty(
        nombre_ventana,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    if not camara.isOpened():
        print("Error: no se pudo abrir la cámara.")
        return

    # Aquí continúa el resto de tu código:
    # lectura de cámara, detección de manos y dibujo del piano.

    

    if not camara.isOpened():
        print("Error: no se pudo abrir la camara del PC.")
        detector.close()
        pygame.mixer.quit()
        return

    # Reducir un poco la resolución para mantener fluidez
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nAIR PIANO iniciado.")
    print("Muestra las manos frente a la camara.")
    print("Baja los dedos para tocar las notas.")
    print("Presiona ESC para salir.\n")

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

            # Conexiones finas y puntos de las manos
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

            detectar_notas(
                resultados, ancho, alto, imagen
            )

            dibujar_encabezado(imagen, hay_manos)
            dibujar_piano(imagen, ancho, alto)

            cv2.imshow("AIR PIANO | Instrumento virtual", imagen)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        camara.release()
        cv2.destroyAllWindows()
        detector.close()
        pygame.mixer.quit()
        print("AIR PIANO cerrado.")


if __name__ == "__main__":
    main()
