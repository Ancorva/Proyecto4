#############################################################################
#                                                                           #
# Proyecto 4: Simulación de modulación 16-QAM para envío de imagenes por un #
# canal ruidoso                                                             #
#                                                                           #
# Estudiante: Andrés Corrales Vargas                                        #
# Carnet: B72400                                                            #
# Grupo: 2                                                                  #
#                                                                           #
#############################################################################

# *************************** PYTHON LIBRARIES **************************** #

from PIL import Image
from scipy import fft
import matplotlib.pyplot as plt
import numpy as np
import time

# ************************* FUNCIONES IMPORTANTES ************************* #


def fuente_info(imagen):
    '''
    Una función que simula una fuente de
    información al importar una imagen y
    retornar un vector de NumPy con las
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales

    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''
    img = Image.open(imagen)

    return np.array(img)


def rgb_a_bit(array_imagen):
    '''
    Convierte los pixeles de base
    decimal (de 0 a 255) a binaria
    (de 00000000 a 11111111).

    :param imagen: array de una imagen
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = array_imagen.shape

    # Número total de elementos (pixeles x canales)
    n_elementos = x * y * z

    # Convertir la imagen a un vector unidimensional de n_elementos
    pixeles = np.reshape(array_imagen, n_elementos)

    # Convertir los canales a base 2
    bits = [format(pixel, '08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))

    return bits_Rx.astype(int)


def canal_ruidoso(senal_Tx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano (potencia = varianza)
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx


def bits_a_rgb(bits_Rx, dimensiones):
    '''
    Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)


def modulador16QAM(bits, fc, mpb):
    '''
    Un método que simula el esquema de
    modulación digital 16-QAM.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpb: Cantidad de muestras por bit en onda portadora
    :return: El vector con la señal modulada
    :return: La potencia promedio de la señal modulada
    :return: La onda coseno de la señal portadora s(t)
    :return: La onda seno de la señal portadora s(t)
    :return: Los bits de entrada (señal moduladora)
    '''

    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits)  # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora s(t)
    Tc = 1 / fc  # Periodo [s],

    # Dado que dos bits son representados en un mismo periodo, se
    # multiplica mpb*2 (para representar dos símbolos por periodo)
    t_periodo = np.linspace(0, Tc, mpb*2)

    # Creación de un periodo de la portadora senoidal
    portadora_sen = np.sin(2*np.pi*fc*t_periodo)

    # Creación de un periodo de la portadora cosenoidal
    portadora_cos = np.cos(2*np.pi*fc*t_periodo)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpb)

    # Creación de la señal que se transmitirá por el canal ruidoso
    senal_Tx = np.zeros(t_simulacion.shape)

    # Señal moduladora (bits de entrada)
    moduladora = np.zeros(t_simulacion.shape)

    # 4. Asignar las formas de onda según los bits (16-QAM)
    for i in range(0, N, 4):
        if (bits[i] == 0 and bits[i+1] == 0):
            senal_Tx[i*mpb:(i+2)*mpb] = portadora_cos * -3
            moduladora[i*mpb:(i+1)*mpb] = 0
            moduladora[(i+1)*mpb:(i+2)*mpb] = 0

        elif (bits[i] == 0 and bits[i+1] == 1):
            senal_Tx[i*mpb:(i+2)*mpb] = portadora_cos * -1
            moduladora[i*mpb:(i+1)*mpb] = 0
            moduladora[(i+1)*mpb:(i+2)*mpb] = 1

        elif (bits[i] == 1 and bits[i+1] == 1):
            senal_Tx[i*mpb:(i+2)*mpb] = portadora_cos * 1
            moduladora[i*mpb:(i+1)*mpb] = 1
            moduladora[(i+1)*mpb:(i+2)*mpb] = 1

        else:
            senal_Tx[i*mpb:(i+2)*mpb] = portadora_cos * 3
            moduladora[i*mpb:(i+1)*mpb] = 1
            moduladora[(i+1)*mpb:(i+2)*mpb] = 0

        if (bits[i+2] == 0 and bits[i+3] == 0):
            senal_Tx[(i+2)*mpb:(i+4)*mpb] = portadora_sen * 3
            moduladora[(i+2)*mpb:(i+3)*mpb] = 0
            moduladora[(i+3)*mpb:(i+4)*mpb] = 0

        elif (bits[i+2] == 0 and bits[i+3] == 1):
            senal_Tx[(i+2)*mpb:(i+4)*mpb] = portadora_sen * 1
            moduladora[(i+2)*mpb:(i+3)*mpb] = 0
            moduladora[(i+3)*mpb:(i+4)*mpb] = 1

        elif (bits[i+2] == 1 and bits[i+3] == 1):
            senal_Tx[(i+2)*mpb:(i+4)*mpb] = portadora_sen * -1
            moduladora[(i+2)*mpb:(i+3)*mpb] = 1
            moduladora[(i+3)*mpb:(i+4)*mpb] = 1

        else:
            senal_Tx[(i+2)*mpb:(i+4)*mpb] = portadora_sen * -3
            moduladora[(i+2)*mpb:(i+3)*mpb] = 1
            moduladora[(i+3)*mpb:(i+4)*mpb] = 0

    # 5. Calcular la potencia promedio de la señal modulada
    P_senal_Tx = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)

    return senal_Tx, P_senal_Tx, portadora_sen, portadora_cos, moduladora


def demodulador16QAM(senal_Rx, portadora_sen, portadora_cos, mpb):
    '''
    Un método que simula un bloque demodulador de señales, bajo un
    esquema 16-QAM. El criterio de demodulación se basa en
    decodificación por detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora_sen: La parte seno de la onda portadora c(t)
    :param mpb: Número de muestras por bit
    :return: Los bits de la señal demodulada
    :return: La señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits (símbolos) en transmisión
    N = int(M / mpb)

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(senal_Rx.shape)

    # Demodulación:
    for i in range(0, N, 4):

        # Producto interno de dos funciones para la portadora coseno
        producto_cos = (senal_Rx[i*mpb:(i+2)*mpb] * portadora_cos)
        Ep_cos = np.sum(producto_cos)

        # Reconstruyendo la parte coseno de la señal demodulada
        senal_demodulada[i*mpb:(i+2)*mpb] = producto_cos

        # Producto interno de dos funciones para la portadora seno
        producto_sen = (senal_Rx[(i+2)*mpb:(i+4)*mpb] * portadora_sen)
        Ep_sen = np.sum(producto_sen)

        # Reconstruyendo la parte seno de la señal demodulada
        senal_demodulada[(i+2)*mpb:(i+4)*mpb] = producto_sen

        # Criterio de decisión por detección de energía: Las energias tienden
        # a ser muy altas y se encontró mediante varias pruebas, que el valor
        # tal que se pueda diferenciar un 1 de un 3 o o un -3 de un -1 es -40
        # y 40

        # Criterio de decisión por detección de energía en portadora coseno
        if Ep_cos > 40:
            bits_Rx[i] = 1
            bits_Rx[i+1] = 0

        elif Ep_cos > 0 and Ep_cos < 40:
            bits_Rx[i] = 1
            bits_Rx[i+1] = 1

        elif Ep_cos > -40 and Ep_cos < 0:
            bits_Rx[i] = 0
            bits_Rx[i+1] = 1

        elif Ep_cos < -40:
            bits_Rx[i] = 0
            bits_Rx[i+1] = 0

        # Criterio de decisión por detección de energía en portadora seno
        if Ep_sen > 40:
            bits_Rx[i+2] = 0
            bits_Rx[i+3] = 0

        elif Ep_sen > 0 and Ep_sen < 40:
            bits_Rx[i+2] = 0
            bits_Rx[i+3] = 1

        elif Ep_sen > -40 and Ep_sen < 0:
            bits_Rx[i+2] = 1
            bits_Rx[i+3] = 1

        elif Ep_sen < -40:
            bits_Rx[i+2] = 1
            bits_Rx[i+3] = 0

    return bits_Rx.astype(int), senal_demodulada


# ******************************* PARTE 1 ********************************* #

# Print de título
print("Parte 1: Señales e imagen transmitida y recuperada\n")

# Parámetros
fc = 5000  # Frecuencia de la portadora
mpb = 20   # Muestras por bit en la portadora
SNR = -5   # Relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema 16-QAM
senal_Tx, Pm, portadora_sen, portadora_cos, modul = modulador16QAM(bits_Tx,
                                                                   fc, mpb)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx, senal_demodulada = demodulador16QAM(senal_Rx, portadora_sen,
                                             portadora_cos, mpb)

# 6. Visualizar el cambio entre las señales
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
ax1.plot(modul[0:1000], color='r', lw=2)
ax1.set_title("Señales importantes de modulación 16-QAM")
ax1.set_ylabel('$b(t)$')

# La señal modulada por 16-QAM
ax2.plot(senal_Tx[0:1000], color='g', lw=2)
ax2.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_Rx[0:1000], color='b', lw=2)
ax3.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax4.plot(senal_demodulada[0:1000], color='m', lw=2)
ax4.set_ylabel('$b^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()

# 7. Se visualiza la imagen recibida
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10, 6))

# 8. Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 9. Calcular número de errores
print("\nTransmisión de imagen con SNR = {}".format(SNR))
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# 10. Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Imagen Transmitida')

# 11. Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Imagen Recuperada')
Fig.tight_layout()

plt.imshow(imagen_Rx)
plt.show()


# ******************************* PARTE 2 ********************************* #

# Print de título
print("\n\nParte 2: Ergodicidad de senal_Tx\n")

# 1. Cálculo del promedio estadístico de senal_Tx

# El promedio estadístico es también el valor esperado
E_senal_Tx = np.average(senal_Tx)

print("Promedio estadístico de senal_Tx: {:0.9f}".format(E_senal_Tx))

# 2. Cálculo del promedio temporal

# Muestras de la señal
Nm = len(senal_Tx)

# Número de símbolos (198 x 89 x 8 x 3)
Ns = Nm // mpb

# Tiempo de un periodo
Tc = 1 / fc

# Periodo en segundos de un ciclo

t_simulacion = np.linspace(0, Ns*Tc, Ns*mpb)

T = t_simulacion[len(t_simulacion)-1]

# Como se integra de 0 a infinito, no es necesario poner 2T en denominador
# antes de integral. Solo debe ponerse T.
A_senal_Tx = (1 / T) * np.trapz(senal_Tx, t_simulacion)

print("Promedio temporal de senal_Tx: {:0.9f}".format(A_senal_Tx))

# 3. Cálculo de porcentaje de error entre el promedio estadístico y temporal
Porcentaje_Error = (abs(A_senal_Tx-E_senal_Tx)/A_senal_Tx)*100

print("Porcentaje de error entre promedios: {:0.9f}%".format(Porcentaje_Error))

# Prueba de ergodicidad con 0.01% de tolerancia

if (Porcentaje_Error < 0.01):
    print("\nConclusión: senal_Tx es un proceso ergódico.")

else:
    print("\nConclusión: senal_Tx NO es un proceso ergódico.")


# ******************************* PARTE 3 ********************************* #

# Transformada de Fourier
senal_f = fft.fft(senal_Tx)

# Muestras de la señal
Nm = len(senal_Tx)

# Número de símbolos (198 x 89 x 8 x 3)
Ns = Nm // mpb

# Tiempo del símbolo = medio periodo de la onda portadora (dos símbolos por
# periodo por usar modulación 16-QAM)
Ts = 1 / (2*fc)

# Tiempo entre muestras (hay dos muestras por período de muestreo)
Tm = (2*Ts) / mpb

# Tiempo de la simulación
T = Ns * Ts

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2)

# Gráfica
print("\n\nParte 3: Densidad Espectral de Potencia de senal_Tx\n")
plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]), 2))
plt.xlim(0, 10000)
plt.title("Densidad Espectral de Potencia de senal_Tx")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid()
plt.show()
