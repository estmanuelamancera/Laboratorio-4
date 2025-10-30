# Laboratorio4-Señales Electromiograficas
## DESCRIPCIÓN 
En este repositorio analizaremos la practica desarollada que se titula **"Señales Electromiograficas"**, se realizo la captura y análisis de señales emuladas y reales, aplicando filtrado, segmentación, análisis espectral (FFT).Ademas se calcularon las frecuencias media y mediana de las contracciones musculares para observar cómo varían durante el esfuerzo para detectar la aparición de la fatiga.
## OBJETIVOS
-Aplicar el filtrado de señales continuas para procesar una señal electromiográfica
(EMG).
-Detectar la aparición de fatiga muscular mediante el análisis espectral de
contracciones musculares individuales.
-Comparar el comportamiento de una señal emulada y una señal real en términos
de frecuencia media y mediana.
-Emplear herramientas computacionales para el procesamiento, segmentación y
análisis de señales biomédicas
# PROCEDIMIENTO
## PARTE A Captura de la señal emulada 
En esta primera parte del laboratorio se genero una señal EMG por medio del generador de señales el cual simuló cinco contracciones musculares voluntarias, representando la actividad eléctrica del músculo durante el esfuerzo,con una frecuencia de muestreo 5000 Hz y una duracion de 10 s. Los datos que se obtuvieron se almacenaron en un archivo de texto (Señal EMG5000.txt), para luego segmentarla por contraccion y calcular la frecuencia media y mediana de cada una, asi se puede observar cómo varía el contenido espectral a lo largo del tiempo.
Esta señal tuvo un analisis por energia,ya que la señal original es altamente ruidosa y oscilatoria, entonces el código calcula la energía local, que es la potencia promedio de la señal en una ventana de tiempo,asi los valores negativos desaparecen, ya que se elevan al cuadrado,las zonas con mayor actividad muscular (mayor amplitud EMG) se reflejan como picos de energía y se pueden detectar fácilmente las contracciones.

<img width="400" height="800" alt="image" src="cardiaca.png" /> <br>

### CODIGO

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import pandas as pd
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

```

En esta parte del codigo se importan las librerias necesarias para el analisis, como: scipy.signal para la deteccion de los picos y scipy.fft para la transformada rápida de Fourier (FFT), ademas se agrega Google Drive para acceder al archivo con la señal (Señal EMG5000.txt).

```

ruta = '/content/drive/MyDrive/Señal EMG5000.txt'
fs = 5000  # Frecuencia de muestreo [Hz]

emg = np.loadtxt(ruta)
T = 1/fs
t = np.arange(0, len(emg)*T, T)
print(f"Duración total: {t[-1]:.2f}s  |  Muestras: {len(emg)}")

```
En este segmento del codigo se define la frecuencia de muestreo, se carga la señal EMG desde el archivo que subimos,se calcula el vector de tiempo t para graficar la señal en el eje temporal y se muestra la duración total y número de muestras.

```
emg = emg - np.mean(emg)
if np.max(np.abs(emg)) != 0:
    emg = emg / np.max(np.abs(emg))

```

Este segmento es la normalizacion, se centra la señal en cero eliminando el offset y se divide por el valor máximo absoluto para que quede normalizada entre -1 y 1.

```
ventana = int(0.1 * fs)  # 100 ms
energia = np.convolve(emg**2, np.ones(ventana)/ventana, mode='same')
```
Este segmento es el calculo de energia local:Se calcula la energía promedio de la señal dentro de una ventana móvil de 100 ms para suavizar la señal y resaltar los periodos de contraccion.

```
peaks, _ = find_peaks(energia, distance=fs*0.8)
if len(peaks) > 5:
    peaks = peaks[:5]

anchura = int(0.5 * fs)
zonas_validas = [(max(0, p - anchura//2), min(len(emg), p + anchura//2)) for p in peaks]
print(f" Contracciones detectadas: {len(zonas_validas)}")
```

Esta parte del codigo hace la deteccion de contraccion de manera automatica, utilizando find_peaks() para detectar picos de energía, que corresponden a contracciones con un tiempo de 0,8s entre cada contraccion, se toman las primeras cinco contracciones y se crea un intervalo con estas cinco contracciones.


```
# --- (1) Señal original ---
plt.figure(figsize=(12,4))
plt.plot(t, emg, color='#4da6ff', linewidth=1.2)
plt.title('Señal EMG normalizada', color='#035a84')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True, alpha=0.4)
plt.show()

# --- (2) Energía local ---
plt.figure(figsize=(12,4))
plt.plot(t, energia, color='#7ec8e3', label='Energía local')
plt.title('Energía local de la señal EMG (potencia por ventana)', color='#035a84')
plt.xlabel('Tiempo [s]')
plt.ylabel('Energía')
plt.legend()
plt.grid(True, alpha=0.4)
plt.show()

# --- (3) Segmentación automática ---
plt.figure(figsize=(12,4))
plt.plot(t, emg, color='#4da6ff', label='Señal EMG', linewidth=1.2)
for i, (ini, fin) in enumerate(zonas_validas):
    plt.axvspan(t[ini], t[fin], color='#b3e0ff', alpha=0.6, label='Contracción' if i==0 else "")
plt.title('Segmentación automática de contracciones (por energía)', color='#035a84')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True, alpha=0.4)
plt.show()

```

En este segmento del codigo se graficaron tres señales: la origina que muestra la señal EMG normalizada, energia local que muestra cómo varía la potencia de la señal y segmentacion automatica indicando las sombras azules que muestran las zonas de contracción detectadas.


```
def mean_median_freq(signal, fs):
    N = len(signal)
    freqs = fftfreq(N, 1/fs)[:N//2]
    fft_vals = np.abs(fft(signal))[:N//2]
    P = fft_vals**2
    if np.sum(P) == 0:
        return 0, 0
    f_mean = np.sum(freqs * P) / np.sum(P)
    cumulative_power = np.cumsum(P)
    f_median = freqs[np.where(cumulative_power >= cumulative_power[-1]/2)[0][0]]
    return f_mean, f_median

# Calcular frecuencias para las contracciones detectadas
resultados = []
for i, (ini, fin) in enumerate(zonas_validas[:5], 1):
    contr = emg[ini:fin]
    f_mean, f_median = mean_median_freq(contr, fs)
    resultados.append([i, f_mean, f_median])

if len(resultados) > 0:
    tabla = pd.DataFrame(resultados, columns=['Contracción','Frecuencia media (Hz)','Frecuencia mediana (Hz)'])
    print("\nResultados de frecuencia por contracción:")
    display(tabla)

    # --- (4) Evolución de frecuencias ---
    plt.figure(figsize=(8,5))
    plt.plot(tabla['Contracción'], tabla['Frecuencia media (Hz)'], 'o-', color='#7ec8e3', label='Frecuencia media', linewidth=2)
    plt.plot(tabla['Contracción'], tabla['Frecuencia mediana (Hz)'], 's--', color='#035a84', label='Frecuencia mediana', linewidth=2)
    plt.title('Evolución de las frecuencias EMG por contracción', fontsize=13, color='#035a84')
    plt.xlabel('Contracción')
    plt.ylabel('Frecuencia [Hz]')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.show()
```
Esta parte  del código aplica la Transformada Rápida de Fourier (FFT) a cada contracción de la señal EMG para obtener su espectro de frecuencias.
A partir de ese espectro se calcula la frecuencia media, que representa el promedio ponderado del contenido en frecuencia, y la frecuencia mediana, que divide la energía espectral en dos mitades iguales.
Estas frecuencias permiten analizar cómo cambia el contenido de alta frecuencia durante el esfuerzo muscular: una disminución progresiva de estos valores indica la aparición de fatiga.
Finalmente, los resultados se presentan en una tabla y una gráfica que muestran la evolución de ambas frecuencias en cada contracción.


```
    contr1 = emg[zonas_validas[0][0]:zonas_validas[0][1]]
    N = len(contr1)
    freqs = fftfreq(N, 1/fs)[:N//2]
    fft_vals = np.abs(fft(contr1))[:N//2]
    P = fft_vals**2

    f_mean, f_median = mean_median_freq(contr1, fs)

    plt.figure(figsize=(8,5))
    plt.plot(freqs, P, color='#4da6ff', linewidth=1.2)
    plt.axvline(f_mean, color='#7ec8e3', linestyle='--', label=f'F media = {f_mean:.1f} Hz')
    plt.axvline(f_median, color='#035a84', linestyle='--', label=f'F mediana = {f_median:.1f} Hz')
    plt.title('Espectro de frecuencia (FFT) de una contracción', color='#035a84')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Potencia (a.u.)')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.show()

else:
    print("No se pudo calcular frecuencias porque no hay contracciones detectadas.")

```

La ultima parte del codigo selecciona la primera contracción detectada en la señal EMG (contr1) y se aplica la Transformada Rápida de Fourier (FFT) para obtener su espectro de amplitud.
La variable P representa la potencia espectral (energía contenida en cada frecuencia), que permite identificar qué rangos de frecuencia dominan la actividad muscular.Luego, el código calcula nuevamente la frecuencia media y la frecuencia mediana, y las muestra como líneas verticales sobre la gráfica:
La frecuencia media y La frecuencia mediana
Si no se detectan contracciones, el programa muestra un aviso para evitar errores en el cálculo.

### DIAGRAMAS

<img width="400" height="800" alt="image" src="img1.jpg" /> <br>

En la grafica se observa la señal EMG normalizada en amplitud,los valores del eje Y van de -1 a 1, ya que la señal fue escalada (dividida por su valor máximo) para que todos los datos queden dentro de ese rango,La amplitud indica la intensidad de la actividad eléctrica generada por las fibras musculares.

<img width="400" height="800" alt="image" src="img2.jpg" /> <br>

En la grafica se observa la energía local de la señal EMG,esta gráfica representa la energía promedio de la señal EMG a lo largo del tiempo, calculada usando una ventana deslizante de 100 ms.
La energía local refleja cuánta actividad eléctrica muscular hay en cada instante del registro,el cálculo de la energía local transforma una señal oscilatoria y ruidosa en una envolvente positiva y estable, donde los máximos de energía representan contracciones y los mínimos representan descanso o inactividad.

<img width="400" height="800" alt="image" src="img3.jpg" /> <br>

En la grafica se observa la detección y segmentación automática de las contracciones musculares a partir de la energía local calculada anteriormente,el objetivo es mostrar en qué momentos del tiempo el algoritmo detectó actividad muscular significativa, subrayada por una franja azul claro.

<img width="400" height="800" alt="image" src="img4.jpg" /> <br>

La tabla presenta los valores espectrales obtenidos tras aplicar la Transformada Rápida de Fourier (FFT) a cada contracción individual,las frecuencias medias se mantienen alrededor de 40 Hz, lo cual es consistente con una señal EMG emulada estable,La frecuencia mediana también permanece casi constante (≈6 Hz), indicando que el contenido de energía en frecuencia no varía significativamente entre contracciones lo cual indica que esta simulación esta sin presencia de fatiga muscular real.

<img width="400" height="800" alt="image" src="img5.jpg" /> <br>

La gráfica compara los valores de frecuencia media y frecuencia mediana obtenidos para cada contracción de la señal EMG.Cada punto representa una contracción, y las líneas muestran la tendencia de ambos parámetros espectrales a medida que avanza el tiempo o el esfuerzo muscular,Esta gráfica demuestra que la señal EMG emulada presenta comportamiento estable entre contracciones, reflejando un músculo sin fatiga,tal como se espera en una señal generada artificialmente.

<img width="400" height="800" alt="image" src="img6.jpg" /> <br>

Esta gráfica presenta el espectro de frecuencias de una sola contracción muscular obtenida de la señal EMG.
La FFT transforma la señal del dominio del tiempo (amplitud vs. tiempo) al dominio de la frecuencia (potencia vs. frecuencia), permitiendo analizar qué frecuencias predominan en la actividad muscular,el espectro tiene un pico pronunciado en bajas frecuencias, lo que indica que la mayor parte de la energía del músculo se concentra en componentes lentas,la energía disminuye rápidamente a medida que aumenta la frecuencia, lo que es típico en señales EMG, ya que las contracciones musculares contienen más información en bajas frecuencias.
