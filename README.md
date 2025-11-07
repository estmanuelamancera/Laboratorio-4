# Señal Electromiográfica EMG
## DESCRIPCIÓN 
En este repositorio analizaremos la practica desarollada que se titula **"Señales Electromiograficas"**, se realizo la captura y análisis de señales emuladas y reales, aplicando filtrado, segmentación, análisis espectral (FFT).Ademas se calcularon las frecuencias media y mediana de las contracciones musculares para observar cómo varían durante el esfuerzo para detectar la aparición de la fatiga.
## OBJETIVOS
-Aplicar el filtrado de señales continuas para procesar una señal electromiográfica(EMG).<br>
-Detectar la aparición de fatiga muscular mediante el análisis espectral de contracciones musculares individuales.<br>
-Comparar el comportamiento de una señal emulada y una señal real en términos de frecuencia media y mediana.<br>
-Emplear herramientas computacionales para el procesamiento, segmentación y análisis de señales biomédicas.<br>
# PROCEDIMIENTO
## PARTE A Captura de la señal emulada 
En esta primera parte del laboratorio se genero una señal EMG por medio del generador de señales el cual simuló cinco contracciones musculares voluntarias, representando la actividad eléctrica del músculo durante el esfuerzo,con una frecuencia de muestreo 5000 Hz y una duracion de 10 s. Los datos que se obtuvieron se almacenaron en un archivo de texto (Señal EMG5000.txt), para luego segmentarla por contraccion y calcular la frecuencia media y mediana de cada una, asi se puede observar cómo varía el contenido espectral a lo largo del tiempo.
Esta señal tuvo un analisis por energia,ya que la señal original es altamente ruidosa y oscilatoria, entonces el código calcula la energía local, que es la potencia promedio de la señal en una ventana de tiempo,asi los valores negativos desaparecen, ya que se elevan al cuadrado,las zonas con mayor actividad muscular (mayor amplitud EMG) se reflejan como picos de energía y se pueden detectar fácilmente las contracciones.<br>


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

# PARTE B- Procesamiento y análisis de señal EMG real

## Procedimiento
En esta segunda parte del laboratorio se trabajo el procesamiento de una señal electromiográfica (EMG) real, registrada sobre un grupo muscular durante la ejecución de contracciones voluntarias en este caso lo trabajamos especificamente en el antebrazo. Buscando analizar la actividad eléctrica de este y observando el comportamiento del espectro de frecuencias durante el esfuerzo y la posible aparición de fatiga muscular.<br>
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/a011ac17-75be-4cf7-b7b8-3fe72f4c350e" /><br>

Para ello, se aplicó un filtro pasa banda de 20–450 Hz que permitió eliminar artefactos de movimiento y ruido eléctrico, conservando solo las frecuencias fisiológicamente relevantes de la señal EMG. Posteriormente, la señal se segmentó en contracciones individuales, se calculó para cada una la frecuencia media y mediana mediante la Transformada de Fourier, y se graficaron los resultados para evaluar la tendencia del espectro.<br>

Este análisis permite relacionar los cambios en el contenido frecuencial con los procesos fisiológicos de la fatiga, ya que una disminución progresiva de las frecuencias indica una menor velocidad de conducción de las fibras musculares y una reducción de la frecuencia de activación de las unidades motoras.<br>

## Diagrama
<img width="800" height="1100" alt="image" src="https://github.com/user-attachments/assets/2eb8bcd2-3333-41fa-aa60-70d78ddb5ff1" /><br>

## Codigo 
```
# Señal original
axs[0].plot(t, senal, color='gray', linewidth=0.8)
axs[0].set_title('Señal EMG Original Completa')
axs[0].set_xlabel('Tiempo [s]')
axs[0].set_ylabel('Amplitud [mV]')
axs[0].grid(True)
```
<img width="1662" height="485" alt="image" src="https://github.com/user-attachments/assets/70406ade-afc6-44f3-9523-0842d927aad7" /><br>

Este bloque del código se presenta una comparación entre la señal electromiográfica (EMG) original y la versión filtrada mediante un filtro pasa banda con un rango de 20 a 450 Hz. Este procedimiento de filtrado permite suprimir el ruido de baja frecuencia —producido por movimientos o variaciones en la línea base— y reducir las componentes de alta frecuencia que no corresponden a la actividad muscular. De esta manera, la señal filtrada (en azul) adquiere una apariencia más clara y fiel a la actividad eléctrica del músculo, lo que facilita su análisis en etapas posteriores, como la identificación de contracciones o el estudio del espectro de frecuencias.<br>
### Aplicación del filtro 
```
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def aplicar_filtro(senal, lowcut=20, highcut=450, fs=1000, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, senal)

senal_filtrada = aplicar_filtro(senal, 20, 450, Fs)

# Original vs Filtrada (superpuestas)
axs[1].plot(t, senal, color='gray', linewidth=0.8, alpha=0.6, label='Original')
axs[1].plot(t, senal_filtrada, color='pink', linewidth=1.2, label='Filtrada (20–450 Hz)')
axs[1].set_title('Comparación: Señal EMG Original vs Filtrada')
axs[1].set_xlabel('Tiempo [s]')
axs[1].set_ylabel('Amplitud [mV]')
axs[1].legend(loc='upper right')
axs[1].grid(True)
```
<img width="1945" height="828" alt="image" src="https://github.com/user-attachments/assets/0eb9094d-6f8f-45af-b204-6a1cd401e9e4" /><br>

## GRÁFICA FILTRADA VS ORIGINAL
La figura presenta un acercamiento a los primeros tres segundos de la señal EMG, donde se compara la forma de onda original (en gris) con la señal filtrada mediante un filtro pasa banda de 20 a 450 Hz (en rosada). Este zoom permite observar con mayor claridad cómo el proceso de filtrado suprime las variaciones lentas y el ruido de baja frecuencia, destacando las oscilaciones rápidas asociadas con la actividad eléctrica real del músculo durante las contracciones. Gracias a este procedimiento, la señal obtenida mantiene las componentes fisiológicamente relevantes, lo que facilita el análisis posterior de la dinámica muscular y la frecuencia de contracción.
### Código 
```
# --- Gráfica comparativa: señal original y filtrada superpuestas ---
plt.figure(figsize=(14,6))

# Señal original en gris
plt.plot(t, senal, color='gray', linewidth=0.8, alpha=0.6, label='Señal original')

# Señal filtrada en rosado más destacada
plt.plot(t, senal_filtrada, color='#ff69b4', linewidth=1.2, label='Señal filtrada (20–450 Hz)')

plt.title('Comparación: Señal EMG Original vs Filtrada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# --- Gráfico con zoom en una región (por ejemplo, primeros 3 segundos) ---
inicio_zoom = 0       # segundo inicial
fin_zoom = 3          # segundo final
muestras_zoom = slice(int(inicio_zoom * Fs), int(fin_zoom * Fs))

plt.figure(figsize=(14,5))
plt.plot(t[muestras_zoom], senal[muestras_zoom], color='gray', linewidth=0.8, alpha=0.6, label='Original')
plt.plot(t[muestras_zoom], senal_filtrada[muestras_zoom], color='#ff69b4', linewidth=1.2, label='Filtrada (20–450 Hz)')
plt.title(f'Zoom de la Señal EMG (entre {inicio_zoom}s y {fin_zoom}s)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

```
<img width="1957" height="670" alt="image" src="https://github.com/user-attachments/assets/0cc1cb49-369b-4004-8e8a-b0a2cc350eac" /><br>

## SEGMENTACIÓN

En esta fase del procesamiento, la señal EMG previamente filtrada se dividió en 83 contracciones musculares individuales. Esta segmentación permite analizar de manera independiente cada evento de contracción, facilitando la evaluación de la forma de onda, la duración y las variaciones de amplitud. Como se muestra en la figura, cada segmento corresponde a una contracción distinta del músculo, evidenciando cómo la actividad eléctrica cambia a lo largo del tiempo. Este procedimiento resulta fundamental para investigaciones relacionadas con la fatiga muscular, la activación motora y el análisis de patrones de esfuerzo.<br>

```
num_contracciones = 83
L = len(senal_filtrada)
segmentos = np.array_split(senal_filtrada, num_contracciones)
```

<img width="1972" height="1107" alt="image" src="https://github.com/user-attachments/assets/f8bc4dfb-1ddd-4771-befc-4e9676e99155" /><br>

## CÁLCULO DE MEDIA Y MEDIANA (GRÁFICA)
En esta etapa se calcularon la frecuencia media y la frecuencia mediana correspondientes a cada una de las 83 contracciones musculares segmentadas. Estos parámetros espectrales permiten analizar cómo se distribuye la energía de la señal EMG en el dominio de la frecuencia. La frecuencia media representa el promedio ponderado de las componentes frecuenciales, mientras que la frecuencia mediana señala el punto que divide el espectro de potencia en dos mitades iguales. En la gráfica se aprecia la evolución de ambas medidas a lo largo de las contracciones, evidenciando variaciones que pueden relacionarse con cambios en la activación muscular o con la aparición de fatiga durante el registro.<br>
```
def calcular_frecuencias(segmento, fs):
    f, Pxx = welch(segmento, fs=fs, nperseg=1024)
    Pxx = Pxx / np.sum(Pxx)  # Normalizar el espectro de potencia
    freq_media = np.sum(f * Pxx)
    cum_Pxx = np.cumsum(Pxx)
    freq_mediana = f[np.where(cum_Pxx >= 0.5)[0][0]]
    return freq_media, freq_mediana

frecuencia_media = []
frecuencia_mediana = []

for seg in segmentos:
    f_mean, f_median = calcular_frecuencias(seg, Fs)
    frecuencia_media.append(f_mean)
    frecuencia_mediana.append(f_median)

# %% --- Graficar la evolución de las frecuencias ---
plt.figure(figsize=(10,5))
plt.plot(frecuencia_media, 'o-', label='Frecuencia Media')
plt.plot(frecuencia_mediana, 's-', label='Frecuencia Mediana')
plt.title('Evolución de la Frecuencia Media y Mediana (83 Contracciones)')
plt.xlabel('Número de Contracción')
plt.ylabel('Frecuencia [Hz]')
plt.legend()
plt.grid()
plt.show()

tabla_resultados = pd.DataFrame({
    'Contracción': np.arange(1, num_contracciones + 1),
    'Frecuencia Media (Hz)': np.round(frecuencia_media, 2),
    'Frecuencia Mediana (Hz)': np.round(frecuencia_mediana, 2)
})

print(tabla_resultados)
tabla_resultados.to_csv('/content/drive/MyDrive/frecuencias_EMG.csv', index=False)
print("\nArchivo 'frecuencias_EMG.csv' guardado en tu Drive con los resultados.")
```
<img width="1622" height="847" alt="image" src="https://github.com/user-attachments/assets/fe0cd771-377d-4cc2-bf8a-c5ceae17c20a" /><br>

<img width="1035" height="428" alt="image" src="https://github.com/user-attachments/assets/c57288b4-a995-4186-90bf-2dae5a0f2e3f" /><br>

La gráfica de evolución de la frecuencia media y mediana a lo largo de las 84 contracciones musculares muestra cómo varía la distribución espectral de la señal EMG durante el registro. En las primeras contracciones, ambas frecuencias se mantienen relativamente estables entre 70 y 100 Hz, mientras que hacia la mitad del registro se notan fluctuaciones y pequeñas caídas que podrían estar relacionadas con el inicio de la fatiga muscular o con cambios en la fuerza de las contracciones. En las últimas contracciones, se observa un incremento marcado en ambas frecuencias, lo que sugiere un mayor esfuerzo o reclutamiento de más fibras musculares. En general, los resultados reflejan cómo la actividad eléctrica del músculo cambia con el tiempo y el nivel de esfuerzo.<br>


# PARTE C  Análisis espectral mediante FFT
En esta parte del laboratorio se implementó la Transformada Rápida de Fourier (FFT) sobre las contracciones obtenidas de la señal EMG real, con el propósito de analizar su comportamiento en el dominio de la frecuencia. Este procedimiento permitió observar cómo varía el contenido espectral a lo largo del esfuerzo muscular y detectar posibles indicios de fatiga a partir de la reducción en las componentes de alta frecuencia. Además, se compararon los espectros de las primeras y últimas contracciones, identificando el desplazamiento del pico espectral asociado con el esfuerzo sostenido. Finalmente, los resultados obtenidos se emplearon para evaluar la utilidad del análisis espectral como herramienta diagnóstica en estudios de electromiografía y desempeño fisiológico muscular.
## Diagrama
<img width="1024" height="768" alt="Diagrama de Flujo Árbol de decisiones Sencillo Verde (4)" src="https://github.com/user-attachments/assets/0046a1ff-1253-4b2c-90b0-27faa4f538af" /><br>
## PROCEDIMIENTO <br>
En esta parte del código se aplica un filtro digital pasa banda Butterworth de cuarto orden entre 20 y 450 Hz para limpiar la señal EMG y conservar únicamente las frecuencias relacionadas con la actividad muscular.Las frecuencias inferiores a 20 Hz suelen corresponder a movimientos del cuerpo o desplazamientos de la línea base, mientras que las superiores a 450 Hz se asocian a ruido eléctrico. La implementación con butter() y filtfilt() permite realizar un filtrado sin desfase, evitando retrasos en la señal y preservando su forma original. El resultado es una señal más estable y precisa, que refleja de manera fiel la actividad eléctrica del músculo, siendo fundamental para las etapas posteriores de detección y análisis espectral.
### Código
```
# Filtrado pasa banda (20–450 Hz)
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

b, a = butter_bandpass(20, 450, Fs)
senal_filtrada = filtfilt(b, a, v)
```
## Transformada rápida de Fourier (FFT) 

Se utilizó la Transformada Rápida de Fourier (FFT) para analizar la señal EMG en el dominio de la frecuencia y observar cómo se distribuye su energía en diferentes bandas. Este método permite identificar las frecuencias dominantes que aparecen durante la contracción muscular y evaluar cambios entre el inicio y el final del experimento. En el código, se convierte la magnitud de la FFT a decibelios (dB) y se representa en escala logarítmica para resaltar mejor las variaciones espectrales. Esto es importante porque un desplazamiento del contenido frecuencial hacia frecuencias más bajas puede indicar fatiga muscular, ya que el músculo genera señales eléctricas más lentas cuando se cansa.
### Código
```
def graficar_fft(segmento, fs, titulo, color):
    N = len(segmento)
    f = np.fft.rfftfreq(N, 1/fs)
    fft_mag = np.abs(np.fft.rfft(segmento))
    fft_mag_db = 20 * np.log10(fft_mag + 1e-12)
    plt.semilogx(f, fft_mag_db, color=color, linewidth=1.5)
    plt.title(titulo)
    plt.xlabel("Frecuencia [Hz] (log)")
    plt.ylabel("Magnitud [dB]")
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.xlim(10, 500)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
graficar_fft(seg4, Fs, "FFT Contracción 4 (Inicio)", color='blue')
plt.subplot(1,2,2)
graficar_fft(seg83, Fs, "FFT Contracción 83 (Final)", color='fuchsia')
plt.tight_layout()
plt.show()
```
<img width="2111" height="899" alt="image" src="https://github.com/user-attachments/assets/e7ef7f60-9c45-4df7-a751-461321afaea3" /><br>

## Espectro de amplitud 
En esta parte del código se realizó el cálculo del espectro de amplitud de las contracciones seleccionadas (la 4 y la 83) utilizando la Transformada Rápida de Fourier (FFT) y representándolo en una escala logarítmica de frecuencia. Este análisis permite observar cómo se distribuye la energía de la señal EMG en diferentes frecuencias y comparar la actividad muscular al inicio y al final del registro. Al graficar ambas contracciones —una en color azul y la otra en fucsia— se puede visualizar si existe un desplazamiento del contenido espectral hacia frecuencias más bajas, lo cual es un indicador típico de fatiga muscular.
### Código
```
# ESPECTRO DE AMPLITUD (ESCALA LOGARÍTMICA)

plt.figure(figsize=(12,5))

# Contracción 4
N1 = len(seg_inicial)
fft_vals1 = np.fft.fft(seg_inicial)
fft_mag1 = np.abs(fft_vals1) / N1
fft_mag1 = fft_mag1[:N1//2] * 2
freqs1 = np.fft.fftfreq(N1, 1/Fs)[:N1//2]

plt.plot(freqs1, fft_mag1, color='blue', linewidth=1.5, label='Contracción 4 (Inicio)')

# Contracción 83
N2 = len(seg_final)
fft_vals2 = np.fft.fft(seg_final)
fft_mag2 = np.abs(fft_vals2) / N2
fft_mag2 = fft_mag2[:N2//2] * 2
freqs2 = np.fft.fftfreq(N2, 1/Fs)[:N2//2]

plt.plot(freqs2, fft_mag2, color='fuchsia', linewidth=1.5, label='Contracción 83 (Final)')

# Configuración del gráfico
plt.xscale('log')
plt.xlim(10, 500)
plt.title('Espectro de Amplitud (escala logarítmica)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
```
<img width="2104" height="852" alt="image" src="https://github.com/user-attachments/assets/cda1d12b-2c00-4eef-97dc-f92e2be05a53" /><br>

## Espectro de amplitud (Welch) 
En esta parte del código se aplicó el método de Welch para obtener el espectro de amplitud promedio de las contracciones seleccionadas. Este método mejora la estimación del contenido en frecuencia al dividir la señal en segmentos, calcular la FFT de cada uno y promediar los resultados, reduciendo así el ruido y las variaciones instantáneas. Al representar los espectros en escala logarítmica, se facilita la comparación entre la contracción 4 (inicio) y la contracción 83 (final). Si el pico de energía se desplaza hacia frecuencias más bajas en la contracción final, esto sugiere una disminución en la frecuencia media y, por tanto, la presencia de fatiga muscular.
### Código
```
# ESPECTRO DE AMPLITUD (WELCH - logarítmico)

nperseg = 256
f3, Pxx1 = welch(seg4, fs=Fs, nperseg=nperseg)
f4, Pxx2 = welch(seg83, fs=Fs, nperseg=nperseg)

plt.figure(figsize=(12,5))
plt.semilogx(f3, 10*np.log10(Pxx1 + 1e-12), color='blue', label='Contracción 4 (Inicio)', linewidth=1.8)
plt.semilogx(f4, 10*np.log10(Pxx2 + 1e-12), color='fuchsia', label='Contracción 83 (Final)', linewidth=1.8)
plt.xlim(10, 500)
plt.xlabel("Frecuencia [Hz] (log)")
plt.ylabel("Densidad espectral [dB]")
plt.title("Espectro de Amplitud (Welch - Escala Logarítmica)")
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.tight_layout()
plt.show()
```
<img width="2132" height="866" alt="image" src="https://github.com/user-attachments/assets/a2ffa716-399f-4f3b-9b5b-e00b0e52435f" /><br>

## Calculos del pico espectral y gráfica
En esta parte del código se calcularon dos indicadores clave del análisis espectral: el pico espectral y el centroide de frecuencia. El pico espectral representa la frecuencia donde la energía del músculo es máxima, mientras que el centroide indica la frecuencia promedio ponderada por la potencia del espectro. Estos valores permiten cuantificar los cambios en el contenido frecuencial entre contracciones. Posteriormente, se realizó una gráfica comparativa (en este caso de puntos o barras) para visualizar la diferencia entre las contracciones 4 y 83. Una disminución en ambas frecuencias refleja un desplazamiento del espectro hacia componentes de menor frecuencia, lo cual es un indicador característico de la fatiga muscular.
### Código
```
# CÁLCULO DEL PICO ESPECTRAL Y CENTROIDE

def calcular_pico_centroide(f, Pxx):
    idx_max = np.argmax(Pxx)
    f_pico = f[idx_max]
    centroide = np.sum(f * Pxx) / np.sum(Pxx)
    return f_pico, centroide

f_pico1, centroide1 = calcular_pico_centroide(f3, Pxx1)
f_pico2, centroide2 = calcular_pico_centroide(f4, Pxx2)

print("\n--- RESULTADOS ---")
print(f"Contracción 4 → Pico espectral: {f_pico1:.2f} Hz | Centroide: {centroide1:.2f} Hz")
print(f"Contracción 83 → Pico espectral: {f_pico2:.2f} Hz | Centroide: {centroide2:.2f} Hz")
```
<img width="1852" height="1002" alt="image" src="https://github.com/user-attachments/assets/c81b5382-afe9-41e1-9647-605827edba4c" /><br>
## Conclusiones sobre el uso del análisis espectral en electromiografía

El análisis espectral es una herramienta fundamental en electromiografía, ya que permite estudiar la distribución de frecuencias de la señal muscular y detectar cambios asociados al estado fisiológico del músculo. A través de métodos como la Transformada Rápida de Fourier (FFT) y el método de Welch, es posible observar cómo la energía de la señal EMG se concentra en diferentes rangos de frecuencia dependiendo del nivel de contracción y la presencia de fatiga.
Cuando el músculo se fatiga, las fibras conducen los impulsos eléctricos más lentamente, lo que genera un desplazamiento del espectro hacia frecuencias más bajas y una disminución del pico y del centroide espectral.
Por lo tanto, el análisis espectral no solo ofrece una representación cuantitativa del comportamiento muscular, sino que también se convierte en una herramienta diagnóstica útil para evaluar el rendimiento, la fatiga y posibles disfunciones neuromusculares, aportando información objetiva para la valoración clínica y el seguimiento en fisioterapia, rehabilitación y ergonomía.
