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
