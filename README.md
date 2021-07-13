# Proyecto4
Proyecto 4: Simulación de modulación 16-QAM para envío de imagenes por un canal ruidoso.

La modulación de amplitud en cuadratura (**QAM**, *Quadrature Amplitude Modulation*) de 16 símbolos (16-QAM) es un tipo de modulación IQ que utiliza más posibles amplitudes de dos ondas portadoras seno y coseno para representar 16 símbolos distintos, pues puede portar y transmitir hasta 4 bits por periodo ($2^4 = 16$).

La codificación para un símbolo b1 b2 b3 b4 en 16-QAM es:

s(t) = A1 cos(2 π fc t) + A2 sen(2 π fc t)

donde

○ A1 = -3    si    b1 b2 = 0 0

○ A1 = -1    si    b1 b2 = 0 1

○ A1 =  1    si    b1 b2 = 1 1

○ A1 =  3    si    b1 b2 = 1 0

y

○ A2 =  3    si    b3 b4 = 0 0

○ A2 =  1    si    b3 b4 = 0 1

○ A2 = -1    si    b3 b4 = 1 1

○ A2 = -3    si    b3 b4 = 1 0

• Para desarrollar la simulación de comunicaciones usando la modulación 16-QAM, se siguieron los siguientes pasos:

   1. Modulador: Se generaron dos señales portadoras tipo seno y coseno. Cada una de estas señales se modificaron para poder llevar dos bits por periodo de cada señal. Se             generaron hasta 4 casos distintos por señal para representar las 4 posibles combinaciones de dos bits en cada una. Las amplitudes se definieron por bit según la convención       descrita arriba.
   
   2. Demodulador: Se usó el mismo criterio de energía para determinar que la señal entrante fuera un bit 1 o 0. La diferencia es que al haber 4 casos posibles para cada onda         portadora, y como el periodo era más largo se obtuvieron energías mucho mayores. Mediante pruebas se descubrió que al representar amplitudes pequeñas de la señal, ya             fuera las amplitudes de 1 o -1, estas se representaban con energías cercanas a 20 y -20 respectivamnete. Asimismo, para las amplitudes grandes 3 o -3 de la señal modulada       se observó energías cercanas a 60 y -60 respectivamente. Así se determinó que un buen criterio para diferenciar amplitudes de 1 de 3 y -1 de -3 fue tomar el valor central       entre 20 y 60 para diferenciar 1 de 3 y -20 y -60 para diferenciar a -1 de -3. Se determinó que estos números eran respectivamente 40 y -40.

Los resultados obtenidos de las señales importantes se muestran a continuación:

<img align='center' src='https://github.com/Ancorva/Proyecto4/blob/main/Se%C3%B1ales.png' width ='800'/>

En orden descendente, la señal roja representa algunos bits de la imagen enviada. La señal verde representa la onda portadora modulada a partir de los bits que se querían enviar. La señal azul representa la señal recibida por el receptor, luego de que la onda portadora de la imagen pasara por un canal ruidoso con SNR = -5 (veáse P$.ipynb para más información). Finalmente, la señal magenta representa la señal demodulada.

La imagen recuperada después de la transmisión se muestra a continuación:


<img align='center' src='https://github.com/Ancorva/Proyecto4/blob/main/Imagenes.png' width ='800'/>

• Para las pruebas de ergodicidad y estacionaridad simplemente se calculó el promedio estadístico y el promedio temporal de senal_Tx y mediante un porcentaje de error se           determinó que tan diferentes eran. Esto se hizo ya que se sabe que si estos valores son iguales, el proceso es ergódico. Como margen de igualdad para determinar si eran         iguales, se usó una tolerancia del 0.01%. En el promedio temporal, se usó en vez de 1/2T, 1/T ya que se hizo la integración desde 0 hasta el final de la simulación (T) y no     desde -T hasta T (en otras palabras, se recorrió en la i ntegral una sola vez T y no 2 veces).

En esta parte, los resutados fueron los siguientes:

Promedio estadístico de senal_Tx: 0.003796864
Promedio temporal de senal_Tx: 0.003796687
Porcentaje de error entre promedios: 0.004658966%

Se debe recordar que un proceso es ergódico si el promedio estadístico es igual al promedio temporal. Dado que es casi imposible que estos dos den igual dadas las limitaciones computacionales, se optó por una tolerancia del 0.01% de margen de error entre señales para definir la igualdad. Bajo estas circunstancia se determinó que senal_Tx es un proceso ergódico.

• Para la densidad espectral de potencia se calculó usando la fórmula respectiva, tomando en cuenta que se representaron dos bits por periodo para hacer los cálculos. Los resultados fueron los siguientes:



<img align='center' src='https://github.com/Ancorva/Proyecto4/blob/main/DensidadEspectral.png' width ='800'/>
