# Proyecto4
Proyecto 4: Simulación de modulación 16-QAM para envío de imagenes por un canal ruidoso.

La modulación de amplitud en cuadratura (**QAM**, *Quadrature Amplitude Modulation*) de 16 símbolos (16-QAM) es un tipo de modulación IQ que utiliza más posibles amplitudes de dos ondas portadoras seno y coseno para representar 16 símbolos distintos, pues puede portar y transmitir hasta 4 bits por periodo ($2^4 = 16$).

La codificación para un símbolo b1 b2 b3 b4 en 16-QAM es:

s(t) = A1 cos(2 π fc t) + A2 sen(2 π fc t)

donde

$$
A_1 = 
\begin{cases}
             -3, &   \text{si} \quad b_1 b_2 = 00 \\
             -1, &   \text{si} \quad b_1 b_2 = 01 \\
             1, &  \text{si}   \quad b_1 b_2 = 11 \\
             3, &  \text{si}   \quad, b_1 b_2 = 10 \\
\end{cases}
$$

y

$$
A_2 = 
\begin{cases}
             3, &   \text{si} \quad b_3 b_4 = 00 \\
             1, &   \text{si} \quad b_3 b_4 = 01 \\
             -1, &  \text{si}   \quad b_3 b_4 = 11 \\
             -3, &  \text{si}   \quad b_3 b_4 = 10 \\
\end{cases}
$$

• Para desarrollar la simulación de comunicaciones usando la modulación 16-QAM, se siguieron los siguientes pasos:

   1. Modulador: Se generaron dos señales portadoras tipo seno y coseno. Cada una de estas señales se modificaron para poder llevar dos bits por periodo de cada señal. Se             generaron hasta 4 casos distintos por señal para representar las 4 posibles combinaciones de dos bits en cada una. Las amplitudes se definieron por bit según la convención       descrita arriba.
   
   2. Demodulador: Se usó el mismo criterio de energía para determinar que la señal entrante fuera un bit 1 o 0. La diferencia es que al haber 4 casos posibles para cada onda         portadora, y como el periodo era más largo se obtuvieron energías mucho mayores. Mediante pruebas se descubrió que al representar amplitudes pequeñas de la señal, ya             fuera las amplitudes de 1 o -1, estas se representaban con energías cercanas a 20 y -20 respectivamnete. Asimismo, para las amplitudes grandes 3 o -3 de la señal modulada       se observó energías cercanas a 60 y -60 respectivamente. Así se determinó que un buen criterio para diferenciar amplitudes de 1 de 3 y -1 de -3 fue tomar el valor central       entre 20 y 60 para diferenciar 1 de 3 y -20 y -60 para diferenciar a -1 de -3. Se determinó que estos números eran respectivamente 40 y -40.

• Para las pruebas de ergodicidad y estacionaridad simplemente se calculó el promedio estadístico y el promedio temporal de senal_Tx y mediante un porcentaje de error se           determinó que tan diferentes eran. Esto se hizo ya que se sabe que si estos valores son iguales, el proceso es ergódico. Como margen de igualdad para determinar si eran         iguales, se usó una tolerancia del 0.01%. En el promedio temporal, se usó en vez de 1/2T, 1/T ya que se hizo la integración desde 0 hasta el final de la simulación (T) y no     desde -T hasta T (en otras palabras, se recorrió en la i ntegral una sola vez T y no 2 veces).

• Para la densidad espectral de potencia se calculó usando la fórmula respectiva, tomando en cuenta que se representaron dos bits por periodo para hacer los cálculos
