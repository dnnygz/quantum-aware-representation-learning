# Informe de ejecución A/B — 2026-10-28

## Evidencia realmente ejecutada

Suite completa: **88 pruebas aprobadas**, incluidas 11 nuevas de Fase 2.
Se verificaron aislamiento de particiones, fit sólo en training, transformaciones
angulares, formas, gradientes de inputs/encoder/VQC, reproducibilidad exacta de
entrenamiento sintético y persistencia de los cinco modelos/controles.

Smoke real: **fold 0, training 668, validation 167 (129 ER+, 38 ER−), semilla 1729,
dos épocas** para las redes. Logistic Regression C=1 usa convergencia propia.
Este resultado es exclusivamente una comprobación de funcionamiento.
No se ejecutaron tres semillas, cinco folds, ni locked test. C0 no está implementado.
Se repitió el smoke por verificación de metadatos y las métricas coincidieron.

| Modelo | ROC-AUC | AP ER+ | AP ER− | Balanced accuracy | Entrenamiento (s) |
|---|---:|---:|---:|---:|---:|
| A | 0.7442 | 0.9085 | 0.5297 | 0.6167 | 0.1756 |
| A_classical | 0.9598 | 0.9860 | 0.8874 | 0.8568 | 0.0019 |
| B | 0.9535 | 0.9813 | 0.8676 | 0.8963 | 0.1324 |
| B_frozen_probe | 0.9535 | 0.9823 | 0.8447 | 0.9055 | 0.0013 |
| B_classical | 0.9737 | 0.9925 | 0.8886 | 0.9094 | 0.0100 |

Fuente: `runs/phase2/smoke_verified_1729/comparison.csv`. Copia pequeña sin IDs de
pacientes: `docs/phase2_ab_smoke_metrics.csv`. Los tiempos dependen del arranque,
cachés y entorno; no son un benchmark. El diagnóstico congelado sólo cuenta el
ajuste de la nueva cabeza; debe añadirse entrenamiento de B para costo total.

## Hallazgos observados

- El pipeline produce predicciones finitas y métricas para las cinco variantes.
- B obtiene mayor ROC-AUC que A en esta ejecución de dos épocas. Este contraste
  no establece una mejora estable del método: difieren representación, encoder
  y trayectoria de inicialización; falta el presupuesto piloto completo.
- La cabeza LR nueva sobre z congelado iguala el ROC-AUC observado de B+VQC.
  La representación aprendida por B permite clasificar con una cabeza clásica
  en esta partición. No demuestra dominancia clásica general ni ventaja cuántica.
- B clásico también clasifica eficazmente con el mismo encoder pequeño.
- No aparecieron NaN/inf ni gradientes desconectados. Las normas no garantizan
  por sí mismas utilidad científica de los parámetros cuánticos.

En B, la norma media de gradientes del encoder fue 0.8144 → 0.6522; la del VQC 0.2153 → 0.1501.
BCE train/validation al terminar: A 0.5587/0.5251, B 0.2748/0.2243.
No se observa una brecha adversa train/validation en estas dos épocas; es
insuficiente para descartar sobreajuste. Las pérdidas de training son promedios
online de minibatches y no evaluaciones del checkpoint final sobre training.

Cada VQC procesó 1837 estados de pacientes en 45 llamadas QNode (entrenamiento,
validation por época y predicción final). Backprop no ejecuta parameter-shift.
Tiempo total de la ejecución: 18.87 s; pico RSS 4.21 GiB.
El preprocesamiento se registra por separado en la tabla CSV. Este smoke no
permite estimar un sobrecosto estable de B: predominan preparación y variabilidad
de ejecución, y los tiempos de entrenamiento son inferiores a un segundo.

## Limitaciones y siguiente protocolo

Ejecutar primero el presupuesto congelado de 50 épocas y tres semillas sobre
fold 0; después cinco folds usando la misma configuración. Comparar diferencias
pareadas A/B y sus controles, dispersión entre semillas, curvas, gradientes y
costos incluyendo preprocesamiento y ajuste del probe. No ajustar decisiones
según locked test. No usar esta tabla para claims confirmatorios, originalidad,
ventaja cuántica o superioridad sobre el estado del arte.

Antes de C0, fijar el protocolo y comprobar estabilidad de A/B. C0 deberá partir
de una reproducción explícita de la formulación geométrica de NQE y documentar
su contraste con QPMeL; no introducir regularización nueva hasta esa revisión.
La preparación actual es infraestructura para esa comparación, no contribución nueva.
