# Experimentos A y B: protocolo de development

## Auditoría y alcance

Auditoría 2026-10-28: Git inicialmente limpio. Fase 1 usa `src/data/`, pytest,
scikit-learn y un entorno uv/Python 3.12. No había implementación de PCA, VQC o
Fase 2. Se encontraron counts canónicos (1044 × 60660), etiquetas ER+=1/ER−=0 y
`data/processed/splits/phase1c_inner_cv.json`: 835 development, 209 locked test,
cinco folds y semilla 1729. No falta ningún artefacto indispensable local.
Estos datos no están versionados y deben obtenerse con el procedimiento de Fase 1
al reproducir en otro equipo. No se modifican los pipelines ni resultados de Fase 1.

La carga nueva aplica un filtro Arrow por IDs de development antes de materializar
counts y etiquetas. Sólo se consultan los IDs locked para verificar aislamiento;
no se transforman sus observaciones ni se evalúan sus etiquetas. Los hashes de
archivos completos son exclusivamente trazabilidad, sin calcular estadísticas.

## Arquitecturas congeladas para el piloto

Todas las transformaciones se ajustan desde los counts de training de cada fold:
filtro CPM, referencia TMM congelada, log2-CPM, eliminación de varianza exactamente
cero y StandardScaler génico, reutilizando `_build_preprocessor` de Fase 1.
No se reutiliza una matriz normalizada sobre la cohorte completa.

- A: PCA aleatorizado k=4 → StandardScaler de scores → ángulos → VQC.
- B: PCA aleatorizado k=32 → StandardScaler de scores → Linear(32,8), Tanh,
  Linear(8,4) → ángulos → el mismo VQC. El encoder tiene 300 parámetros;
  AdamW aplica weight decay 0.001. No se entrena un MLP sobre todos los genes.
- A clásico: los mismos cuatro scores estandarizados → LogisticRegression
  C=1, intercepto, sin ponderación de clases, máximo 2000 iteraciones.
- B clásico: exactamente la representación intermedia y arquitectura de encoder
  de B, entrenados desde cero → Linear(4,1), Sigmoid.
- Diagnóstico B congelado: encoder del último epoch de B en modo eval, sin
  gradientes; LogisticRegression C=1 nueva se ajusta sólo en z de training y se
  evalúa en z de validation. No se seleccionan encoder, C o umbral con validation.
  Su checkpoint referencia el encoder y la representación de B.

PCA tiene semilla fija 1729 independiente de la inicialización de redes. Se guarda
la varianza explicada, estado completo de transformaciones e IDs de ajuste.
El escalado angular es `pi*tanh(z/2)` en [−π,π]. En A, la media y escala de z
provienen sólo de training; en B se aprende z conjuntamente y la función angular
es fija, diferenciable y no requiere estadísticas de validation. Los controles
clásicos usan los scores/z antes de la función angular; esta diferencia de cabeza
forma parte de la comparación y no implica equivalencia de capacidad.

Circuito único: cuatro qubits, una codificación RY de los cuatro ángulos, dos
capas RY/RZ por qubit y CNOT 0→1, 1→2, 2→3 en cada capa. Se mide ⟨Z₃⟩.
`p(ER+) = clip((1+⟨Z₃⟩)/2, 1e-7, 1-1e-7)`, BCE no ponderada, umbral fijo 0.5.
No hay calibración entrenable adicional. Simulación CPU float64, `default.qubit`,
shots=None, sin ruido, diferenciación `interface=torch, diff_method=backprop`,
según la [documentación oficial](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html).

El VQC tiene 16 parámetros; la cabeza lineal tiene 5. El recuento nominal no
implica igual capacidad: por ejemplo, algunas rotaciones pueden ser redundantes
respecto al observable. El diagnóstico LR usa un optimizador y presupuesto
diferentes. Su tiempo excluye el costo previo del encoder, que debe sumarse al
comparar costos totales. B y B clásico tienen el mismo encoder (300 parámetros),
épocas, tamaño de minibatch, lr y regularización; no necesariamente idéntica
trayectoria de optimización. A/B comparten configuración de VQC pero sus sorteos
de inicialización pueden diferir porque B inicializa primero el encoder.

## Ejecución

Desde la raíz, con Python 3.12 (extra quantum requiere Python >=3.11):

```bash
uv sync --extra dev --extra quantum --locked
uv run --extra dev --extra quantum python -m pytest -q
uv run --extra quantum python -m src.phase2.run --smoke --output runs/phase2/smoke_nuevo
uv run --extra quantum python -m src.phase2.run --config configs/phase2_ab.yaml --output runs/phase2/cv_ab
```

Configuraciones independientes: `configs/phase2_a.yaml`, `configs/phase2_b.yaml`;
ambas usan la misma infraestructura y ejecutan sus controles. Cada salida debe
ser nueva: nunca se sobreescriben ejecuciones. No hay comando para locked test.

`--smoke` fuerza fold 0, semilla 1729 y dos épocas, conservando TODOS los pacientes
del fold. Para múltiples semillas en una partición, copiar el YAML y usar
`folds: [0]`, `seeds: [1729,1730,1731]`. El protocolo completo usa cinco folds,
tres semillas y 50 épocas. Las redes tienen el mismo presupuesto de épocas y
pasos de optimización (21 minibatches/época en fold 0). Se evalúa el último epoch;
no hay early stopping, búsqueda de hiperparámetros ni selección exploratoria.
Se registra validation cada epoch únicamente para diagnóstico.

## Artefactos y métricas

Cada carpeta de ejecución contiene config expandida, versiones, entorno completo, copia del código de Fase 2, commit y patch
de cambios rastreados, hashes de datos/splits, tabla `comparison.csv`, y por
fold/experimento el preprocesador/PCA persistido. Por modelo/semilla: checkpoint,
métricas JSON, predicciones de validation CSV y curvas (redes). Las curvas guardan
BCE train/validation y media de norma L2 de gradientes por minibatch para encoder
y cabeza; el encoder identidad de A tiene norma cero por definición.

ROC-AUC, AP ER+ y AP ER−, balanced accuracy, accuracy, log-loss y matriz de confusión
ordenada [ER−,ER+]. Los segundos por modelo cubren entrenamiento con evaluación
por época; el preprocesamiento se registra aparte. Las llamadas QNode cuentan
batches, y `circuit_sample_evaluations` cuenta estados por paciente incluidos
train y validaciones. Backprop reutiliza simulación diferenciable: estos números
no equivalen a ejecuciones de hardware ni a evaluaciones parameter-shift.
`resources.json` contiene tiempo total y pico RSS del proceso con unidades.

Los checkpoints se guardan bajo `runs/phase2/`, ignorado por Git. Los pickle y
state_dict permiten restaurar con `Representation`, `Model(inputs, encoder=...)`
y las versiones registradas; cargar únicamente artefactos propios confiables.
No se guarda una copia de la matriz de expresión. El estado PCA puede ocupar
varios MB y se comparte entre semillas. La copia `source/` y sus hashes conservan también el código de Fase 2 aún no
añadido a Git; el preprocesamiento de Fase 1 se referencia mediante el commit.

## Niveles de evidencia

1. Pruebas sintéticas: integridad de software, no evidencia biomédica.
2. Smoke real de una partición y dos épocas: funcionamiento, no estimación final.
3. Piloto de una partición y múltiples semillas: estabilidad preliminar.
4. Cinco folds y múltiples semillas: evaluación de development. Comparar de
   forma pareada por fold/semilla; semillas y folds no son réplicas independientes.
5. Confirmatorios futuros: protocolo preregistrado y eventual locked test sólo
   cuando esté autorizado científicamente. Ninguno de estos pasos demuestra ventaja cuántica.

Ver `docs/phase2_ab_results.md` para lo realmente ejecutado y siguientes pasos.
C0 no está implementado; NQE/QPMeL quedan fuera del alcance actual.
