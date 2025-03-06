# PINN-BGK Solver

Este proyecto implementa un método de redes neuronales informadas por la física (PINN) para resolver la ecuación de Burgers en el contexto de Optimización y Control en Redes Distribuidas.

## Descripción

El paquete **PINN-BGK** utiliza PyTorch para entrenar un modelo basado en una red MLP, integrando técnicas de early stopping, visualización de métricas y guardado de resultados (modelo, estadísticas y gráficos). La configuración se centraliza en un archivo YAML para facilitar su modificación sin alterar el código fuente.

## Instalación

1. **Clonar el repositorio** a tu máquina.
2. **Acceder a la carpeta del proyecto**:
   ```bash
   cd "D:/Personal/Area de Trabajo/Universidad/Matriculas/Novena/Optimización y Control en Redes Distribuidas/Proyecto Final/code"
   ```
3. **Instalar el proyecto en modo editable**:
   ```bash
   pip install -e .
   ```

## Requerimientos

Consulta el archivo [requirements.txt](requirements.txt) para conocer las dependencias principales, que incluyen:
- PyTorch, torchvision y torchaudio
- NumPy, SciPy, scikit-learn, pandas
- Matplotlib, seaborn
- PyYAML, tqdm
- y otras herramientas de desarrollo y calidad de código

## Modos de Operación

El framework soporta tres modos principales de operación, todos configurables a través del archivo YAML:

### 1. Entrenamiento del Modelo

```bash
PINN-BGK --mode train
```

Este modo entrena un modelo PINN utilizando los hiperparámetros definidos en la sección `model` y `training` del archivo de configuración. El proceso incluye:

- Carga de datos de entrenamiento
- Inicialización del modelo con la arquitectura especificada
- Entrenamiento con early stopping
- Generación de gráficos de métricas (pérdida, precisión)
- Visualización de la solución predicha vs. real
- Guardado del modelo entrenado y estadísticas

### 2. Generación de Datos Sintéticos

```bash
PINN-BGK --mode generate
```

Este modo genera datos sintéticos para diferentes ecuaciones y escenarios físicos:

- **Ecuación de Burgers 1D**: Genera la solución analítica
- **Flujo de Kovasznay**: Implementa la solución para el flujo estacionario de Navier-Stokes
- **Vórtice de Taylor-Green**: Simula un vórtice que decae con el tiempo
- **Flujo en Cavidad**: Implementa la simulación del flujo en una cavidad con tapa móvil

Cada generación produce archivos .mat con datos y un informe JSON detallado con metadatos.

### 3. Evaluación de Modelos

```bash
PINN-BGK --mode evaluate
```

Este modo evalúa un modelo previamente entrenado con datos específicos:

- Carga el modelo desde la ruta especificada
- Evalúa el modelo con los datos proporcionados
- Calcula métricas (MAE, MSE, RMSE, R2)
- Genera visualizaciones comparativas
- Produce un mapa de calor de errores
- Guarda los resultados detallados en formato JSON

## Uso de Configuración Personalizada

Puedes proporcionar una configuración personalizada mediante:

```bash
PINN-BGK --mode [modo] --config ruta/a/configuracion.yaml
```

## Estructura del Proyecto

```
Proyecto Final/
├── code/
│   ├── config/
│   │   └── config.yaml         # Configuración centralizada para todos los modos
│   ├── requirements.txt        # Requerimientos de Python
│   ├── README.md               # Este archivo
│   ├── setup.py                # Configuración del paquete e entry point
│   ├── src/
│   │   ├── __init__.py         # Permite tratar src como un paquete
│   │   ├── main.py             # Entry point del proyecto con modos de operación
│   │   ├── data_handlers/      # Manejo de datos y generación
│   │   │   ├── data_loader.py  # Cargador de datos
│   │   │   ├── fluid_data_generator.py # Generador de datos de fluidos
│   │   │   └── generators/     # Implementaciones específicas de generadores
│   │   ├── model/
│   │   │   └── train.py        # Clase Trainer para entrenamiento
│   │   ├── utils/
│   │   │   ├── data_loader.py  # Utilidades para carga de datos
│   │   │   ├── visualization.py # Funciones para visualización
│   │   │   └── device_utils.py # Utilidades para dispositivos (CPU/GPU)
│   │   └── structure_model/
│   │       └── pinn_structure_v1.py # Definición del modelo PINN
└── data/
    ├── training/               # Datos para entrenamiento
    │   └── burgers_shock_mu_01_pi.mat
    ├── synthetic/              # Datos generados sintéticamente
    └── evaluation/             # Resultados de evaluación
```

## Archivo de Configuración YAML

El archivo `config.yaml` se organiza en secciones para cada modo de operación:

### Modelo y Entrenamiento

```yaml
model:
  type: "mlp"
  layers: [2, 50, 50, 50, 50, 1]
  activation_function: "Tanh"
  learning_rate: 0.001

training:
  epochs: 100
  batch_size: 32
  epochs_no_improve: 20
  min_loss_improvement: 1e-5
  seed: 42
  memory_limit_gb: 4
```

### Parámetros Físicos

```yaml
physics:
  nu: 0.01
  boundary_type: "dirichlet"
  physics_type: "burgers"
```

### Generación de Datos

```yaml
data_generation:
  type: "burgers"  # Opciones: burgers, kovasznay, taylor_green, lid_driven_cavity
  spatial_points: 256
  time_points: 100
  # Parámetros específicos
  Re: 40               # Para Kovasznay
  U0: 1.0              # Para flujos
  resolution: 32       # Para mallas
  Nx: 32               # Para mallas 2D
  Ny: 32               # Para mallas 2D
  T: 2.0               # Tiempo total
  num_steps: 50        # Pasos temporales
```

### Evaluación de Modelos

```yaml
evaluation:
  model_path: "output/Model_mlp_1_0.001_42/model/weights_model.pth"
  data_path: "data/training/burgers_shock_mu_01_pi.mat"
  output_folder: "evaluation_results"
  metrics: ["MAE", "MSE", "RMSE", "R2"]
  visualize: true
  compare_with_analytical: false
```

## Generación de Datos Sintéticos y Analíticos

El framework permite generar datos para diversos escenarios de fluidos tanto de forma **sintética** (utilizando generadores analíticos que no requieren dependencias externas como FEniCS) como **numérica** (usando FEniCS, si está instalado). Las opciones disponibles son:

- **taylor_green**:  
  Genera datos para el vórtice de Taylor-Green.  
  Parámetros recomendados:  
  --nu (viscosidad), --nx, --ny (resolución espacial) y --nt (pasos temporales).  
  Ejemplo:
  ```bash
  python src/utils/analytical_fluid_generators.py --problem taylor_green --nu 0.01 --nx 64 --ny 64 --nt 20
  ```

- **kovasznay**:  
  Genera datos para el flujo de Kovasznay.  
  Parámetros recomendados:  
  --re (número de Reynolds), --nx y --ny (resolución de la malla).  
  Ejemplo:
  ```bash
  python src/utils/analytical_fluid_generators.py --problem kovasznay --re 40 --nx 64 --ny 64
  ```

- **cavity_flow**:  
  Genera datos para el flujo en cavidad (lid driven cavity).  
  Parámetros recomendados:  
  --re (Reynolds) y --n (resolución de la malla, equivalente a la dimensión espacial).  
  Ejemplo:
  ```bash
  python src/utils/analytical_fluid_generators.py --problem cavity_flow --re 100 --nx 64
  ```

- **burgers**:  
  Genera la solución analítica para la ecuación de Burgers 1D.  
  Parámetros recomendados:  
  --nu (coeficiente de viscosidad), --nx (puntos espaciales) y --nt (puntos temporales).  
  Ejemplo:
  ```bash
  python src/utils/analytical_fluid_generators.py --problem burgers --nu 0.01 --nx 256 --nt 100
  ```

Cada uno de estos comandos generará un archivo (por ejemplo, en formato .mat) con los datos resultantes y, si se solicita, visualizaciones automáticas de la solución. Los parámetros pueden ajustarse según sea necesario para experimentos o pruebas rápidas.

## Generating Synthetic/Analytical Fluid Data

To generate various fluid datasets (e.g., Taylor-Green, Kovasznay, etc.), run:

```bash
python src/utils/analytical_fluid_generators.py --problem taylor_green --nu 0.01 --nx 64 --ny 64 --nt 20
```

Available problems:
- taylor_green
- kovasznay
- cavity_flow
- burgers

Use the arguments --nu, --re, --nx, --ny, --nt as needed.

## Gestión de Entorno con UV

Este proyecto incluye soporte para [UV](https://github.com/astral-sh/uv), una herramienta moderna para la gestión de entornos virtuales y paquetes de Python escrita en Rust. UV ofrece una alternativa mucho más rápida a pip, venv y virtualenv.

### Ventajas de UV

- **Instalación más rápida**: Hasta 10-100x más rápido que pip, especialmente crítico para librerías científicas pesadas
- **Resolución eficiente de dependencias**: Resuelve conflictos de dependencias de manera más inteligente
- **Paralelismo**: Instala múltiples paquetes simultáneamente
- **Totalmente compatible**: Funciona con requirements.txt y pyproject.toml
- **Interfaz unificada**: Combina la gestión de entornos virtuales y la instalación de paquetes

### Uso de UV con el Proyecto

Hemos incluido un script `setup_uv.py` para facilitar la configuración del entorno:

```bash
# Ejecutar el script de configuración
python setup_uv.py

# Activar el entorno virtual (Windows)
.venv\Scripts\activate

# Activar el entorno virtual (Linux/macOS)
source .venv/bin/activate
```

Este script:
1. Instala UV si no está presente
2. Crea un entorno virtual en la carpeta `.venv`
3. Instala todas las dependencias del proyecto desde requirements.txt

### Comandos UV Útiles

Una vez instalado UV, puede usar estos comandos directamente:

```bash
# Instalar un nuevo paquete
uv pip install nombre_paquete

# Actualizar dependencias
uv pip sync requirements.txt

# Generar un lockfile para mejorar reproducibilidad
uv pip compile requirements.txt -o requirements.lock

# Ejecutar código en el entorno virtual
uv run python src/main.py
```

### Comparación con Herramientas Tradicionales

| Característica | UV | pip + venv | conda |
|---------------|-----|------------|-------|
| Velocidad de instalación | Muy rápida | Lenta | Media |
| Manejo de entornos | Integrado | Separado (venv) | Integrado |
| Resolución de dependencias | Avanzada | Básica | Avanzada |
| Compatibilidad con C/C++ | ✓ | ✓ | ✓ |
| Simplicidad | Alta | Media | Baja |
| Tamaño en disco | Bajo | Bajo | Alto |

### Compatibilidad

UV es compatible con Python 3.7+ en todos los sistemas operativos principales (Windows, macOS, Linux).

## Herramientas de Automatización

El proyecto incluye varias herramientas para automatizar la ejecución de experimentos en diferentes entornos:

### Makefile (Linux/macOS)

Para usuarios de sistemas Unix, se proporciona un `Makefile` completo con opciones para entrenar distintos modelos, generar datos o ejecutar evaluaciones:

```bash
# Entrenar un modelo específico
make train_burgers
make train_kovasznay
make train_taylor_green
make train_cavity_flow

# Entrenar todos los modelos secuencialmente
make train_all

# Generar datos
make generate_burgers
make generate_all

# Ejecutar experimentos
make experiment_batch_size
make experiment_layers

# Ver todas las opciones disponibles
make help
```

### Script Batch (Windows)

Para usuarios de Windows, se incluye `run_models.bat` que ofrece un menú interactivo para ejecutar diferentes tareas:

```
run_models.bat
```

El script permite seleccionar entre opciones como entrenar modelos específicos, generar datos o evaluar modelos mediante un menú fácil de usar.

### Script SLURM (Clusters HPC)

Para entornos de computación de alto rendimiento (HPC), se incluye `run_pinn.slurm` para enviar trabajos a gestores de colas basados en SLURM:

```bash
# Enviar un trabajo para todos los modelos
sbatch run_pinn.slurm

# Ejecutar un modelo específico (ej: kovasznay, índice 1)
sbatch --array=1 run_pinn.slurm

# Personalizar parámetros
EPOCHS=200 LR=0.0005 BATCH_SIZE=64 sbatch run_pinn.slurm

# Cambiar el modo de ejecución
MODE=evaluate sbatch --array=0-3 run_pinn.slurm
```

### Scripts de Diagnóstico

Además, se incluyen scripts auxiliares para verificar el entorno y los modelos:

- **check_requirements.py**: Verifica que todas las dependencias estén instaladas correctamente y proporciona instrucciones de instalación para las faltantes.
- **check_models.py**: Analiza la estructura y disponibilidad de los modelos en el sistema.

Estos scripts se ejecutan automáticamente con algunos comandos, pero también pueden invocarse directamente:

```bash
python src/check_requirements.py
python src/check_models.py
```

## Logs y Resultados

- Todos los modos generan logs detallados del proceso.
- Los resultados de entrenamiento se guardan en carpetas organizadas.
- La generación de datos incluye metadatos y estadísticas en JSON.
- Las evaluaciones producen métricas y visualizaciones comparativas.

## Extensibilidad

El framework está diseñado para ser fácilmente extensible:
- Nuevos generadores de datos pueden añadirse en `data_handlers/generators/`
- Nuevas arquitecturas de modelos pueden implementarse extendiendo las clases base
- Los parámetros físicos pueden modificarse sin cambiar el código

## Notas Adicionales

- Se recomienda la instalación editable para actualizar automáticamente el entorno tras modificar el código.
- Las visualizaciones se guardan automáticamente en las carpetas correspondientes al tipo de ejecución.
- Para experimentos rápidos, se pueden reducir los parámetros de entrenamiento en la configuración.

## Citación
@software{pena2023pinnbgk,
  author = {Pe{\~n}a, Cristian},
  title = {{PINN-BGK-Solver}: Un Framework para la Resolución de Ecuaciones Diferenciales con PINN},
  year = {2023},
  url = {https://github.com/CristianPe1/PINN-BGK-Solver},
  version = {1.0.0},
  organization = {Universidad Nacional de Colombia}
}

## Licencia