#!/bin/bash
#SBATCH --job-name=PINN-BGK      # Nombre del trabajo
#SBATCH --output=logs/pinn_%A_%a.out  # Archivo de salida (%A: ID trabajo, %a: ID array)
#SBATCH --error=logs/pinn_%A_%a.err   # Archivo de errores
#SBATCH --time=24:00:00          # Tiempo máximo de ejecución
#SBATCH --ntasks=1               # Número de tareas
#SBATCH --cpus-per-task=4        # CPUs por tarea
#SBATCH --mem=32G                # Memoria por nodo
#SBATCH --gres=gpu:1             # Solicita 1 GPU
#SBATCH --array=0-6              # Array job para diferentes modelos (0-6 = 7 modelos)
##SBATCH --partition=gpu          # Partición con GPUs (descomentarla según el sistema)

# Crear directorio para logs si no existe
mkdir -p logs

# Directorio del proyecto
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd $PROJECT_DIR

# Activar entorno virtual (modificar según tu configuración)
# source /path/to/venv/bin/activate

# Configurar variables según el índice del array
MODEL_TYPES=("burgers" "kovasznay" "taylor_green" "cavity_flow" "kovasznay" "taylor_green" "cavity_flow")
MODEL_NAMES=("pinn_v1" "kovasznay" "taylor_green" "cavity_flow" "lbm_naive_kovasznay" "lbm_naive_taylor_green" "lbm_naive_cavity_flow")
MODE=${MODE:-"train"}              # Modo predeterminado: train (alternativas: generate, evaluate)
MODEL_IDX=${SLURM_ARRAY_TASK_ID:-0}  # Índice 0 por defecto si no se ejecuta como array
MODEL_TYPE=${MODEL_TYPES[$MODEL_IDX]}
MODEL_NAME=${MODEL_NAMES[$MODEL_IDX]}

# Parámetros personalizables (pueden ser modificados al enviar el trabajo)
EPOCHS=${EPOCHS:-100}
LR=${LR:-0.001}
BATCH_SIZE=${BATCH_SIZE:-32}
TAU=${TAU:-0.6}           # Parámetro específico para LBM
OMEGA=${OMEGA:-1.7}       # Parámetro específico para LBM
LATTICE_TYPE=${LATTICE_TYPE:-"D2Q9"} # Tipo de lattice para LBM

# Determinar si es un modelo LBM
IS_LBM=0
if [[ $MODEL_NAME == *"lbm"* ]]; then
    IS_LBM=1
fi

# Establecer configuración personalizada
CONFIG_FILE="config/slurm_${MODEL_TYPE}_${MODEL_NAME}.yaml"
cat > $CONFIG_FILE << EOL
# Configuración autogenerada para trabajo SLURM
selected_model: "${MODEL_NAME}"

models:
  ${MODEL_NAME}:
    type: "${MODEL_TYPE}"
EOL

# Añadir configuración específica según el tipo de modelo
if [ $IS_LBM -eq 1 ]; then
    # Configuración para modelos LBM
    VARIANT=$(echo $MODEL_NAME | sed 's/lbm_\([^_]*\)_.*/\1/')
    cat >> $CONFIG_FILE << EOL
    variant: "${VARIANT}"
    layers: [9, 50, 50, 9]
    activation_function: "Tanh"
    learning_rate: ${LR}
    physics_type: "${MODEL_TYPE}"
EOL
else
    # Configuración para modelos estándar
    cat >> $CONFIG_FILE << EOL
    layers: [50, 50, 50]
    activation_function: "Tanh"
    learning_rate: ${LR}
    physics_type: "${MODEL_TYPE}"
EOL
fi

# Añadir configuración de entrenamiento
cat >> $CONFIG_FILE << EOL

training:
  epochs: ${EPOCHS}
  batch_size: ${BATCH_SIZE}
  early_stopping:
    enabled: true
  epochs_no_improve: 20
  memory_limit_gb: 16
EOL

# Añadir configuración específica para LBM si es necesario
if [ $IS_LBM -eq 1 ]; then
    cat >> $CONFIG_FILE << EOL

lattice_boltzmann:
  lattice_type: "${LATTICE_TYPE}"
  tau: ${TAU}
  omega: ${OMEGA}
  grid_size_x: 128
  grid_size_y: 128
  physics_type: "lattice_boltzmann"
EOL
fi

# Mostrar información sobre el trabajo
echo "=============================================="
echo "Trabajo SLURM iniciado: $(date)"
echo "ID del trabajo: ${SLURM_JOB_ID}"
echo "ID del array: ${SLURM_ARRAY_TASK_ID}"
echo "Nodo: ${SLURM_JOB_NODELIST}"
echo "=============================================="
echo "Modo: ${MODE}"
echo "Modelo: ${MODEL_NAME} (${MODEL_TYPE})"
echo "Épocas: ${EPOCHS}"
echo "Learning rate: ${LR}"
echo "Batch size: ${BATCH_SIZE}"
if [ $IS_LBM -eq 1 ]; then
    echo "LBM específico: Sí"
    echo "Tipo de lattice: ${LATTICE_TYPE}"
    echo "Tau: ${TAU}"
    echo "Omega: ${OMEGA}"
    echo "Variante: ${VARIANT}"
fi
echo "Archivo de configuración: ${CONFIG_FILE}"
echo "=============================================="

# Verificar GPUs disponibles
if command -v nvidia-smi &> /dev/null; then
    echo "Información de GPU:"
    nvidia-smi
else
    echo "nvidia-smi no disponible. ¿Está trabajando en una GPU?"
fi

# Ejecutar el script principal
echo "Ejecutando PINN-BGK en modo ${MODE}..."
python src/main.py --mode ${MODE} --config ${CONFIG_FILE}

# Comprobar estado de salida
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "=============================================="
    echo "Trabajo completado con éxito: $(date)"
    echo "=============================================="
else
    echo "=============================================="
    echo "ERROR: El trabajo ha terminado con código de error ${EXIT_CODE}"
    echo "=============================================="
fi

exit $EXIT_CODE
