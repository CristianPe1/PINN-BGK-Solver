# Makefile para automatizar la ejecución de modelos PINN-BGK

# Variables configurables
PYTHON = python
SRC_DIR = src
CONFIG_DIR = config
DATA_DIR = data

# Comandos básicos
TRAIN_CMD = $(PYTHON) $(SRC_DIR)/main.py --mode train
GENERATE_CMD = $(PYTHON) $(SRC_DIR)/main.py --mode generate
EVALUATE_CMD = $(PYTHON) $(SRC_DIR)/main.py --mode evaluate

# Configuraciones de entrenamiento
DEFAULT_CONFIG = $(CONFIG_DIR)/config.yaml
BURGERS_CONFIG = $(CONFIG_DIR)/burgers_config.yaml
KOVASZNAY_CONFIG = $(CONFIG_DIR)/kovasznay_config.yaml
TAYLOR_GREEN_CONFIG = $(CONFIG_DIR)/taylor_green_config.yaml
CAVITY_FLOW_CONFIG = $(CONFIG_DIR)/cavity_flow_config.yaml

# Configuraciones para Lattice Boltzmann
LBM_KOVASZNAY_CONFIG = $(CONFIG_DIR)/lbm_kovasznay_config.yaml
LBM_TAYLOR_GREEN_CONFIG = $(CONFIG_DIR)/lbm_taylor_green_config.yaml
LBM_CAVITY_FLOW_CONFIG = $(CONFIG_DIR)/lbm_cavity_flow_config.yaml

# Verifica requisitos de instalación
check_requirements:
	$(PYTHON) $(SRC_DIR)/check_requirements.py

# Entrenar diferentes modelos con configuraciones específicas
train_default: check_requirements
	$(TRAIN_CMD) --config $(DEFAULT_CONFIG)

# Objetivos para modelos estándar
train_burgers: check_requirements
	@echo "Entrenando modelo para ecuación de Burgers..."
	@cp $(DEFAULT_CONFIG) $(BURGERS_CONFIG)
	@sed -i 's/"selected_model": ".*"/"selected_model": "pinn_v1"/g' $(BURGERS_CONFIG)
	@sed -i 's/"physics_type": ".*"/"physics_type": "burgers"/g' $(BURGERS_CONFIG)
	$(TRAIN_CMD) --config $(BURGERS_CONFIG)

train_kovasznay: check_requirements
	@echo "Entrenando modelo para flujo de Kovasznay..."
	@cp $(DEFAULT_CONFIG) $(KOVASZNAY_CONFIG)
	@sed -i 's/"selected_model": ".*"/"selected_model": "kovasznay"/g' $(KOVASZNAY_CONFIG)
	@sed -i 's/"physics_type": ".*"/"physics_type": "kovasznay"/g' $(KOVASZNAY_CONFIG)
	$(TRAIN_CMD) --config $(KOVASZNAY_CONFIG)

train_taylor_green: check_requirements
	@echo "Entrenando modelo para vórtice de Taylor-Green..."
	@cp $(DEFAULT_CONFIG) $(TAYLOR_GREEN_CONFIG)
	@sed -i 's/"selected_model": ".*"/"selected_model": "taylor_green"/g' $(TAYLOR_GREEN_CONFIG)
	@sed -i 's/"physics_type": ".*"/"physics_type": "taylor_green"/g' $(TAYLOR_GREEN_CONFIG)
	$(TRAIN_CMD) --config $(TAYLOR_GREEN_CONFIG)

train_cavity_flow: check_requirements
	@echo "Entrenando modelo para flujo en cavidad..."
	@cp $(DEFAULT_CONFIG) $(CAVITY_FLOW_CONFIG)
	@sed -i 's/"selected_model": ".*"/"selected_model": "cavity_flow"/g' $(CAVITY_FLOW_CONFIG)
	@sed -i 's/"physics_type": ".*"/"physics_type": "cavity_flow"/g' $(CAVITY_FLOW_CONFIG)
	$(TRAIN_CMD) --config $(CAVITY_FLOW_CONFIG)

# Nuevos objetivos para modelos Lattice Boltzmann
train_lbm_kovasznay: check_requirements
	@echo "Entrenando modelo Lattice Boltzmann para flujo de Kovasznay..."
	@cp $(DEFAULT_CONFIG) $(LBM_KOVASZNAY_CONFIG)
	@sed -i 's/"selected_model": ".*"/"selected_model": "lbm_naive_kovasznay"/g' $(LBM_KOVASZNAY_CONFIG)
	@sed -i 's/"physics_type": ".*"/"physics_type": "kovasznay"/g' $(LBM_KOVASZNAY_CONFIG)
	$(TRAIN_CMD) --config $(LBM_KOVASZNAY_CONFIG)

train_lbm_taylor_green: check_requirements
	@echo "Entrenando modelo Lattice Boltzmann para vórtice de Taylor-Green..."
	@cp $(DEFAULT_CONFIG) $(LBM_TAYLOR_GREEN_CONFIG)
	@sed -i 's/"selected_model": ".*"/"selected_model": "lbm_naive_taylor_green"/g' $(LBM_TAYLOR_GREEN_CONFIG)
	@sed -i 's/"physics_type": ".*"/"physics_type": "taylor_green"/g' $(LBM_TAYLOR_GREEN_CONFIG)
	$(TRAIN_CMD) --config $(LBM_TAYLOR_GREEN_CONFIG)

train_lbm_cavity_flow: check_requirements
	@echo "Entrenando modelo Lattice Boltzmann para flujo en cavidad..."
	@cp $(DEFAULT_CONFIG) $(LBM_CAVITY_FLOW_CONFIG)
	@sed -i 's/"selected_model": ".*"/"selected_model": "lbm_naive_cavity_flow"/g' $(LBM_CAVITY_FLOW_CONFIG)
	@sed -i 's/"physics_type": ".*"/"physics_type": "cavity_flow"/g' $(LBM_CAVITY_FLOW_CONFIG)
	$(TRAIN_CMD) --config $(LBM_CAVITY_FLOW_CONFIG)

# Entrenar todos los modelos secuencialmente - versiones estándar
train_all: train_burgers train_kovasznay train_taylor_green train_cavity_flow
	@echo "Todos los modelos estándar han sido entrenados."

# Entrenar todos los modelos Lattice Boltzmann
train_all_lbm: train_lbm_kovasznay train_lbm_taylor_green train_lbm_cavity_flow
	@echo "Todos los modelos Lattice Boltzmann han sido entrenados."

# Generación de datos para diferentes problemas
generate_burgers:
	@echo "Generando datos para ecuación de Burgers..."
	@cp $(DEFAULT_CONFIG) $(CONFIG_DIR)/generate_burgers.yaml
	@sed -i 's/"type": ".*"/"type": "burgers"/g' $(CONFIG_DIR)/generate_burgers.yaml
	$(GENERATE_CMD) --config $(CONFIG_DIR)/generate_burgers.yaml

generate_kovasznay:
	@echo "Generando datos para flujo de Kovasznay..."
	@cp $(DEFAULT_CONFIG) $(CONFIG_DIR)/generate_kovasznay.yaml
	@sed -i 's/"type": ".*"/"type": "kovasznay"/g' $(CONFIG_DIR)/generate_kovasznay.yaml
	$(GENERATE_CMD) --config $(CONFIG_DIR)/generate_kovasznay.yaml

generate_taylor_green:
	@echo "Generando datos para vórtice de Taylor-Green..."
	@cp $(DEFAULT_CONFIG) $(CONFIG_DIR)/generate_tg.yaml
	@sed -i 's/"type": ".*"/"type": "taylor_green"/g' $(CONFIG_DIR)/generate_tg.yaml
	$(GENERATE_CMD) --config $(CONFIG_DIR)/generate_tg.yaml

generate_cavity:
	@echo "Generando datos para flujo en cavidad..."
	@cp $(DEFAULT_CONFIG) $(CONFIG_DIR)/generate_cavity.yaml
	@sed -i 's/"type": ".*"/"type": "lid_driven_cavity"/g' $(CONFIG_DIR)/generate_cavity.yaml
	$(GENERATE_CMD) --config $(CONFIG_DIR)/generate_cavity.yaml

# Generación de datos para Lattice Boltzmann
generate_lbm_data:
	@echo "Generando datos específicos para Lattice Boltzmann..."
	@cp $(DEFAULT_CONFIG) $(CONFIG_DIR)/generate_lbm.yaml
	@sed -i 's/"type": ".*"/"type": "lattice_boltzmann"/g' $(CONFIG_DIR)/generate_lbm.yaml
	@sed -i 's/"lattice_type": ".*"/"lattice_type": "D2Q9"/g' $(CONFIG_DIR)/generate_lbm.yaml
	$(GENERATE_CMD) --config $(CONFIG_DIR)/generate_lbm.yaml

generate_all: generate_burgers generate_kovasznay generate_taylor_green generate_cavity generate_lbm_data
	@echo "Todos los datos han sido generados."

# Evaluación de modelos
evaluate_latest:
	$(EVALUATE_CMD)

evaluate_burgers:
	@echo "Evaluando modelo de Burgers..."
	@cp $(DEFAULT_CONFIG) $(CONFIG_DIR)/eval_burgers.yaml
	@sed -i 's/"physics_type": ".*"/"physics_type": "burgers"/g' $(CONFIG_DIR)/eval_burgers.yaml
	$(EVALUATE_CMD) --config $(CONFIG_DIR)/eval_burgers.yaml

evaluate_kovasznay:
	@echo "Evaluando modelo de Kovasznay..."
	@cp $(DEFAULT_CONFIG) $(CONFIG_DIR)/eval_kovasznay.yaml
	@sed -i 's/"physics_type": ".*"/"physics_type": "kovasznay"/g' $(CONFIG_DIR)/eval_kovasznay.yaml
	$(EVALUATE_CMD) --config $(CONFIG_DIR)/eval_kovasznay.yaml

evaluate_lbm:
	@echo "Evaluando modelo Lattice Boltzmann..."
	@cp $(DEFAULT_CONFIG) $(CONFIG_DIR)/eval_lbm.yaml
	@sed -i 's/"physics_type": ".*"/"physics_type": "lattice_boltzmann"/g' $(CONFIG_DIR)/eval_lbm.yaml
	$(EVALUATE_CMD) --config $(CONFIG_DIR)/eval_lbm.yaml

# Experimentos con diferentes configuraciones de los modelos
experiment_batch_size:
	@echo "Ejecutando experimento con diferentes tamaños de batch..."
	@for size in 16 32 64 128 256; do \
		echo "Entrenando con batch_size=$$size"; \
		cp $(DEFAULT_CONFIG) $(CONFIG_DIR)/batch_$$size.yaml; \
		sed -i "s/\"batch_size\": [0-9]*/\"batch_size\": $$size/g" $(CONFIG_DIR)/batch_$$size.yaml; \
		$(TRAIN_CMD) --config $(CONFIG_DIR)/batch_$$size.yaml; \
	done

experiment_layers:
	@echo "Ejecutando experimento con diferentes configuraciones de capas..."
	@for layers in 2 3 4 5; do \
		echo "Entrenando con $$layers capas ocultas"; \
		cp $(DEFAULT_CONFIG) $(CONFIG_DIR)/layers_$$layers.yaml; \
		sed -i "s/\"layers\": \[[0-9, ]*\]/\"layers\": [50$(shell printf ',%s' $(shell seq -s ' ' $$layers | sed 's/[0-9]/50/g'))]/g" $(CONFIG_DIR)/layers_$$layers.yaml; \
		$(TRAIN_CMD) --config $(CONFIG_DIR)/layers_$$layers.yaml; \
	done

# Experimento con diferentes variantes de LBM
experiment_lbm_variants:
	@echo "Ejecutando experimento con diferentes variantes de Lattice Boltzmann..."
	@for variant in naive sym cons; do \
		echo "Entrenando con variante LBM=$$variant"; \
		cp $(DEFAULT_CONFIG) $(CONFIG_DIR)/lbm_$$variant.yaml; \
		sed -i "s/\"variant\": \".*\"/\"variant\": \"$$variant\"/g" $(CONFIG_DIR)/lbm_$$variant.yaml; \
		sed -i "s/\"selected_model\": \".*\"/\"selected_model\": \"lbm_$$variant_kovasznay\"/g" $(CONFIG_DIR)/lbm_$$variant.yaml; \
		$(TRAIN_CMD) --config $(CONFIG_DIR)/lbm_$$variant.yaml; \
	done

# Limpieza de archivos temporales
clean:
	@echo "Eliminando archivos temporales de configuración..."
	rm -f $(CONFIG_DIR)/*_config.yaml
	rm -f $(CONFIG_DIR)/generate_*.yaml
	rm -f $(CONFIG_DIR)/eval_*.yaml
	rm -f $(CONFIG_DIR)/batch_*.yaml
	rm -f $(CONFIG_DIR)/layers_*.yaml
	rm -f $(CONFIG_DIR)/lbm_*.yaml

# Ayuda
help:
	@echo "PINN-BGK Makefile - Opciones disponibles:"
	@echo "  make train_default       - Entrena con la configuración por defecto"
	@echo "  make train_burgers       - Entrena modelo para ecuación de Burgers"
	@echo "  make train_kovasznay     - Entrena modelo para flujo de Kovasznay"
	@echo "  make train_taylor_green  - Entrena modelo para vórtice de Taylor-Green"
	@echo "  make train_cavity_flow   - Entrena modelo para flujo en cavidad"
	@echo "  make train_all           - Entrena todos los modelos estándar secuencialmente"
	@echo ""
	@echo "  make train_lbm_kovasznay - Entrena modelo Lattice Boltzmann para flujo de Kovasznay"
	@echo "  make train_lbm_taylor_green - Entrena modelo LBM para vórtice de Taylor-Green"
	@echo "  make train_lbm_cavity_flow - Entrena modelo LBM para flujo en cavidad"
	@echo "  make train_all_lbm       - Entrena todos los modelos LBM secuencialmente"
	@echo ""
	@echo "  make generate_burgers    - Genera datos para ecuación de Burgers"
	@echo "  make generate_lbm_data   - Genera datos específicos para Lattice Boltzmann"
	@echo "  make generate_all        - Genera datos para todos los problemas"
	@echo ""
	@echo "  make evaluate_latest     - Evalúa el modelo más reciente"
	@echo "  make evaluate_lbm        - Evalúa modelo LBM específicamente"
	@echo ""
	@echo "  make experiment_batch_size - Ejecuta experimento con diferentes tamaños de batch"
	@echo "  make experiment_layers   - Ejecuta experimento con diferentes configuraciones de capas"
	@echo "  make experiment_lbm_variants - Ejecuta experimento con variantes de LBM"
	@echo ""
	@echo "  make clean               - Elimina archivos temporales"

.PHONY: check_requirements train_default train_burgers train_kovasznay train_taylor_green train_cavity_flow train_all \
		train_lbm_kovasznay train_lbm_taylor_green train_lbm_cavity_flow train_all_lbm \
		generate_burgers generate_kovasznay generate_taylor_green generate_cavity generate_lbm_data generate_all \
		evaluate_latest evaluate_burgers evaluate_kovasznay evaluate_lbm \
		experiment_batch_size experiment_layers experiment_lbm_variants clean help
