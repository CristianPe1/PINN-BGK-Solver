"""
Script para generar datos sintéticos de problemas de fluidos
"""
import os
import sys
import argparse
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Verificar si podemos importar el generador de datos
try:
    from data_handlers.data_generator import DataGenerator, FENICS_AVAILABLE, SYNTHETIC_GENERATORS_AVAILABLE
except ImportError as e:
    logger.error(f"Error al importar el generador de datos: {e}")
    print("\nComprobando requisitos del sistema...")
    try:
        # Intentar ejecutar check_requirements.py si existe
        script_dir = os.path.dirname(os.path.abspath(__file__))
        check_req_script = os.path.join(script_dir, "check_requirements.py")
        if os.path.exists(check_req_script):
            import subprocess
            subprocess.run([sys.executable, check_req_script])
        else:
            print("El script check_requirements.py no se encuentra disponible.")
    except Exception as check_error:
        print(f"Error al ejecutar check_requirements.py: {check_error}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Generador de datos para problemas de fluidos')
    
    # Tipo de fluido
    parser.add_argument('--fluid', type=str, required=True, choices=['taylor_green', 'kovasznay', 'cavity_flow', 'burgers'],
                        help='Tipo de fluido a generar')
    
    # Método de generación
    parser.add_argument('--method', type=str, default='synthetic', choices=['synthetic', 'numerical'],
                        help='Método de generación (sintético o numérico)')
    
    # Parámetros comunes
    parser.add_argument('--nx', type=int, default=64, 
                        help='Número de puntos en dirección x')
    parser.add_argument('--ny', type=int, default=64,
                        help='Número de puntos en dirección y')
    parser.add_argument('--nt', type=int, default=20,
                        help='Número de puntos en dirección t')
    
    # Parámetros físicos
    parser.add_argument('--nu', type=float, default=0.01,
                        help='Viscosidad cinemática')
    parser.add_argument('--re', type=float, default=40,
                        help='Número de Reynolds')
    
    # Parámetros de salida
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directorio donde guardar los resultados')
    parser.add_argument('--format', type=str, default='hdf5', choices=['hdf5', 'npz', 'csv', 'mat'],
                        help='Formato de guardado de los datos')
    parser.add_argument('--no-vis', action='store_true',
                        help='No generar visualizaciones')
    
    args = parser.parse_args()
    
    # Verificar compatibilidad de método y disponibilidad
    if args.method == 'numerical' and not FENICS_AVAILABLE:
        logger.error("El método 'numerical' requiere FEniCS, pero FEniCS no está disponible.")
        print("\nError: No se puede usar el método 'numerical' sin FEniCS instalado.")
        print("Por favor, instale FEniCS o use el método 'synthetic'.")
        print("Para más información, ejecute: python src/check_requirements.py")
        return 1

    if args.method == 'synthetic' and not SYNTHETIC_GENERATORS_AVAILABLE:
        logger.error("El método 'synthetic' requiere los generadores sintéticos, pero no están disponibles.")
        print("\nError: No se puede usar el método 'synthetic' sin los generadores adecuados.")
        print("Verifique que el archivo src/utils/fluid_data_generators.py existe.")
        return 1
    
    # Configurar parámetros según el tipo de fluido
    if args.fluid == 'taylor_green':
        params = {
            'nu': args.nu,
            'nx': args.nx,
            'ny': args.ny,
            'nt': args.nt
        }
    elif args.fluid == 'kovasznay':
        params = {
            're': args.re,
            'nx': args.nx,
            'ny': args.ny
        }
    elif args.fluid == 'cavity_flow':
        params = {
            're': args.re,
            'n': args.nx
        }
    elif args.fluid == 'burgers':
        params = {
            'nu': args.nu,
            'nx': args.nx,
            'nt': args.nt
        }
    
    # Crear generador
    generator = DataGenerator(
        spatial_points=args.nx,
        time_points=args.nt,
        output_dir=args.output_dir
    )
    
    # Generar datos
    logger.info(f"Generando datos de {args.fluid} usando método {args.method}")
    result = generator.generate_fluid_data(
        fluid_type=args.fluid,
        method=args.method,
        params=params,
        visualize=not args.no_vis,
        save=True,
        format=args.format
    )
    
    if result:
        if isinstance(result, str):
            logger.info(f"Datos generados y guardados en: {result}")
        else:
            logger.info(f"Datos generados correctamente")
    else:
        logger.error("Error al generar los datos")
        return 1
    
    # Guardar informe de generación
    report_path = generator.save_generation_report()
    logger.info(f"Informe de generación guardado en: {report_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
