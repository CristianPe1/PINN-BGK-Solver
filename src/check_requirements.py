"""
Script para verificar los requisitos y dependencias del proyecto
"""
import sys
import os
import platform
import importlib.util
import subprocess

def check_package(package_name):
    """
    Verifica si un paquete está instalado.
    
    Args:
        package_name (str): Nombre del paquete
        
    Returns:
        tuple: (is_available, version)
    """
    package_spec = importlib.util.find_spec(package_name)
    
    if package_spec is not None:
        try:
            package = importlib.import_module(package_name)
            version = getattr(package, "__version__", "desconocida")
            return True, version
        except ImportError:
            return True, "desconocida"
    else:
        return False, None
        
def print_status(name, is_available, version=None, required=False, notes=None):
    """Imprime el estado de una dependencia"""
    status = "✓ INSTALADO" if is_available else "✗ FALTA"
    if required and not is_available:
        status = "✗ REQUERIDO"
        
    version_str = f"v{version}" if version else ""
    
    print(f"{status:12} {name:20} {version_str:15} {notes or ''}")

def get_installation_instructions(package_name):
    """
    Proporciona instrucciones para instalar un paquete.
    """
    instructions = {
        "torch": {
            "pip": "pip install torch",
            "conda": "conda install pytorch -c pytorch",
            "notes": "Visita https://pytorch.org/get-started/locally/ para instrucciones específicas según tu sistema"
        },
        "fenics": {
            "pip": "pip no soporta instalación directa de FEniCS",
            "conda": "conda install -c conda-forge fenics",
            "notes": "Para instalación completa, visita https://fenicsproject.org/download/"
        },
        "mshr": {
            "pip": "pip no soporta instalación directa de mshr",
            "conda": "conda install -c conda-forge mshr",
            "notes": "Requiere FEniCS previamente instalado."
        },
        "matplotlib": {
            "pip": "pip install matplotlib",
            "conda": "conda install matplotlib",
            "notes": ""
        },
        "numpy": {
            "pip": "pip install numpy",
            "conda": "conda install numpy",
            "notes": ""
        },
        "scipy": {
            "pip": "pip install scipy",
            "conda": "conda install scipy",
            "notes": ""
        },
        "h5py": {
            "pip": "pip install h5py",
            "conda": "conda install h5py",
            "notes": ""
        }
    }
    
    return instructions.get(package_name, {
        "pip": f"pip install {package_name}",
        "conda": f"conda install {package_name}",
        "notes": ""
    })

def check_fenics():
    """
    Verificar si FEniCS está correctamente instalado.
    """
    print("\n=== Verificación detallada de FEniCS ===")
    
    fenics_available, fenics_version = check_package("fenics")
    if not fenics_available:
        print("FEniCS no está instalado en este entorno Python.")
        print("\nOpciones de instalación para FEniCS:")
        
        system = platform.system()
        if system == "Windows":
            print("En Windows, se recomienda uno de estos métodos:")
            print("1. Usar WSL (Windows Subsystem for Linux):")
            print("   - Instale WSL2 desde Microsoft Store")
            print("   - En WSL, ejecute: sudo apt-get install software-properties-common")
            print("   - Luego: sudo add-apt-repository ppa:fenics-packages/fenics")
            print("   - Y: sudo apt-get update && sudo apt-get install fenics")
            
            print("\n2. Usar Docker:")
            print("   - Instale Docker Desktop desde https://www.docker.com/products/docker-desktop")
            print("   - Ejecute: docker run -ti -v %cd%:/home/fenics/shared quay.io/fenicsproject/stable")
        
        elif system == "Linux":
            print("En Linux:")
            print("- Ubuntu/Debian: sudo apt-get install software-properties-common")
            print("- sudo add-apt-repository ppa:fenics-packages/fenics")
            print("- sudo apt-get update && sudo apt-get install fenics")
            print("- O con conda: conda install -c conda-forge fenics")
        
        elif system == "Darwin":  # macOS
            print("En macOS:")
            print("- Usando Homebrew: brew install fenics")
            print("- O con conda: conda install -c conda-forge fenics")
        
        print("\nMás información: https://fenicsproject.org/download/")
        return False
    
    mshr_available, mshr_version = check_package("mshr")
    
    print(f"FEniCS está instalado (versión {fenics_version}).")
    
    if not mshr_available:
        print("El módulo mshr no está instalado. Este es necesario para algunas funcionalidades de generación de mallas.")
        print("Se puede instalar con: conda install -c conda-forge mshr")
        
    try:
        import fenics as fe
        mesh = fe.UnitSquareMesh(2, 2)
        print("✓ Prueba básica de FEniCS exitosa (creación de malla).")
    except Exception as e:
        print(f"✗ Error al ejecutar una operación básica de FEniCS: {str(e)}")
        return False
        
    return True

def main():
    """
    Función principal para verificar requisitos del proyecto.
    """
    print("=== Verificando requisitos del proyecto ===")
    print(f"Python: {platform.python_version()}")
    print(f"Sistema: {platform.system()} {platform.release()}")
    
    # Verificar paquetes esenciales
    packages_to_check = [
        {"name": "numpy", "required": True},
        {"name": "scipy", "required": True},
        {"name": "matplotlib", "required": True},
        {"name": "torch", "required": True},
        {"name": "h5py", "required": False},
        {"name": "fenics", "required": False},
        {"name": "mshr", "required": False},
    ]
    
    print("\n{:12} {:20} {:15} {}".format("ESTADO", "PAQUETE", "VERSIÓN", "NOTAS"))
    print("-" * 70)
    
    missing_packages = []
    missing_required = False
    
    for package in packages_to_check:
        is_available, version = check_package(package["name"])
        
        # Notas adicionales según el paquete
        notes = None
        if package["name"] == "fenics":
            notes = "Necesario para simulaciones numéricas"
        elif package["name"] == "torch":
            if is_available:
                # Verificar si CUDA está disponible
                try:
                    import torch
                    cuda_available = torch.cuda.is_available()
                    notes = f"CUDA {'disponible' if cuda_available else 'no disponible'}"
                except:
                    notes = "Error al verificar CUDA"
            else:
                notes = "Requerido para modelos de deep learning"
        
        print_status(package["name"], is_available, version, package["required"], notes)
        
        if not is_available:
            missing_packages.append(package["name"])
            if package["required"]:
                missing_required = True
    
    # Crear directorio para datos generados si no existe
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "generated")
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir, exist_ok=True)
            print("\n✓ Directorio para datos creado:", data_dir)
        except Exception as e:
            print(f"\n✗ Error al crear directorio para datos: {str(e)}")
    
    # Información de instalación para paquetes faltantes
    if missing_packages:
        print("\n=== Paquetes faltantes ===")
        for package in missing_packages:
            instructions = get_installation_instructions(package)
            print(f"\nPara instalar {package}:")
            print(f"- Con pip: {instructions['pip']}")
            print(f"- Con conda: {instructions['conda']}")
            if instructions['notes']:
                print(f"- Nota: {instructions['notes']}")
        
        # Si falta FEniCS, ofrecer verificación detallada
        if "fenics" in missing_packages:
            print("\nFEniCS no está instalado. ¿Desea ver instrucciones detalladas? (s/n)")
            response = input().lower()
            if response.startswith('s'):
                check_fenics()
    
    # Verificar si se han completado los requisitos
    if missing_required:
        print("\n⚠ ¡ADVERTENCIA! Faltan paquetes requeridos.")
        print("El proyecto no funcionará correctamente hasta que instale todos los paquetes marcados como REQUERIDO.")
        return 1
    else:
        if missing_packages:
            print("\n✓ Todos los paquetes requeridos están instalados.")
            print("  Hay paquetes opcionales faltantes que pueden habilitar funcionalidades adicionales.")
        else:
            print("\n✓ Todos los paquetes están instalados correctamente.")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())
