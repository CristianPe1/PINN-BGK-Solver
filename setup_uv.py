
import subprocess
import os
import sys
import platform

def main():
    """Configura el proyecto usando UV para gestión de dependencias"""
    print("Configurando entorno para PINN-BGK con UV...")
    
    # Verificar si uv está instalado
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("UV está instalado correctamente")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("UV no está instalado. Instalando...")
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    
    # Crear entorno virtual
    if not os.path.exists(".venv"):
        print("Creando entorno virtual...")
        subprocess.run(["uv", "venv", ".venv"], check=True)
    
    # Activar entorno virtual (esto no funciona directamente en Python, solo muestra el comando)
    if platform.system() == "Windows":
        activate_cmd = ".venv\\Scripts\\activate"
    else:
        activate_cmd = "source .venv/bin/activate"
    print(f"Para activar el entorno, ejecute: {activate_cmd}")
    
    # Instalar dependencias
    if os.path.exists("requirements.txt"):
        print("Instalando dependencias desde requirements.txt...")
        if platform.system() == "Windows":
            subprocess.run([".venv\\Scripts\\uv", "pip", "install", "-r", "requirements.txt"], check=True)
        else:
            subprocess.run([".venv/bin/uv", "pip", "install", "-r", "requirements.txt"], check=True)
    else:
        print("No se encontró requirements.txt, instalando desde setup.py...")
        if platform.system() == "Windows":
            subprocess.run([".venv\\Scripts\\uv", "pip", "install", "-e", "."], check=True)
        else:
            subprocess.run([".venv/bin/uv", "pip", "install", "-e", "."], check=True)
    
    print("Entorno configurado correctamente con UV")
    print(f"Para activar el entorno, ejecute: {activate_cmd}")

if __name__ == "__main__":
    main()
