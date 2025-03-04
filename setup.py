import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# Leer la descripción larga del README si existe
long_description = ""
readme_path = os.path.join(here, "README.md")
if os.path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

# Leer los requerimientos desde requirements.txt, filtrando líneas no válidas
requirements = []
requirements_path = os.path.join(here, "requirements.txt")
if os.path.exists(requirements_path):
    with open(requirements_path, encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#") and not line.startswith("--")]

# Configuración del paquete PINN-BGK:
setup(
    name="PINN-BGK",
    version="1.4.3",
    author="Cristian Peña",
    author_email="cpenav@unal.edu.co",
    description="Proyecto Final: Optimización y Control en Redes Distribuidas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[req for req in requirements if not req.startswith("git+")],
    dependency_links=[req for req in requirements if req.startswith("git+")],
    entry_points={
        "console_scripts": [
            "PINN-BGK=main:main"  # Se asume que en main.py se define main()
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
