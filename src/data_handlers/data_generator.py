import os
import json
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime
import sys
import importlib.util

# Intentar importar FEniCS con manejo de errores
FENICS_AVAILABLE = False
try:
    import fenics as fe
    import mshr
    FENICS_AVAILABLE = True
except ImportError:
    print("FEniCS no está disponible. Las simulaciones numéricas no funcionarán.")
    print("Para instalar FEniCS, siga las instrucciones en: https://fenicsproject.org/download/")
    print("Para Windows, se recomienda usar Docker o WSL2.")
    fe = None
    mshr = None

from .data_loader import DataLoader

# Verificar si el módulo fluid_data_generators está disponible e importarlo
SYNTHETIC_GENERATORS_AVAILABLE = False
try:
    # Intentamos importar los generadores sintéticos
    from ..utils.fluid_data_generators import (
        TaylorGreenDataGenerator, 
        KovasznayDataGenerator, 
        CavityFlowDataGenerator
    )
    SYNTHETIC_GENERATORS_AVAILABLE = True
except ImportError:
    print("Generadores de fluidos sintéticos no disponibles. Se usarán solo soluciones numéricas.")

# Configuración del logger
logger = logging.getLogger("fluid_generator")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DataGenerator(DataLoader):
    """
    Clase para generar datos de simulaciones de fluidos utilizando FEniCS.
    Hereda de DataLoader y añade funcionalidades específicas para simulaciones de fluidos.
    """
    def __init__(self, spatial_points=256, time_points=100, output_dir=None):
        """
        Inicializa el generador con parámetros específicos.
        
        Args:
            spatial_points (int): Número de puntos en la malla espacial
            time_points (int): Número de puntos en la malla temporal
            output_dir (str, optional): Directorio donde guardar los resultados. 
                                      Por defecto es "data/generated"
        """
        super().__init__(spatial_points, time_points)
        
        if output_dir is None:
            # Directorio por defecto para datos generados
            self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                          "data", "generated")
        else:
            self.output_dir = output_dir
            
        # Crear directorio si no existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Crear subdirectorios para cada tipo de fluido
        self.fluid_dirs = {
            'burgers': os.path.join(self.output_dir, "burgers"),
            'taylor_green': os.path.join(self.output_dir, "taylor_green"),
            'kovasznay': os.path.join(self.output_dir, "kovasznay"),
            'cavity_flow': os.path.join(self.output_dir, "cavity_flow"),
        }
        
        for dir_path in self.fluid_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Crear subdirectorio de logs
        self.logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Configurar el FileHandler para el logger
        log_file = os.path.join(self.logs_dir, f"fluid_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(fh)
            # Añadir también un StreamHandler para ver logs en la consola
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        
        # Generar un identificador único para esta sesión de generación
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Inicializar lista para registrar todas las simulaciones generadas
        self.generated_simulations = []
        
        # Verificar disponibilidad de generadores
        if not FENICS_AVAILABLE:
            logger.warning("FEniCS no está disponible. Las simulaciones numéricas no funcionarán.")
            logger.warning("Solo se podrán generar datos sintéticos.")
        else:
            logger.info("FEniCS disponible para simulaciones numéricas.")
            
        if not SYNTHETIC_GENERATORS_AVAILABLE:
            logger.warning("Generadores sintéticos no disponibles. Algunas funcionalidades estarán limitadas.")
        else:
            logger.info("Generadores sintéticos disponibles para todos los tipos de flujo.")
        
        logger.info(f"FluidDataGenerator inicializado: spatial_points={spatial_points}, time_points={time_points}")
        logger.info(f"Directorio de salida: {self.output_dir}")
    
    def generate_fluid_data(self, fluid_type, method='synthetic', params=None, visualize=True, 
                           save=True, format='hdf5'):
        """
        Punto de entrada principal para generar datos de fluidos según el tipo especificado.
        
        Args:
            fluid_type (str): Tipo de fluido ('taylor_green', 'kovasznay', 'cavity_flow', 'burgers')
            method (str): Método de generación ('synthetic', 'numerical')
            params (dict): Parámetros específicos del fluido
            visualize (bool): Si se generan visualizaciones
            save (bool): Si se guardan los datos
            format (str): Formato de guardado ('hdf5', 'npz', 'mat', 'csv')
            
        Returns:
            dict: Datos generados (o ruta a los datos guardados si save=True)
        """
        logger.info(f"Generando datos para fluido: {fluid_type} usando método: {method}")
        
        # Establecer parámetros por defecto si no se proporcionan
        if params is None:
            params = {}
        
        # Directorio específico para este tipo de fluido
        fluid_dir = self.fluid_dirs.get(fluid_type, self.output_dir)
        os.makedirs(fluid_dir, exist_ok=True)
        
        # Subcarpeta para visualizaciones
        vis_dir = os.path.join(fluid_dir, "visualizaciones")
        if visualize:
            os.makedirs(vis_dir, exist_ok=True)
        
        # Generar datos según el tipo de fluido y método
        if method == 'synthetic' and SYNTHETIC_GENERATORS_AVAILABLE:
            if fluid_type == 'taylor_green':
                return self._generate_taylor_green_synthetic(params, fluid_dir, vis_dir, 
                                                           visualize, save, format)
            elif fluid_type == 'kovasznay':
                return self._generate_kovasznay_synthetic(params, fluid_dir, vis_dir, 
                                                        visualize, save, format)
            elif fluid_type == 'cavity_flow':
                return self._generate_cavity_flow_synthetic(params, fluid_dir, vis_dir, 
                                                          visualize, save, format)
            elif fluid_type == 'burgers':
                return self.burgers_synthetic_solution(params.get('nu', 0.01/np.pi), save)
            else:
                logger.error(f"Tipo de fluido desconocido: {fluid_type}")
                return None
        elif method == 'numerical':
            if not FENICS_AVAILABLE:
                logger.error("No se pueden generar datos numéricos. FEniCS no está disponible.")
                print("\n=== ERROR: FEniCS NO DISPONIBLE ===")
                print("Para generar simulaciones numéricas, necesita instalar FEniCS:")
                print("- Visite: https://fenicsproject.org/download/")
                print("- O use un entorno Docker: docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable")
                print("- En Windows, considere usar WSL2 o Docker Desktop\n")
                return None
                
            if fluid_type == 'taylor_green':
                return self.generate_taylor_green_vortex(**params, save=save)
            elif fluid_type == 'kovasznay':
                return self.generate_kovasznay_flow(**params, save=save)
            elif fluid_type == 'cavity_flow':
                return self.generate_lid_driven_cavity(**params, save=save)
            elif fluid_type == 'burgers':
                return self.generate_burgers_data(**params, save=save)
            else:
                logger.error(f"Tipo de fluido desconocido: {fluid_type}")
                return None
        else:
            if not SYNTHETIC_GENERATORS_AVAILABLE and method == 'synthetic':
                logger.error("No se pueden generar datos sintéticos. Los generadores no están disponibles.")
            elif method not in ['synthetic', 'numerical']:
                logger.error(f"Método desconocido: {method}")
            return None
    
    def _generate_taylor_green_synthetic(self, params, fluid_dir, vis_dir, visualize, save, format):
        """Genera datos sintéticos para el flujo de Taylor-Green"""
        nu = params.get('nu', 0.01)
        nx = params.get('nx', self.spatial_points)
        ny = params.get('ny', self.spatial_points)
        nt = params.get('nt', self.time_points)
        
        # Crear generador
        generator = TaylorGreenDataGenerator(nu=nu)
        
        # Generar datos
        data = generator.generate_data(nx=nx, ny=ny, nt=nt)
        
        # Visualizar si se solicita
        if visualize:
            for t_idx in range(0, nt, max(1, nt//5)):  # 5 instantes de tiempo
                generator.plot_data(data, vis_dir, t_idx)
        
        # Guardar datos
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(fluid_dir, f"taylor_green_nu{nu}_nx{nx}_ny{ny}_nt{nt}_{timestamp}.{format}")
            generator.save_data(data, output_path, format=format)
            
            # Registrar simulación
            self._log_simulation('taylor_green_synthetic', 
                                params, 
                                output_path,
                                {'data_shape': f"{nx}x{ny}x{nt}"})
            
            return output_path
        
        return data
    
    def _generate_kovasznay_synthetic(self, params, fluid_dir, vis_dir, visualize, save, format):
        """Genera datos sintéticos para el flujo de Kovasznay"""
        re = params.get('re', 40)
        nx = params.get('nx', self.spatial_points)
        ny = params.get('ny', self.spatial_points)
        
        # Crear generador
        generator = KovasznayDataGenerator(re=re)
        
        # Generar datos
        data = generator.generate_data(nx=nx, ny=ny)
        
        # Visualizar si se solicita
        if visualize:
            generator.plot_data(data, vis_dir)
        
        # Guardar datos
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(fluid_dir, f"kovasznay_re{re}_nx{nx}_ny{ny}_{timestamp}.{format}")
            generator.save_data(data, output_path, format=format)
            
            # Registrar simulación
            self._log_simulation('kovasznay_synthetic', 
                                params, 
                                output_path,
                                {'data_shape': f"{nx}x{ny}"})
            
            return output_path
        
        return data
    
    def _generate_cavity_flow_synthetic(self, params, fluid_dir, vis_dir, visualize, save, format):
        """Genera datos sintéticos para el flujo en cavidad"""
        re = params.get('re', 100)
        n = params.get('n', self.spatial_points)
        
        # Crear generador
        generator = CavityFlowDataGenerator(re=re)
        
        # Generar datos
        data = generator.generate_data(n=n)
        
        # Visualizar si se solicita
        if visualize:
            generator.plot_data(data, vis_dir)
        
        # Guardar datos
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(fluid_dir, f"cavity_flow_re{re}_n{n}_{timestamp}.{format}")
            generator.save_data(data, output_path, format=format)
            
            # Registrar simulación
            self._log_simulation('cavity_flow_synthetic', 
                                params, 
                                output_path,
                                {'data_shape': f"{n}x{n}"})
            
            return output_path
        
        return data

    def _log_simulation(self, simulation_type, parameters, output_file, metadata=None):
        """
        Registra información sobre una simulación generada.
        
        Args:
            simulation_type (str): Tipo de simulación (ej: "kovasznay", "burgers")
            parameters (dict): Parámetros usados
            output_file (str): Ruta del archivo generado
            metadata (dict, optional): Metadatos adicionales
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        simulation_info = {
            "timestamp": timestamp,
            "simulation_type": simulation_type,
            "parameters": parameters,
            "output_file": output_file,
            "metadata": metadata or {}
        }
        
        # Añadir a la lista de simulaciones
        self.generated_simulations.append(simulation_info)
        
        # Guardar como archivo JSON con timestamped filename
        metadata_file = os.path.join(
            self.logs_dir, 
            f"{simulation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(metadata_file, 'w') as f:
            json.dump(simulation_info, f, indent=2)
        
        # Log info
        logger.info(f"Simulación generada: {simulation_type}")
        logger.info(f"Parámetros: {parameters}")
        logger.info(f"Archivo de salida: {output_file}")
        logger.info(f"Metadatos guardados en: {metadata_file}")
        
        return metadata_file
        
    def save_generation_report(self):
        """
        Guarda un informe de todas las simulaciones generadas en esta sesión.
        
        Returns:
            str: Ruta al archivo de informe generado
        """
        if not self.generated_simulations:
            logger.warning("No hay simulaciones registradas para generar un informe.")
            return None
            
        report = {
            "session_id": self.session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_simulations": len(self.generated_simulations),
            "simulations": self.generated_simulations
        }
        
        report_file = os.path.join(self.logs_dir, f"generation_report_{self.session_id}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Informe de generación guardado en: {report_file}")
        logger.info(f"Total de simulaciones generadas: {len(self.generated_simulations)}")
        
        return report_file
        
    def generate_burgers_data(self, nu=0.01/np.pi, nx=256, nt=100, save=True):
        """
        Genera datos para la ecuación de Burgers utilizando FEniCS.
        
        Args:
            nu (float): Viscosidad
            nx (int): Número de puntos espaciales
            nt (int): Número de puntos temporales
            save (bool): Si True, guarda los resultados en formato .mat
            
        Returns:
            tuple: (X, T, U) donde X e T son mallas de coordenadas y U son 
                  los valores de velocidad y presión
        """
        if not FENICS_AVAILABLE:
            logger.error("Esta función requiere FEniCS instalado.")
            raise RuntimeError("Esta función requiere FEniCS instalado.")
        
        logger.info(f"Generando datos de Burgers con nu={nu}, nx={nx}, nt={nt}...")
        start_time = datetime.now()
        
        # Implementación basada en la función main_burgers de fluids_generators.py
        # Básicamente llamamos a la función existente
        from .generators.fluids_generators import burgers_exact_solution
        
        # Crear malla rectangular
        mesh = fe.IntervalMesh(nx, -1, 1)
        
        # Espacios funcionales
        V = fe.FunctionSpace(mesh, "CG", 1)
        Q = fe.FunctionSpace(mesh, "CG", 1)
        W = fe.FunctionSpace(mesh, V * Q)
        
        # Definir funciones
        w = fe.Function(W)
        (u, p) = fe.split(w)
        (v_test, q_test) = fe.TestFunctions(W)
        
        # Solución exacta para condiciones de frontera
        u_ex, p_ex = burgers_exact_solution(nu)
        
        # Condiciones de frontera
        bc_left = fe.DirichletBC(W.sub(0), fe.Expression("uL(x[0])", degree=4, uL=u_ex), "near(x[0], -1)")
        bc_right = fe.DirichletBC(W.sub(0), fe.Expression("uR(x[0])", degree=4, uR=u_ex), "near(x[0], 1)")
        bcs = [bc_left, bc_right]
        
        # Ecuación de Burgers
        f = fe.Constant(0.0)
        dt = fe.Constant(1.0 / nt)
        
        F_inertia = fe.dot(u - u_ex, v_test) * fe.dx
        F_pressure = fe.dot(p - p_ex, q_test) * fe.dx
        F_continuity = fe.dot(fe.grad(u) * u, v_test) * fe.dx
        F_viscous = nu * fe.dot(fe.grad(u), fe.grad(v_test)) * fe.dx
        
        F_total = (F_inertia + F_pressure + F_continuity + F_viscous - fe.dot(f, v_test)) * fe.dx
        
        # Resolver
        fe.solve(F_total == 0, w, bcs, solver_parameters={"newton_solver": {"relative_tolerance": 1e-6}})
        
        # Extraer solución
        u_sol, p_sol = w.split()
        
        
        
    def generate_kovasznay_flow(self, Re=40, N=32, save=True):
        """
        Genera datos para el flujo de Kovasznay utilizando FEniCS.
        
        Args:
            Re (float): Número de Reynolds
            N (int): Resolución de la malla
            save (bool): Si True, guarda los resultados en formato .mat
            
        Returns:
            tuple: (X, T, U) donde X e T son mallas de coordenadas y U son 
                  los valores de velocidad y presión
        """
        if not FENICS_AVAILABLE:
            logger.error("Esta función requiere FEniCS instalado.")
            raise RuntimeError("Esta función requiere FEniCS instalado.")
        
        logger.info(f"Generando flujo de Kovasznay con Re={Re}, N={N}...")
        start_time = datetime.now()
        
        # Implementación basada en la función main_kovasznay_flow de fluids_generators.py
        # Básicamente llamamos a la función existente
        from .generators.fluids_generators import kovasznay_exact_solution
        
        # Crear malla rectangular
        domain = mshr.Rectangle(fe.Point(0.0, 0.0), fe.Point(1.5, 1.0))
        mesh = mshr.generate_mesh(domain, N)
        
        # Espacios funcionales
        V = fe.VectorElement("P", mesh.ufl_cell(), 2)
        Q = fe.FiniteElement("P", mesh.ufl_cell(), 1)
        W = fe.FunctionSpace(mesh, V * Q)

        # Definición de funciones
        w = fe.Function(W)
        (u, p) = fe.split(w)
        (v_test, q_test) = fe.TestFunctions(W)
        
        # Solución analítica para condiciones de frontera
        u_ex, v_ex, p_ex = kovasznay_exact_solution(Re)
        
        # Condiciones de frontera
        class KovasznayBC(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary
        
        boundary_markers = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary_markers.set_all(0)
        KovasznayBC().mark(boundary_markers, 1)
        
        bc_u = fe.DirichletBC(W.sub(0),
                            fe.Expression(("uK(x[0], x[1])",
                                           "vK(x[0], x[1])"),
                                          degree=4,
                                          uK=u_ex,
                                          vK=v_ex),
                            boundary_markers, 1)
        
        bcs = [bc_u]
        
        # Ecuación de Navier-Stokes
        mu = 1.0 / Re  # viscosidad cinemática
        rho = 1.0      # densidad
        f = fe.Constant((0.0, 0.0))  # Fuerza externa
        
        eps = lambda u_: fe.sym(fe.grad(u_))
        F_inertia = rho * fe.dot(fe.grad(u) * u, v_test)
        F_pressure = -p * fe.div(v_test)
        F_continuity = q_test * fe.div(u)
        F_viscous = 2 * mu * fe.inner(eps(u), eps(v_test))
        
        F_total = (F_inertia + F_pressure + F_continuity + F_viscous - fe.dot(f, v_test)) * fe.dx
        
        # Resolver
        fe.solve(F_total == 0, w, bcs, 
                solver_parameters={"newton_solver": 
                                  {"relative_tolerance": 1e-8}})
        
        # Extraer solución
        u_sol, p_sol = w.split()
        
        # Convertir a numpy para guardar
        # Obtenemos coordenadas y solución en formato de matriz de puntos
        vertices = mesh.coordinates()
        cells = mesh.cells()
        
        # Evaluar la solución en una malla uniforme para graficar
        nx, ny = self.spatial_points, self.time_points
        x_range = np.linspace(0.0, 1.5, nx)
        y_range = np.linspace(0.0, 1.0, ny)
        
        X, Y = np.meshgrid(x_range, y_range)
        U = np.zeros((nx, ny))
        V = np.zeros((nx, ny))
        P = np.zeros((nx, ny))
        
        # Interpolación de la solución en la malla uniforme
        u_interp = fe.interpolate(u_sol.sub(0), fe.FunctionSpace(mesh, "CG", 1))
        v_interp = fe.interpolate(u_sol.sub(1), fe.FunctionSpace(mesh, "CG", 1))
        p_interp = fe.interpolate(p_sol, fe.FunctionSpace(mesh, "CG", 1))
        
        for i in range(nx):
            for j in range(ny):
                point = fe.Point(x_range[i], y_range[j])
                try:
                    U[i, j] = u_interp(point)
                    V[i, j] = v_interp(point)
                    P[i, j] = p_interp(point)
                except:
                    # Punto fuera del dominio
                    pass
        
        # Construir tensores
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        U_tensor = torch.tensor(U, dtype=torch.float32)
        V_tensor = torch.tensor(V, dtype=torch.float32)
        P_tensor = torch.tensor(P, dtype=torch.float32)
        
        # Combinar U y V en un tensor de velocidades
        UV_tensor = torch.stack([U_tensor, V_tensor], dim=2)
        
        # Calcular estadísticas sobre los datos generados
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        
        output_file = None
        if save:
            # Guardar resultados en formato .mat
            output_file = os.path.join(self.output_dir, f"kovasznay_Re{Re}_N{N}.mat")
            savemat(output_file, {
                'x': X,
                'y': Y,
                'u': U,
                'v': V,
                'p': P,
                'parameters': {
                    'Re': Re,
                    'N': N,
                    'spatial_points': self.spatial_points,
                    'time_points': self.time_points
                }
            })
            logger.info(f"Datos guardados en {output_file}")
        
        # Calcular algunas estadísticas sobre los datos
        stats = {
            "u_max": float(np.max(U)),
            "u_min": float(np.min(U)),
            "u_mean": float(np.mean(U)),
            "v_max": float(np.max(V)),
            "v_min": float(np.min(V)),
            "v_mean": float(np.mean(V)),
            "p_max": float(np.max(P)),
            "p_min": float(np.min(P)),
            "p_mean": float(np.mean(P)),
            "simulation_time_seconds": elapsed_time
        }
        
        # Registrar la simulación
        params = {
            "Re": Re,
            "N": N,
            "spatial_points": self.spatial_points,
            "time_points": self.time_points
        }
        
        metadata = {
            "statistics": stats,
            "fenics_version": fe.__version__ if hasattr(fe, "__version__") else "unknown",
            "computation_time": elapsed_time,
            "mesh_info": {
                "num_vertices": mesh.num_vertices(),
                "num_cells": mesh.num_cells()
            }
        }
        
        metadata_file = self._log_simulation("kovasznay_flow", params, output_file, metadata)
        
        return X_tensor, Y_tensor, UV_tensor, P_tensor

    def generate_taylor_green_vortex(self, nu=0.01, U0=1.0, Nx=32, Ny=32, T=2.0, num_steps=50, save=True):
        """
        Genera datos para el vórtice de Taylor-Green utilizando FEniCS.
        
        Args:
            nu (float): Viscosidad
            U0 (float): Amplitud inicial
            Nx, Ny (int): Resolución de la malla
            T (float): Tiempo total de simulación
            num_steps (int): Número de pasos de tiempo
            save (bool): Si True, guarda los resultados
        """
        if not FENICS_AVAILABLE:
            logger.error("Esta función requiere FEniCS instalado.")
            raise RuntimeError("Esta función requiere FEniCS instalado.")
        
        logger.info(f"Generando vórtice de Taylor-Green con nu={nu}, U0={U0}, Nx={Nx}, Ny={Ny}, T={T}...")
        start_time = datetime.now()
        
        # Implementación similar a main_taylor_green_vortex
        dt = T / num_steps
        
        # Crear malla rectangular [0, 2*pi] x [0, 2*pi]
        mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(2*np.pi, 2*np.pi), Nx, Ny)
        
        # Definir espacios
        V = fe.VectorElement("P", mesh.ufl_cell(), 2)
        Q = fe.FiniteElement("P", mesh.ufl_cell(), 1)
        W = fe.FunctionSpace(mesh, V * Q)

        # Condiciones iniciales
        w0 = fe.Function(W)
        
        class InitCond(fe.UserExpression):
            def __init__(self, U0, **kwargs):
                self.U0 = U0
                super().__init__(**kwargs)
            def eval(self, values, x):
                values[0] = self.U0 * np.sin(x[0]) * np.cos(x[1])  # u_x
                values[1] = -self.U0 * np.cos(x[0]) * np.sin(x[1]) # u_y
            def value_shape(self):
                return (2,)
        
        # Asignar campo de velocidad inicial
        ic_expr = InitCond(U0=U0, degree=4)
        w0.sub(0).interpolate(ic_expr)
        
        # Variables
        w = fe.Function(W)
        w.assign(w0)
        (u, p_) = fe.split(w)
        (v_test, q_test) = fe.TestFunctions(W)
        
        # Ecuaciones
        f = fe.Constant((0.0, 0.0))
        rho = 1.0
        dt_fe = fe.Constant(dt)
        u_n, p_n = fe.split(w0)
        
        # Términos variaciones
        F_inertia = rho * fe.dot((u - u_n) / dt_fe, v_test) * fe.dx \
                    + rho * fe.dot(fe.grad(u_n)*u_n, v_test) * fe.dx
        F_pressure = fe.dot(fe.grad(p_), v_test)*fe.dx
        F_continuity = fe.dot(q_test, fe.div(u))*fe.dx
        F_viscous = 2*nu*fe.inner(fe.sym(fe.grad(u)), fe.sym(fe.grad(v_test)))*fe.dx
        
        # Ecuación total
        F_total = F_inertia + F_pressure - F_continuity + F_viscous - fe.dot(f, v_test)*fe.dx
        
        # Lista para guardar soluciones en cada paso de tiempo
        u_history = []
        p_history = []
        
        # Avanzar en el tiempo
        t = 0.0
        for step in range(num_steps):
            t += dt
            # Resolver
            fe.solve(F_total == 0, w, [], 
                    solver_parameters={"newton_solver": 
                                      {"relative_tolerance": 1e-6}})
            
            # Actualizar w0 con la solución del paso actual
            w0.assign(w)
            
            # Guardar la solución actual
            u_sol, p_sol = w.split()
            u_history.append(u_sol.copy())
            p_history.append(p_sol.copy())
            
            if step % 10 == 0 or step == num_steps - 1:
                logger.info(f"Paso de tiempo {step+1}/{num_steps} completado (t={t:.3f})")
        
        # Construir malla uniforme para graficar/guardar
        nx, ny = self.spatial_points, self.time_points
        x_range = np.linspace(0, 2*np.pi, nx)
        y_range = np.linspace(0, 2*np.pi, ny)
        
        X, Y = np.meshgrid(x_range, y_range)
        U_final = np.zeros((nx, ny))
        V_final = np.zeros((nx, ny))
        P_final = np.zeros((nx, ny))
        
        # Interpolar la solución final
        u_interp = fe.interpolate(u_history[-1].sub(0), fe.FunctionSpace(mesh, "CG", 1))
        v_interp = fe.interpolate(u_history[-1].sub(1), fe.FunctionSpace(mesh, "CG", 1))
        p_interp = fe.interpolate(p_history[-1], fe.FunctionSpace(mesh, "CG", 1))
        
        for i in range(nx):
            for j in range(ny):
                point = fe.Point(x_range[i], y_range[j])
                try:
                    U_final[i, j] = u_interp(point)
                    V_final[i, j] = v_interp(point)
                    P_final[i, j] = p_interp(point)
                except:
                    pass
        
        # Convertir a tensores
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        U_tensor = torch.tensor(U_final, dtype=torch.float32)
        V_tensor = torch.tensor(V_final, dtype=torch.float32)
        P_tensor = torch.tensor(P_final, dtype=torch.float32)
        
        # Combinar velocidades
        UV_tensor = torch.stack([U_tensor, V_tensor], dim=2)
        
        # Calcular estadísticas
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        
        output_file = None
        if save:
            output_file = os.path.join(self.output_dir, f"taylor_green_nu{nu}_U0{U0}_T{T}.mat")
            savemat(output_file, {
                'x': X,
                'y': Y,
                'u': U_final,
                'v': V_final,
                'p': P_final,
                'parameters': {
                    'nu': nu,
                    'U0': U0,
                    'T': T,
                    'num_steps': num_steps,
                    'spatial_points': self.spatial_points,
                    'time_points': self.time_points,
                    'Nx': Nx,
                    'Ny': Ny
                }
            })
            logger.info(f"Datos guardados en {output_file}")
        
        stats = {
            "u_max": float(np.max(U_final)),
            "u_min": float(np.min(U_final)),
            "u_mean": float(np.mean(U_final)),
            "v_max": float(np.max(V_final)),
            "v_min": float(np.min(V_final)),
            "v_mean": float(np.mean(V_final)),
            "p_max": float(np.max(P_final)),
            "p_min": float(np.min(P_final)),
            "p_mean": float(np.mean(P_final)),
            "simulation_time_seconds": elapsed_time,
            "energy": float(np.mean(U_final**2 + V_final**2)),
        }
        
        params = {
            "nu": nu,
            "U0": U0,
            "T": T,
            "num_steps": num_steps,
            "Nx": Nx,
            "Ny": Ny,
            "spatial_points": self.spatial_points,
            "time_points": self.time_points
        }
        
        metadata = {
            "statistics": stats,
            "computation_time": elapsed_time,
            "mesh_info": {
                "num_vertices": mesh.num_vertices(),
                "num_cells": mesh.num_cells(),
                "domain": "2π x 2π"
            }
        }
        
        metadata_file = self._log_simulation("taylor_green_vortex", params, output_file, metadata)
        
        return X_tensor, Y_tensor, UV_tensor, P_tensor

    def generate_lid_driven_cavity(self, nu=0.01, U0=1.0, N=32, T=2.0, num_steps=50, save=True):
        """
        Genera datos para el flujo en cavidad conducido por tapa utilizando FEniCS.
        
        Args:
            nu (float): Viscosidad
            U0 (float): Velocidad de la tapa
            N (int): Resolución de la malla
            T (float): Tiempo total
            num_steps (int): Número de pasos de tiempo
            save (bool): Si True, guarda los resultados
        """
        if not FENICS_AVAILABLE:
            raise RuntimeError("Esta función requiere FEniCS instalado.")
        
        print(f"Generando flujo en cavidad con tapa móvil, nu={nu}, U0={U0}...")
        
        # Implementación basada en main_lid_driven_cavity
        dt = T / num_steps
        
        # Crear malla unitaria
        mesh = fe.UnitSquareMesh(N, N)
        
        # Espacio (u, p)
        V = fe.VectorElement("P", mesh.ufl_cell(), 2)
        Q = fe.FiniteElement("P", mesh.ufl_cell(), 1)
        W = fe.FunctionSpace(mesh, V * Q)
        
        # Variables
        w0 = fe.Function(W)
        w = fe.Function(W)
        (u_n, p_n) = w0.split()
        (u, p) = fe.split(w)
        (v_test, q_test) = fe.TestFunctions(W)
        
        # Definir subdominios de frontera
        class TopBoundary(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], 1.0)
        
        class OtherWalls(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[1] < 1.0 and on_boundary
        
        boundary_markers = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        boundary_markers.set_all(0)
        
        top = TopBoundary()
        walls = OtherWalls()
        
        top.mark(boundary_markers, 1)
        walls.mark(boundary_markers, 2)
        
        # Velocidad en la tapa (U0, 0)
        lid_velocity = fe.Constant((U0, 0.0))
        bc_top = fe.DirichletBC(W.sub(0), lid_velocity, boundary_markers, 1)
        
        # Otras paredes con no-slip
        bc_other = fe.DirichletBC(W.sub(0), fe.Constant((0.0,0.0)), boundary_markers, 2)
        
        bcs = [bc_top, bc_other]
        
        # Campo inicial
        fe.assign(w0.sub(0), fe.Constant((0.0, 0.0)))  # velocidad inicial
        
        # Parámetros
        rho = 1.0
        dt_fe = fe.Constant(dt)
        f = fe.Constant((0.0, 0.0))
        
        # Formulación variacional
        F_inertia = rho * fe.dot((u - u_n)/dt_fe, v_test)*fe.dx \
                    + rho * fe.dot(fe.grad(u)*u, v_test)*fe.dx
        F_pressure = -p * fe.div(v_test)*fe.dx
        F_continuity = q_test * fe.div(u)*fe.dx
        F_viscous = nu * fe.inner(fe.grad(u), fe.grad(v_test))*fe.dx
        
        F_total = F_inertia + F_pressure + F_continuity + F_viscous - fe.dot(f, v_test)*fe.dx
        
        # Lista para almacenar resultados de cada paso temporal
        u_history = []
        p_history = []
        
        # Bucle temporal
        t = 0.0
        for step in range(num_steps):
            t += dt
            fe.solve(F_total == 0, w, bcs,
                     solver_parameters={"newton_solver":
                                        {"relative_tolerance": 1e-6}})
            w0.assign(w)
            
            # Guardar solución en este paso
            u_sol, p_sol = w.split()
            u_history.append(u_sol.copy())
            p_history.append(p_sol.copy())
        
        # Construir malla uniforme para visualizar/guardar
        nx, ny = self.spatial_points, self.time_points
        x_range = np.linspace(0, 1, nx)
        y_range = np.linspace(0, 1, ny)
        
        X, Y = np.meshgrid(x_range, y_range)
        U_final = np.zeros((nx, ny))
        V_final = np.zeros((nx, ny))
        P_final = np.zeros((nx, ny))
        
        # Interpolar la solución final
        u_interp = fe.interpolate(u_history[-1].sub(0), fe.FunctionSpace(mesh, "CG", 1))
        v_interp = fe.interpolate(u_history[-1].sub(1), fe.FunctionSpace(mesh, "CG", 1))
        p_interp = fe.interpolate(p_history[-1], fe.FunctionSpace(mesh, "CG", 1))
        
        for i in range(nx):
            for j in range(ny):
                point = fe.Point(x_range[i], y_range[j])
                try:
                    U_final[i, j] = u_interp(point)
                    V_final[i, j] = v_interp(point)
                    P_final[i, j] = p_interp(point)
                except:
                    pass
        
        # Convertir a tensores
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        U_tensor = torch.tensor(U_final, dtype=torch.float32)
        V_tensor = torch.tensor(V_final, dtype=torch.float32)
        P_tensor = torch.tensor(P_final, dtype=torch.float32)
        
        UV_tensor = torch.stack([U_tensor, V_tensor], dim=2)
        
        if save:
            filename = os.path.join(self.output_dir, f"lid_driven_cavity_nu{nu}_U0{U0}_T{T}.mat")
            savemat(filename, {
                'x': X,
                'y': Y,
                'u': U_final,
                'v': V_final,
                'p': P_final,
                'parameters': {
                    'nu': nu,
                    'U0': U0,
                    'T': T,
                    'num_steps': num_steps
                }
            })
            print(f"Datos guardados en {filename}")
        
        return X_tensor, Y_tensor, UV_tensor, P_tensor
    
    def burgers_synthetic_solution(self, nu=0.01/np.pi, save=True):
        """
        Genera una solución sintética para la ecuación de Burgers 1D.
        
        Args:
            nu (float): Coeficiente de viscosidad
            save (bool): Si True, guarda los resultados
            
        Returns:
            tuple: (x, t, usol)
        """
        logger.info(f"Generando solución sintética de Burgers con nu={nu}...")
        start_time = datetime.now()
        
        # Crear mallas
        x = np.linspace(-1, 1, self.spatial_points)
        t = np.linspace(0, 1, self.time_points)
        X, T = np.meshgrid(x, t)
        usol = np.zeros((self.spatial_points, self.time_points))
        
        # Condición inicial en t=0: -sin(pi*x)
        usol[:, 0] = -np.sin(np.pi * x)
        
        # Para cada tiempo t>0, calcular la solución exacta
        for j in range(1, self.time_points):
            t_val = t[j]
            for i in range(self.spatial_points):
                x_val = x[i]
                # Solución exacta de Burgers para la condición inicial -sin(pi*x)
                denominator = 1 + (1 - np.exp(-nu*np.pi**2*t_val)) * np.cos(np.pi * x_val) / (np.sin(np.pi * x_val) + 1e-8)
                usol[i, j] = -np.sin(np.pi * x_val) * np.exp(-nu*np.pi**2*t_val) / denominator
        
        # Convertir a tensores
        x_tensor = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)
        t_tensor = torch.tensor(t.reshape(1, -1), dtype=torch.float32)
        usol_tensor = torch.tensor(usol, dtype=torch.float32)
        
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        
        output_file = None
        if save:
            output_file = os.path.join(self.output_dir, f"burgers_shock_nu_{nu:.5f}.mat")
            savemat(output_file, {
                'x': x.reshape(-1, 1), 
                't': t.reshape(1, -1), 
                'usol': usol,
                'parameters': {
                    'nu': nu,
                    'spatial_points': self.spatial_points,
                    'time_points': self.time_points
                }
            })
            logger.info(f"Datos guardados en {output_file}")
        
        # Calcular estadísticas
        stats = {
            "u_max": float(np.max(usol)),
            "u_min": float(np.min(usol)),
            "u_mean": float(np.mean(usol)),
            "u_std": float(np.std(usol)),
            "solution_time_seconds": elapsed_time
        }
        
        params = {
            "nu": nu,
            "spatial_points": self.spatial_points,
            "time_points": self.time_points,
            "solution_type": "analytic"
        }
        
        metadata = {
            "statistics": stats,
            "computation_time": elapsed_time,
            "domain_info": {
                "x_range": "[-1, 1]",
                "t_range": "[0, 1]"
            }
        }
        
        metadata_file = self._log_simulation("burgers_synthetic", params, output_file, metadata)
        
        return x_tensor, t_tensor, usol_tensor

if __name__ == "__main__":
    # Configuración personalizada del logger para este bloque
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Ejemplo de uso actualizado con manejo de errores mejorado
    generator = DataGenerator(spatial_points=256, time_points=100)
    
    # Generar datos usando el nuevo método unificado
    print("Generando datos para diferentes fluidos...")
    
    # Detectar automáticamente qué método usar basado en disponibilidad
    method_to_use = 'synthetic' if SYNTHETIC_GENERATORS_AVAILABLE else ('numerical' if FENICS_AVAILABLE else None)
    
    if method_to_use is None:
        print("ERROR: No hay ningún método de generación disponible.")
        print("- Para datos sintéticos: asegúrese de que utils/fluid_data_generators.py existe")
        print("- Para simulaciones numéricas: instale FEniCS (https://fenicsproject.org/download/)")
        sys.exit(1)
    
    print(f"Usando método: {method_to_use}")
    
    # Taylor-Green 
    if method_to_use == 'synthetic' and SYNTHETIC_GENERATORS_AVAILABLE:
        print("1. Generando datos de Taylor-Green (sintético)")
        generator.generate_fluid_data(
            fluid_type='taylor_green',
            method='synthetic',
            params={'nu': 0.01, 'nx': 50, 'ny': 50, 'nt': 10},
            visualize=True,
            format='hdf5'
        )
        
        # Kovasznay sintético
        print("2. Generando datos de Kovasznay (sintético)")
        generator.generate_fluid_data(
            fluid_type='kovasznay',
            method='synthetic',
            params={'re': 40, 'nx': 100, 'ny': 50},
            visualize=True,
            format='hdf5'
        )
    elif FENICS_AVAILABLE:
        print("1. Generando datos de Taylor-Green (numérico)")
        generator.generate_fluid_data(
            fluid_type='taylor_green',
            method='numerical',
            params={'nu': 0.01, 'Nx': 32, 'Ny': 32},
            visualize=True
        )
    else:
        print("No se pueden generar datos: ni FEniCS ni los generadores sintéticos están disponibles")
        sys.exit(1)
    
    # Guardar informe de generación
    report_path = generator.save_generation_report()
    print(f"Informe de generación guardado en: {report_path}")
