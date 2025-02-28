"""
Kovasznay Flow con FEniCS:
Se resuelve numéricamente la ecuación de Navier-Stokes 
incompresible en 2D y se comparan los resultados con la 
solución analítica de Kovasznay.
"""

import fenics as fe
import mshr
import numpy as np

def kovasznay_exact_solution(Re, p0=1.0, u0=1.0):
    """
    Retorna funciones Python que dan la solución analítica de Kovasznay:
        u_exact, v_exact, p_exact
    """
    lam = Re/2 - np.sqrt((Re/2)**2 + 4*(np.pi**2))
    
    def u_exact(x, y):
        return u0 * (1.0 - np.exp(lam*x) * np.cos(2*np.pi*y))
    
    def v_exact(x, y):
        return u0 * (lam/(2*np.pi)) * np.exp(lam*x)*np.sin(2*np.pi*y)
    
    def p_exact(x, y):
        return p0 - 0.5*(u0**2)*np.exp(2*lam*x)
    
    return u_exact, v_exact, p_exact

def main_kovasznay_flow(Re=40, N=32):
    """
    Función principal para:
    1. Crear la malla
    2. Definir el problema Navier-Stokes
    3. Aplicar condiciones de frontera según Kovasznay
    4. Resolver y comparar con la solución analítica
    """
    # --- Parámetros de viscosidad cinemática y densidad ---
    # Asumimos densidad = 1 para simplificar
    mu = 1.0 / Re  # viscosidad dinámica ~ 1/Re si densidad=1
    
    # --- Dominio y malla (rectángulo por ejemplo) ---
    # x en [0, 1.5], y en [0, 1], solo para demo
    domain = mshr.Rectangle(fe.Point(0.0, 0.0), fe.Point(1.5, 1.0))
    mesh = mshr.generate_mesh(domain, N)
    
    # Espacios funcionales: velocidad (vectorial) y presión (escalar)
    V = fe.VectorElement("P", mesh.ufl_cell(), 2) 
    Q = fe.FiniteElement("P", mesh.ufl_cell(), 1)
    W = fe.FunctionSpace(mesh, V * Q)  # Producto para (u, p)

    # Definición de funciones de prueba y desconocidas
    w = fe.Function(W)
    (u, p) = fe.split(w)
    (v_test, q_test) = fe.TestFunctions(W)
    
    # Definir funciones analíticas para cond. de frontera
    u_ex, v_ex, p_ex = kovasznay_exact_solution(Re)
    
    # Condiciones de frontera: se usarán subdominios
    #  Para Kovasznay, se asume el valor analítico en TODO el contorno
    class KovasznayBC(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
    
    boundary_markers = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)
    KovasznayBC().mark(boundary_markers, 1)
    
    # Definimos la condición Dirichlet para la velocidad
    bc_u = fe.DirichletBC(W.sub(0),
                          fe.Expression(("uK(x[0], x[1])",
                                         "vK(x[0], x[1])"),
                                        degree=4,
                                        uK=u_ex,
                                        vK=v_ex),
                          boundary_markers, 1)
    
    bcs = [bc_u]  # Solo se necesitan condiciones para la velocidad
    
    # Ecuación de Navier-Stokes (forma débil)
    # Sea rho = 1, la inercia: (u·∇)u
    rho = 1.0
    f = fe.Constant((0.0, 0.0))  # Fuerza externa nula
    eps = lambda u_: fe.sym(fe.grad(u_))
    
    # Término de inercia
    F_inertia = rho * fe.dot(fe.grad(u) * u, v_test)
    
    # Término de presión
    F_pressure = -p * fe.div(v_test)
    
    # Término de continuidad
    F_continuity = q_test * fe.div(u)
    
    # Viscosidad
    F_viscous = 2 * mu * fe.inner(eps(u), eps(v_test))
    
    # Ecuación completa
    F_total = (F_inertia + F_pressure + F_continuity + F_viscous - fe.dot(f, v_test)) * fe.dx
    
    # Ensamblar y resolver
    fe.solve(F_total == 0, w, bcs, 
             solver_parameters={"newton_solver": 
                                {"relative_tolerance": 1e-8}})
    
    # Separar u y p resultantes
    u_sol, p_sol = w.split()
    
    # --- Comparación con solución analítica ---
    # Se puede muestrear en puntos
    error_u = fe.errornorm(fe.Expression(("uK(x[0], x[1])",
                                         "vK(x[0], x[1])"),
                                         degree=4,
                                         uK=u_ex, 
                                         vK=v_ex),
                           u_sol, "L2")
    
    # O también se pueden computar norm. L2 del dominio 
    error_p = fe.errornorm(fe.Expression("pK(x[0], x[1])", 
                                        degree=4,
                                        pK=p_ex),
                           p_sol, "L2")
    
    print(f"Kovasznay (Re={Re}) -> error_u={error_u}, error_p={error_p}")
    
    return mesh, u_sol, p_sol


def main_taylor_green_vortex(nu=0.01, U0=1.0, Nx=32, Ny=32, T=2.0, num_steps=50):
    """
    - nu: viscosidad
    - U0: amplitud inicial
    - Nx, Ny: número de celdas en cada dirección
    - T: tiempo total de simulación
    - num_steps: cantidad de pasos de tiempo
    """
    dt = T / num_steps
    
    # Crear malla rectangular [0, 2*pi] x [0, 2*pi]
    mesh = fe.RectangleMesh(fe.Point(0,0), fe.Point(2*np.pi, 2*np.pi), Nx, Ny)
    
    # Definir el espacio para velocidad y presión
    V = fe.VectorElement("P", mesh.ufl_cell(), 2)
    Q = fe.FiniteElement("P", mesh.ufl_cell(), 1)
    W = fe.FunctionSpace(mesh, V * Q)

    # Condiciones iniciales
    # Usamos funciones que respeten las condiciones periódicas manualmente
    # (en FEniCS habría que crear un Domain y un Mesh con periodos,
    #  aquí se muestra un approach simplificado).
    
    # Definir la función inicial (u, p)
    w0 = fe.Function(W)
    u0, p0_ = w0.split()
    
    class InitCond(fe.UserExpression):
        def __init__(self, U0, **kwargs):
            self.U0 = U0
            super().__init__(**kwargs)
        def eval(self, values, x):
            # x[0] en [0, 2*pi], x[1] en [0, 2*pi]
            values[0] = self.U0 * np.sin(x[0]) * np.cos(x[1])  # u_x
            values[1] = -self.U0 * np.cos(x[0]) * np.sin(x[1]) # u_y
        def value_shape(self):
            return (2,)
    
    # Asignar campo de velocidad inicial
    ic_expr = InitCond(U0=U0, degree=4)
    w0.sub(0).interpolate(ic_expr)  # sub(0) = velocidad
    
    # Definir funciones para cada paso
    w = fe.Function(W)  # (u, p) en el paso actual
    w.assign(w0)
    (u, p_) = fe.split(w)
    (v_test, q_test) = fe.TestFunctions(W)
    
    # Ecuaciones de Navier-Stokes transientes
    f = fe.Constant((0.0, 0.0))   # sin fuerza externa
    rho = 1.0
    dt_fe = fe.Constant(dt)
    
    # Definir la forma variacional (método de semi-implicación)
    u_n, p_n = fe.split(w0)
    
    # Término convectivo (u·∇)u
    F_inertia = rho * fe.dot((u_n) / dt_fe, v_test) * fe.dx \
                + rho * fe.dot(fe.grad(u_n)*u_n, v_test) * fe.dx
    
    # Presión y continuidad
    F_pressure = fe.dot(fe.grad(p_), v_test)*fe.dx - fe.dot(q_test, fe.div(u))*fe.dx
    
    # Viscosidad
    F_viscous = 2*nu*fe.inner(fe.sym(fe.grad(u)), fe.sym(fe.grad(v_test)))*fe.dx
    
    # Ecuación total
    F_total = F_inertia + F_pressure + F_viscous - fe.dot(f, v_test)*fe.dx
    
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
    
    # Extraer solución final
    u_sol, p_sol = w.split()
    print(f"Taylor-Green finalizado. t={t:.2f}, dt={dt:.4f}, steps={num_steps}")
    
    return mesh, u_sol, p_sol


def main_static_cavity_flow(nu=0.01, N=32, T=1.0, num_steps=50):
    """
    Simulación de cavity flow con todas las paredes con 
    no-slip boundary. El flujo se detendrá salvo que 
    haya alguna fuerza o condición inicial no nula.
    
    - nu: viscosidad
    - N: resolución de malla en X=Y=[0,1]
    - T: tiempo total
    - num_steps: número de pasos
    """
    dt = T/num_steps
    
    # Malla
    mesh = fe.UnitSquareMesh(N, N)
    
    # Espacio funcional
    V = fe.VectorElement("P", mesh.ufl_cell(), 2)
    Q = fe.FiniteElement("P", mesh.ufl_cell(), 1)
    W = fe.FunctionSpace(mesh, V*Q)
    
    # Definir variables
    w0 = fe.Function(W)
    (u_n, p_n) = w0.split()
    
    w = fe.Function(W)
    (u, p) = fe.split(w)
    (v_test, q_test) = fe.TestFunctions(W)
    
    # Paredes con no-slip
    # Se definen con Dirichlet en u = (0,0)
    no_slip = fe.Constant((0.0, 0.0))

    # Identificar bordes
    class Walls(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary  # Todas las paredes

    boundary_markers = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundary_markers.set_all(0)
    Walls().mark(boundary_markers, 1)

    bc_walls = fe.DirichletBC(W.sub(0), no_slip, boundary_markers, 1)
    bcs = [bc_walls]
    
    # Campo inicial no trivial (opcional):
    # Se puede asignar un vórtice inicial
    vortex_expr = fe.Expression(("sin(2*pi*x[0])*sin(2*pi*x[1])",
                                 "-sin(2*pi*x[1])*sin(2*pi*x[0])"),
                                degree=4)
    w0.sub(0).interpolate(vortex_expr)
    
    # Parámetros
    rho = 1.0
    dt_fe = fe.Constant(dt)
    f = fe.Constant((0.0, 0.0)) # Sin fuerza externa
    
    # Formulación varicional (similar a Taylor-Green)
    F_inertia = rho * fe.dot((u_n)/dt_fe, v_test)*fe.dx \
                + rho * fe.dot(fe.grad(u_n)*u_n, v_test)*fe.dx
    F_pressure = fe.dot(fe.grad(p), v_test)*fe.dx - fe.dot(q_test, fe.div(u))*fe.dx
    F_viscous = 2*nu*fe.inner(fe.sym(fe.grad(u)), fe.sym(fe.grad(v_test)))*fe.dx
    
    F_total = F_inertia + F_pressure + F_viscous - fe.dot(f, v_test)*fe.dx
    
    # Bucle de tiempo
    t = 0.0
    for step in range(num_steps):
        t += dt
        fe.solve(F_total == 0, w, bcs, 
                 solver_parameters={"newton_solver":
                                    {"relative_tolerance": 1e-6}})
        w0.assign(w)
    
    u_sol, p_sol = w.split()
    print("Static Cavity Flow finalizado.")
    return mesh, u_sol, p_sol

def main_lid_driven_cavity(nu=0.01, U0=1.0, N=32, T=2.0, num_steps=50):
    """
    - nu: viscosidad
    - U0: velocidad de la tapa
    - N: malla NxN en [0,1] x [0,1]
    - T: tiempo total
    - num_steps: número de pasos
    """
    dt = T / num_steps
    
    # Crear malla unitaria
    mesh = fe.UnitSquareMesh(N, N)
    
    # Espacio (u, p)
    V = fe.VectorElement("P", mesh.ufl_cell(), 2)
    Q = fe.FiniteElement("P", mesh.ufl_cell(), 1)
    W = fe.FunctionSpace(mesh, V * Q)
    
    # Definir w0 y w
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
    
    # Campo inicial (opcional: cero o un pequeño vórtice)
    fe.assign(w0.sub(0), fe.Constant((0.0, 0.0)))  # velocidad inicial
    
    # Parámetros
    rho = 1.0
    dt_fe = fe.Constant(dt)
    f = fe.Constant((0.0, 0.0))
    
    # Semidiscreto: (u_n)/dt + (u_n·grad)u_n = -grad p + nu * lapl(u_n)
    # Forma variacional
    F_inertia = rho * fe.dot((u_n)/dt_fe, v_test)*fe.dx \
                + rho * fe.dot(fe.grad(u_n)*u_n, v_test)*fe.dx
    F_pressure = fe.dot(fe.grad(p), v_test)*fe.dx - fe.dot(q_test, fe.div(u))*fe.dx
    F_viscous = 2*nu*fe.inner(fe.sym(fe.grad(u)), fe.sym(fe.grad(v_test)))*fe.dx
    
    F_total = F_inertia + F_pressure + F_viscous - fe.dot(f, v_test)*fe.dx
    
    t = 0.0
    for step in range(num_steps):
        t += dt
        fe.solve(F_total == 0, w, bcs,
                 solver_parameters={"newton_solver":
                                    {"relative_tolerance": 1e-6}})
        w0.assign(w)
    
    u_sol, p_sol = w.split()
    
    print("Lid-Driven Cavity finalizado.")
    return mesh, u_sol, p_sol


    

if __name__ == "__main__":
    mesh_kova, u_kova, p_kova = main_kovasznay_flow(Re=40, N=32)
    mesh_tg, u_tg, p_tg = main_taylor_green_vortex(nu=0.01, U0=1.0,
                                                   Nx=32, Ny=32,
                                                   T=2.0, num_steps=50)
    mesh_stat_cav, u_stat_cav, p_stat_cav = main_static_cavity_flow(
        nu=0.01, N=32, T=1.0, num_steps=50
    )
    mesh_lid, u_lid, p_lid = main_lid_driven_cavity(
        nu=0.01, U0=1.0, N=32, T=2.0, num_steps=50
    )
