"""
Herramientas de depuración para analizar problemas de visualización y dimensiones.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

def analyze_tensor_dimensions(input_tensor, output_tensor=None, y_pred=None, name="", verbose=True):
    """
    Analiza y muestra las dimensiones de los tensores para ayudar en la depuración.
    
    Args:
        input_tensor: Tensor de entrada
        output_tensor: Tensor de salida real (opcional)
        y_pred: Tensor de predicción (opcional)
        name: Nombre identificativo para el log
        verbose: Si es True, imprime información detallada
        
    Returns:
        dict: Diccionario con información del análisis
    """
    result = {
        "name": name,
        "input_tensor": {
            "shape": tuple(input_tensor.shape) if hasattr(input_tensor, "shape") else None,
            "dtype": str(input_tensor.dtype) if hasattr(input_tensor, "dtype") else None,
            "device": str(input_tensor.device) if hasattr(input_tensor, "device") else None,
        }
    }
    
    # Analizar tensor de salida real si está disponible
    if output_tensor is not None:
        result["output_tensor"] = {
            "shape": tuple(output_tensor.shape) if hasattr(output_tensor, "shape") else None,
            "dtype": str(output_tensor.dtype) if hasattr(output_tensor, "dtype") else None,
            "device": str(output_tensor.device) if hasattr(output_tensor, "device") else None,
        }
    
    # Analizar tensor de predicción si está disponible
    if y_pred is not None:
        result["y_pred"] = {
            "shape": tuple(y_pred.shape) if hasattr(y_pred, "shape") else None,
            "dtype": str(y_pred.dtype) if hasattr(y_pred, "dtype") else None,
            "device": str(y_pred.device) if hasattr(y_pred, "device") else None,
        }
        
        # Verificar compatibilidad de formas para comparación
        if output_tensor is not None:
            result["comparison"] = {
                "shapes_match": output_tensor.shape == y_pred.shape,
                "compatible_batch_size": output_tensor.shape[0] == y_pred.shape[0] if len(output_tensor.shape) > 0 and len(y_pred.shape) > 0 else False,
            }
    
    if verbose:
        print(f"\n=== ANÁLISIS DE TENSORES{f' - {name}' if name else ''} ===")
        print(f"• Input tensor: {result['input_tensor']['shape']} - {result['input_tensor']['dtype']} - {result['input_tensor']['device']}")
        
        if output_tensor is not None:
            print(f"• Output tensor: {result['output_tensor']['shape']} - {result['output_tensor']['dtype']} - {result['output_tensor']['device']}")
            
        if y_pred is not None:
            print(f"• Prediction tensor: {result['y_pred']['shape']} - {result['y_pred']['dtype']} - {result['y_pred']['device']}")
            
        if output_tensor is not None and y_pred is not None:
            comp = result["comparison"]
            print(f"• Formas coinciden: {'✓' if comp['shapes_match'] else '✗'}")
            print(f"• Tamaños de lote compatibles: {'✓' if comp['compatible_batch_size'] else '✗'}")
            
            if not comp['shapes_match'] and comp['compatible_batch_size']:
                print("  NOTA: Las dimensiones de salida no coinciden, pero los tamaños de lote sí.")
                print(f"  - Dimensión output: {output_tensor.shape[1:] if len(output_tensor.shape) > 1 else 1}")
                print(f"  - Dimensión predicción: {y_pred.shape[1:] if len(y_pred.shape) > 1 else 1}")
                
                if len(output_tensor.shape) == 2 and len(y_pred.shape) == 2:
                    if output_tensor.shape[1] == 1 and y_pred.shape[1] >= 3:
                        print("  DETECCIÓN: Parece que el output tiene 1 componente pero la predicción tiene 3 (u,v,p)")
                        print("  Esto es común en modelos de fluidos, considere usar solo la primera componente de y_pred para comparar")
    
    return result

def plot_mesh_grid_viability(tensor_1d, nx=None, ny=None, max_size=5000, output_path=None):
    """
    Verifica y visualiza si un tensor 1D se puede redimensionar a una malla 2D.
    
    Args:
        tensor_1d: Tensor unidimensional a analizar
        nx: Número de puntos en x (si es None, se calcula automáticamente)
        ny: Número de puntos en y (si es None, se calcula automáticamente)
        max_size: Tamaño máximo para visualizar (para tensores grandes)
        output_path: Ruta donde guardar el gráfico (opcional)
    """
    if isinstance(tensor_1d, torch.Tensor):
        tensor_1d = tensor_1d.detach().cpu().numpy()
    
    n_points = len(tensor_1d)
    
    if nx is None:
        # Calculamos nx como la raíz cuadrada aproximada
        nx = int(np.sqrt(n_points))
    
    if ny is None:
        # Calculamos ny para que nx*ny sea aproximadamente igual a n_points
        ny = n_points // nx
    
    print(f"\n=== ANÁLISIS DE VIABILIDAD DE MALLA 2D ===")
    print(f"• Tensor 1D de longitud: {n_points}")
    print(f"• Dimensiones propuestas: nx={nx}, ny={ny} (total: {nx*ny} puntos)")
    
    if nx * ny != n_points:
        print(f"⚠ ADVERTENCIA: Las dimensiones no son exactas. Faltan/sobran {abs(nx*ny - n_points)} puntos")
        
        # Sugerir dimensiones más apropiadas
        factors = []
        for i in range(1, int(np.sqrt(n_points)) + 1):
            if n_points % i == 0:
                factors.append((i, n_points // i))
        
        if factors:
            print("• Dimensiones exactas posibles:")
            for i, (f1, f2) in enumerate(factors):
                print(f"  {i+1}. {f1} × {f2} = {f1*f2}")
        else:
            print("• No se encontraron dimensiones exactas. Considere:")
            print(f"  1. Usar nx={nx}, ny={ny} y truncar/rellenar {abs(nx*ny - n_points)} puntos")
            print(f"  2. Cambiar a dimensiones aproximadas redondeadas: {int(np.sqrt(n_points))}×{int(np.sqrt(n_points))}")
    
    # Visualizar una muestra del tensor como malla, si es posible
    try:
        # Limitar tamaño para tensores grandes
        if n_points > max_size:
            sample_size = min(nx, max_size)
            sample_tensor = tensor_1d[:sample_size*sample_size]
            sample_nx = sample_size
            sample_ny = sample_size
            print(f"• Tensor demasiado grande para visualizar. Mostrando muestra {sample_nx}×{sample_ny}")
        else:
            sample_tensor = tensor_1d
            sample_nx = nx
            sample_ny = ny
            
        # Intentar reshape
        if sample_nx * sample_ny > len(sample_tensor):
            # Rellenar con ceros si falta
            missing = sample_nx * sample_ny - len(sample_tensor)
            sample_tensor = np.pad(sample_tensor, (0, missing), 'constant')
            
        grid = sample_tensor[:sample_nx*sample_ny].reshape(sample_nx, sample_ny)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(grid, cmap='viridis', aspect='auto')
        plt.colorbar(label='Valor')
        plt.title(f'Muestra del tensor como malla {sample_nx}×{sample_ny}')
        
        if output_path:
            plt.savefig(output_path)
            print(f"• Visualización guardada en: {output_path}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"⚠ Error al visualizar tensor como malla: {str(e)}")

if __name__ == "__main__":
    # Ejemplo de uso - creamos un tensor y lo analizamos
    input_tensor = torch.randn(4096, 2)  # Simula 64x64 puntos en coordenadas 2D
    output_tensor = torch.randn(4096, 1)  # Una sola componente (como un campo escalar)
    y_pred = torch.randn(4096, 3)  # Tres componentes (como u,v,p en fluidos)
    
    # Analizar dimensiones
    analyze_tensor_dimensions(input_tensor, output_tensor, y_pred, name="Ejemplo")
    
    # Verificar viabilidad de malla
    plot_mesh_grid_viability(output_tensor[:, 0])
