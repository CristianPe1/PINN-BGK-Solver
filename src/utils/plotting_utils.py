import numpy as np
import matplotlib.pyplot as plt
import os

def plot_taylor_green_flow(data, output_dir):
    """
    Plot Taylor-Green vortex flow safely, avoiding streamplot issues
    
    Args:
        data (dict): Data dictionary with grid, u, v, p fields
        output_dir (str): Directory to save visualizations
        
    Returns:
        list: List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    files = []
    
    # Extract data
    X = data['grid']['X']
    Y = data['grid']['Y']
    T = data['grid']['T']
    u = data['grid']['u']
    v = data['grid']['v']
    p = data['grid']['p']
    
    # Select times to plot
    nt = T.shape[2]
    times_to_plot = [0, nt//2, -1]
    
    for t_idx in times_to_plot:
        # Extract 2D slices for this time
        x_slice = X[:,:,t_idx]
        y_slice = Y[:,:,t_idx]
        u_slice = u[:,:,t_idx]
        v_slice = v[:,:,t_idx]
        p_slice = p[:,:,t_idx]
        
        # Calculate velocity magnitude
        speed = np.sqrt(u_slice**2 + v_slice**2)
        
        # Velocity plot
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(x_slice, y_slice, speed, 50, cmap='viridis')
        plt.colorbar(contour, label='Velocity Magnitude')
        
        # Add velocity vectors
        stride = max(1, min(x_slice.shape[0], x_slice.shape[1]) // 20)
        plt.quiver(x_slice[::stride,::stride], y_slice[::stride,::stride], 
                  u_slice[::stride,::stride], v_slice[::stride,::stride],
                  color='white', scale=20)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Taylor-Green Velocity at t = {T[0,0,t_idx]:.3f}')
        
        vel_file = os.path.join(output_dir, f'taylor_green_vel_t{t_idx}.png')
        plt.savefig(vel_file, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(vel_file)
        
        # Pressure plot with velocity overlay
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(x_slice, y_slice, p_slice, 50, cmap='coolwarm')
        plt.colorbar(contour, label='Pressure')
        
        # Create a denser quiver plot instead of streamplot
        dense_stride = max(1, min(x_slice.shape[0], x_slice.shape[1]) // 30)
        plt.quiver(x_slice[::dense_stride,::dense_stride], y_slice[::dense_stride,::dense_stride], 
                  u_slice[::dense_stride,::dense_stride], v_slice[::dense_stride,::dense_stride],
                  color='white', scale=25, width=0.002, alpha=0.8)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Taylor-Green Pressure at t = {T[0,0,t_idx]:.3f}')
        
        press_file = os.path.join(output_dir, f'taylor_green_press_t{t_idx}.png')
        plt.savefig(press_file, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(press_file)
    
    return files
