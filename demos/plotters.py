import matplotlib.pyplot as plt

def trajectory_and_vector_subplots(solver):
    """ 
    """
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    axs[0].plot(solver.xs[:, 0], solver.xs[:, 2], '-k', linewidth=0.2)
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Z")
    
    # Plot the vector field.
    x, y, z = solver.xs.T
    force = solver.model(solver.xs)
    axs[1].quiver(x, z, force[:, 0], force[:, 2], color='k', alpha=0.7, width=0.005)
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Z")
    
    
def plot_trajectory_with_ref(solver, x_ref):
    """ 
    """
    plt.figure()
    plt.plot(x_ref[:, 0], x_ref[:, 2], 'k-', linewidth=0.2)
    plt.plot(solver.xs[:, 0], solver.xs[:, 2], 'or', markersize=2)
    
def plot_trajectory(solver):
    """ 
    """
    plt.figure()
    plt.plot(solver.xs[:, 0], solver.xs[:, 2], 'k-', alpha=0.7, linewidth=0.2)
    plt.xlabel("X")
    plt.ylabel("Z")
    
def plot_imagepoint_xref_subplots(solver, x_ref):
    """ 
    """
    # Plot the image point trajectory.
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    axs[0].plot(solver.xs[:, 0], solver.xs[:, 2], '-k', linewidth=0.2)
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Z")
        
    # Plot the reference solution trajectory.
    axs[1].plot(x_ref[:, 0], x_ref[:, 2], 'k-', linewidth=0.2)
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Z")
    