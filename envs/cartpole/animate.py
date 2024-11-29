import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate(state_seq, filename):
    x, theta = [],[]
    for state in state_seq:
        theta.append(state.theta)
        x.append(state.x)
    x = jnp.array(x)
    theta = jnp.array(theta)

    # Parameters
    cart_width = 0.4
    cart_height = 0.2
    pole_length = 1.0

    # Sample inputs (replace these with your own data)
    # x = np.linspace(-2, 2, 100)  # Cart position
    # theta = np.pi / 6 * np.sin(np.linspace(0, 4 * np.pi, 100))  # Pole angle

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)

    # Cart
    cart = plt.Rectangle((x[0] - cart_width / 2, -cart_height / 2), cart_width, cart_height, color='black')
    ax.add_patch(cart)

    # Pole
    pole, = ax.plot([x[0], x[0] + pole_length * np.sin(theta[0])], [0, pole_length * np.cos(theta[0])], color='blue', lw=2)

    # Update function for animation
    def update(frame):
        cart.set_xy((x[frame] - cart_width / 2, -cart_height / 2))
        pole.set_data([x[frame], x[frame] + pole_length * np.sin(theta[frame])],
                    [0, pole_length * np.cos(theta[frame])])
        return cart, pole

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(x), blit=True, interval=50)

    # Display the animation
    if filename:
        
        ani.save(filename, writer='ffmpeg')

    else: 
        plt.show()
