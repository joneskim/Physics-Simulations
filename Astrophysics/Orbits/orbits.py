import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that

# Define constants
G = 6.67408e-11 # m^3 kg^-1 s^-2
M_sun = 1.989e30 # kg
m_earth = 5.972e24 # kg
m_moon = 7.34767309e22 # kg

# Define initial conditions
r_earth = [1.4710e11, 0, 0] # m
v_earth = [0, 3.0287e4, 0] # m/s
r_moon = [1.4710e11 + 3.844e8, 0, 0] # m
v_moon = [0, 3.0287e4 + 1.022e3, 0] # m/s

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)

    def move(self, force, dt):
        self.velocity += force / self.mass * dt
        self.position += self.velocity * dt

    def distance(self, other):
        other.position = np.array(other.position, dtype=np.float64)
        return np.sqrt(np.sum((self.position - other.position) ** 2))

    def force(self, other):
        force = np.array([0., 0., 0.], dtype=np.float64)
        r = self.distance(other)
        force_mag = G * self.mass * other.mass / r ** 2
        force_unit = (other.position - self.position) / r
        force = force_mag * force_unit
        return force

class System:
    def __init__(self, bodies):
        self.bodies = bodies

    def evolve(self, dt):
        for i in range(len(self.bodies)):
            force = np.array([0., 0., 0.], dtype=np.float64)  # Ensure force is float64
            for j in range(len(self.bodies)):
                if i != j:
                    # Make sure the data types are correct (e.g., float64)
                    force += self.bodies[i].force(self.bodies[j]).astype(np.float64)
            self.bodies[i].move(force, dt)

# try to make a system with the sun, earth, and moon
sun = Body(M_sun, [0, 0, 0], [0, 0, 0])
earth = Body(m_earth, r_earth, v_earth)
moon = Body(m_moon, r_moon, v_moon)

system = System([sun, earth, moon])

# define time and number of steps
num_steps = 1000  # Increase the number of steps
t = np.linspace(0, 365*24*60*60, num_steps)

# define empty lists to store positions
x_earth = []
y_earth = []
z_earth = []
x_moon = []
y_moon = []
z_moon = []

# evolve the system
for i in range(len(t)):
    system.evolve(t[1] - t[0])  # Use the time step from the array
    x_earth.append(earth.position[0])
    y_earth.append(earth.position[1])
    z_earth.append(earth.position[2])
    x_moon.append(moon.position[0])
    y_moon.append(moon.position[1])
    z_moon.append(moon.position[2])

# animate the system

def animate(i):
    ax.clear()
    ax.set_xlim(-2e11, 2e11)
    ax.set_ylim(-2e11, 2e11)
    ax.set_zlim(-2e11, 2e11)
    ax.plot(x_earth[i], y_earth[i], z_earth[i], 'bo', markersize=20)
    ax.plot(x_moon[i], y_moon[i], z_moon[i], 'ro', markersize=5)
    # plot the sun
    ax.plot([0], [0], [0], 'yo', markersize=50)
    # the sun is not moving, so we don't need to plot it every time
    return ax


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# speed up the animation by decreasing the interval
ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=1)
ani.save('orbits.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()