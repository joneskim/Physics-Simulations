import numpy as np
import matplotlib.pyplot as plt

class Object:
    def __init__(self,y_initial, mass, k):
        self.y_initial = y_initial
        self.mass = mass
        self.k = k


class OscillatorSystem:
    def __init__(self, Object):
        self.Object = Object

    def position(self, t):
        return self.Object.y_initial * np.cos(np.sqrt(self.Object.k / self.Object.mass) * t)

    def velocity(self, t):
        return -self.Object.y_initial * np.sqrt(self.Object.k / self.Object.mass) * np.sin(np.sqrt(self.Object.k / self.Object.mass) * t)

    def kinetic_energy(self, t):
        v = self.velocity(t)
        T = (0.5) * self.Object.mass * (v ** 2)
        return T

    def potential_energy(self, t):
        y = self.position(t)
        U = (0.5) * self.Object.k * (y ** 2)
        return U

    def total_energy(self, t):
        return self.kinetic_energy(t) + self.potential_energy(t)

    def period(self):
        return 2 * np.pi * np.sqrt(self.Object.mass / self.Object.k)
    
    def evolve(self, t):
        return self.position(t), self.velocity(t)


# Define initial conditions
y_initial = 1 # m
mass = 1 # kg
k = 1 # N/m

# Create an object
object = Object(y_initial, mass, k)


# Create an oscillator system
oscillator_system = OscillatorSystem(object)

# Define time and number of steps
num_steps = 1000
t = np.linspace(0, 10, num_steps)

# Define empty lists to store positions
y = []
v = []
T = []
T_total = []

# Evolve the system
for i in range(len(t)):
    y.append(oscillator_system.position(t[i]))
    v.append(oscillator_system.velocity(t[i]))
    T.append(oscillator_system.kinetic_energy(v[i]))
    T_total.append(oscillator_system.total_energy(t[i]))

# Plot the results
plt.plot(t, y, label='Position')
plt.plot(t, v, label='Velocity')
plt.plot(t, T, label='Kinetic Energy')
plt.plot(t, T_total, label='Total Energy')
plt.xlabel('Time (s)')
plt.ylabel('Position (m), Velocity (m/s), Kinetic Energy (J)')
plt.title('Simple Harmonic Motion')
plt.legend()
plt.show()


class DampenedSimpleHarmonicOscillator:

    def __init__(self, Object, gamma):
        self.Object = Object
        self.gamma = gamma

    def position(self, t):
        return self.Object.y_initial * np.exp(-self.gamma * t) * np.cos(np.sqrt(self.Object.k / self.Object.mass) * t)

    def velocity(self, t):
        return -self.Object.y_initial * np.exp(-self.gamma * t) * (self.gamma * np.cos(np.sqrt(self.Object.k / self.Object.mass) * t) + np.sqrt(self.Object.k / self.Object.mass) * np.sin(np.sqrt(self.Object.k / self.Object.mass) * t))

    def kinetic_energy(self, t):
        v = self.velocity(t)
        T = (0.5) * self.Object.mass * (v ** 2)
        return T

    def potential_energy(self, t):
        y = self.position(t)
        U = (0.5) * self.Object.k * (y ** 2)
        return U

    def total_energy(self, t):
        return self.kinetic_energy(t) + self.potential_energy(t)

    def period(self):
        return 2 * np.pi * np.sqrt(self.Object.mass / self.Object.k)

    def evolve(self, t):
        return self.position(t), self.velocity(t)


# Define initial conditions
y_initial = 10 # m
mass = 2 # kg
k = 1 # N/m
gamma = 0.1 # kg/s

# Create an object
object = Object(y_initial, mass, k)

# Create a dampened simple harmonic oscillator
dampened_simple_harmonic_oscillator = DampenedSimpleHarmonicOscillator(object, gamma)

# Define time and number of steps
num_steps = 1000
t = np.linspace(0, 100, num_steps)

# Define empty lists to store positions
y = []
v = []
T = []
T_total = []

# Evolve the system
for i in range(len(t)):
    y.append(dampened_simple_harmonic_oscillator.position(t[i]))
    v.append(dampened_simple_harmonic_oscillator.velocity(t[i]))
    T.append(dampened_simple_harmonic_oscillator.kinetic_energy(t[i]))
    T_total.append(dampened_simple_harmonic_oscillator.total_energy(t[i]))

# Plot the results
plt.plot(t, y, label='Position')
plt.plot(t, v, label='Velocity')
plt.plot(t, T, label='Kinetic Energy')
plt.plot(t, T_total, label='Total Energy')
plt.xlabel('Time (s)')
plt.ylabel('Position (m), Velocity (m/s), Kinetic Energy (J)')
plt.title('Dampened Simple Harmonic Motion')
plt.legend()
plt.show()


# animate the result 
import matplotlib.animation as animation

fig = plt.figure()
ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = np.linspace(-1, 1, 1000)
    y = oscillator_system.position(t[i])
    line.set_data(x, y)
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, interval=20, blit=True)

plt.show()

# animate the dampened result
import matplotlib.animation as animation

fig = plt.figure()
ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-15, 15))
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = np.linspace(-1, 1, 1000)
    y = dampened_simple_harmonic_oscillator.position(t[i])
    line.set_data(x, y)
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, interval=20, blit=True)

plt.show()
