# -*- coding: utf-8 -*-

'''

N body gravitational simulation in 3D space

Plots the trajectories of the bodies, as well as conserved quantities for
validation

'''

# Importing all modules

import numpy as np

from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

# Save the trajectories plot?

SAVE_ANIMATION = False

# Values for G and the softening term eps

G = 1

eps = 0.0


# The following configuration of masses and velocities is chosen to form a stable
# figure-8 loop over the simulation time

mass = np.array([1.0, 1.0, 1.0])

pos = np.array([[ 0.84005076, -0.24308753, -0.48500218],
                [-0.84005076,  0.24308753,  0.48500218],
                [ 0.0,         0.0,         0.0       ]])

vel = np.array([[ 0.40373270,  0.43236573, -0.23310184],
                [ 0.40373270,  0.43236573, -0.23310184],
                [-0.80746540, -0.86473146,  0.46620369]])

# Number of bodies

N = len(mass)

d = pos.shape[1]

def all_accelerations(pos, mass):

    '''

    Calculates the acceleration of every body using pairwise force
    evaluation

    '''

    disp = pos[None, :, :] - pos[:, None, :]

    r2 = (disp**2).sum(axis = -1) + eps**2

    np.fill_diagonal(r2, 1.0)

    inv_r3 = r2**-1.5

    np.fill_diagonal(inv_r3, 0.0)

    weights = mass[None, :] * inv_r3

    acc = G * (weights[:, :, None] * disp).sum(axis = 1)

    return acc

acc = all_accelerations(pos, mass)

def energy(mass, pos, vel):

    '''

    Calculates the total kinetic energy of the system (kinetic + potential)

    '''

    ke = 0.5 * np.sum(mass * (vel**2).sum(axis = -1))

    disp = pos[None, :, :] - pos[:, None, :]

    r = np.sqrt((disp**2).sum(axis = -1) + eps**2)

    i, j = np.triu_indices(len(mass), k = 1)


    gpe = -G * np.sum(mass[i] * mass[j] / r[i, j])

    return ke + gpe

def angular_momentum(mass, pos, vel):

    '''

    Calculates the net angular momentum of the system

    '''

    L = (mass[:, None] * np.cross(pos, vel)).sum(axis = 0)

    return L

def derivatives(t, y):

    '''
    
    Returns the time derivatives of the state vector y
    
    '''

    p = y[:N*d].reshape(N, d)

    v = y[N*d:].reshape(N, d)

    a = all_accelerations(p, mass)

    return np.concatenate((v.ravel(), a.ravel()))

y0 = np.concatenate((pos.ravel(), vel.ravel()))

t_end = 10

t_eval = np.linspace(0, t_end, 5000)


sol = solve_ivp(derivatives, (0, t_end), y0,
                method = 'DOP853',
                t_eval = t_eval,
                rtol = 1e-10, atol = 1e-12)

time = sol.t

# History of all values for plotting later

history_pos = sol.y[:N*d].T.reshape(-1, N, d)

history_vel = sol.y[N*d:].T.reshape(-1, N, d)

history_energy = np.array([energy(mass, p, v)
                           for p, v in zip(history_pos, history_vel)])

history_momentum = np.array([angular_momentum(mass, p, v)
                             for p, v in zip(history_pos, history_vel)])

history_com = np.dot(mass, history_pos) / np.sum(mass)

# Body positions

fig = plt.figure()

ax = fig.add_subplot(projection = '3d')

N = len(mass)

for i in range(N):

    x = history_pos[:, i, 0]

    y = history_pos[:, i, 1]

    z = history_pos[:, i, 2]

    # Plotting trajectories

    ax.plot(x, y, z, linewidth = 0.5, alpha = 0.25, label = f'Body {i + 1}')

    # Marking starting positions

    ax.scatter(history_pos[0, i, 0],
               history_pos[0, i, 1],
               history_pos[0, i, 2],
               marker = 'x',
               s = 40)


# Setting plot limits (padding term gives boundary between data and axes)

padding = 0.1

xmin = history_pos[:, :, 0].min()

xmax = history_pos[:, :, 0].max()

ymin = history_pos[:, :, 1].min()

ymax = history_pos[:, :, 1].max()

zmin = history_pos[:, :, 2].min()

zmax = history_pos[:, :, 2].max()


ax.set_xlim(xmin - padding, xmax + padding)

ax.set_ylim(ymin - padding, ymax + padding)

ax.set_zlim(zmin - padding, zmax + padding)

# Plotting trajectories

ax.legend(loc='upper right')

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_zlabel('z')

ax.set_aspect('equal')

ax.set_title(f'{len(mass)}-body gravitational simulation')

ax.view_init(elev=20, azim = 45)


# Markers for moving bodies

bodies, = ax.plot([], [], [], 'o', color = 'black', markersize = 5)

com, = ax.plot([], [], [], 'x', color = 'red', markersize = 4)


def update(frame):

    '''
    
    Returns the x, y, and z position of each body and the systems centre
    of mass
    
    
    '''

    x_bod = history_pos[frame, :, 0]

    y_bod = history_pos[frame, :, 1]

    z_bod = history_pos[frame, :, 2]

    bodies.set_data(x_bod, y_bod)

    bodies.set_3d_properties(z_bod)


    x_com = history_com[frame, 0]

    y_com = history_com[frame, 1]

    z_com = history_com[frame, 2]

    com.set_data([x_com], [y_com])

    com.set_3d_properties([z_com])

    return com, bodies


ani = FuncAnimation(fig, update, frames = range(0, len(history_pos), 5),
                    interval = 20, blit = False)


if SAVE_ANIMATION:

    ani.save('trajectories.gif', writer = 'pillow', fps = 25, dpi = 80)

plt.show()

# Total system energy

plt.figure()

plt.plot(time, history_energy)

plt.xlabel('Time')

plt.ylabel('Total Energy')

plt.title('Total Energy vs Time')

plt.grid(True)

plt.savefig('total_energy.png', dpi = 150)

plt.show()

# Relative energy error

relative_error = (history_energy - history_energy[0]) / abs(history_energy[0])

plt.figure()

plt.plot(time, relative_error)

plt.xlabel('Time')

plt.ylabel('Relative energy error')

plt.title('Relative energy error vs time')

plt.grid(True)

plt.savefig('relative_energy_error.png', dpi = 150)

plt.show()

# Angular momentum plot

plt.figure()

plt.plot(time, history_momentum)

plt.xlabel('Time')

plt.ylabel('Angular Momentum')

plt.title('Angular Momentum vs Time')

plt.grid(True)

plt.savefig('angular_momentum.png', dpi = 150)

plt.show()
