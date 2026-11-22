# 3D N-Body Gravitational Simulation

A 3D N-body gravitational simulation in Python

## Background

This project simulates N point masses interacting through Newtonian gravity:

- **Force calculation**: The force acting between every pair interaction is computed directly and vectorised with NumPy for efficiency
  
- **Adaptive integration**: The time step size is chosen automatically by SciPy's `solve_ivp`. The size of the step decreases for close encounters to improve accuracy
- **Validation**: As there is no general analytical solution for N ≥ 3, the simulation is validated by keeping track of conserved quantities (total energy, angular momentum vector, centre-of-mass motion) a


## Results

- **Trajectories**: 3D animation of the 3-body system, with parameters currently set to form a stable figure of eight loop

  *Trajectories:*
  ![]()

- **Relative energy error**: For the 3-body system demonstrated in the code, the simulation reaches a maximum relative energy error of ~8e-10

  *Relative energy error:*
   ![]()

- **Angular momentum conservation**: All components of the angular momentum vector remain constant to within the tolerances of the solver, and the   centre of mass stays fixed

  *Angular momentum conservation:*
   ![]()

## Method

Each body follows Newtonian gravity, with a softening term $\epsilon$ added to avoid division by zero

$$\mathbf{F}_{ij} = G \, \frac{m_i m_j \, (\mathbf{r}_j - \mathbf{r}_i)}{\left(|\mathbf{r}_j - \mathbf{r}_i|^2 + \epsilon^2\right)^{3/2}}, \qquad \hat{\mathbf{r}}_{ij} = \frac{\mathbf{r}_j - \mathbf{r}_i}{|\mathbf{r}_j - \mathbf{r}_i|}$$

- The displacement vectors for all pairs are constructed at once as an (N, N, 3) array, then collapsed to give the acceleration for each body
- Positions and velocities are flattened into a single state vector so that 'solve_ivp' can integrate them, with the derivative function returning the velocity and acceleration

**Integrator**

-**DOP583**: An 8th-order Runge-Kutte method. Each step is accepted or rejected against the relative and absolute tolerances, and the size of the next step is chosen accordingly

## Limitations

- **O(N²) Scaling**: Direct summation becomes highly inefficient for large values of N
- **Energy conservation**: The chosen integrator does not conserve energy perfectly, causing the total energy to drift over long simulation timescales
- **Close encounters**: A softening term must be added to the denominator to ensure that no division by zero occurs, despite this altering the physics slightly. Additionally the bodies are treated as point masses, and so can never collide with each other
- **Newtonian gravity**: No relativistic effects are accounted for

## Setup

```bash
pip install numpy scipy matplotlib
```

## Run

```bash
python n_body_sim.py
```

## Possible Extensions

- Implement Numba to improve speed
- Scale to larger N using a Barnes-Hut tree algorithm
- Rewrite some sections of code in C++ to improve speed
