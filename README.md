# Mountain Car Problem

### Description
The **mountain car** is a classic reinforcement learning problem. This problem was first described by [Andrew Moore in his PhD thesis](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.2654) and is defined as follows: a mountain car is moving on a two-hills landscape. The engine of the car isn't stronger enough and even at full throttle the car cannot accelerate up the steep climb. The driver has to find a way to reach the top of the hill. The only solution is to first move away from the goal and up the opposite slope on the left.
The reward in this problem is -1 on all time steps until the car moves past its goal position at the
top of the mountain, which ends the episode. The reward obtained is positive 1.0 only if the car reaches the goal. 

There are three possible **actions**: full *throttle forward* (+1), full *throttle reverse* (-1), and *zero throttle* (0). The car moves according to a simplified physics:

- the state space is defined by the position $x$ obtained through the function $sin(3x)$ in the domain $[-1.2, +0.5]$ (m) and the velocity $\hat{x}$ defined in the interval $[-1.5, +1.5]$ (m/s).

- for each action (left, no-op, right) there is a correspondent force applied to the car: $a= [-2.0, 0.0, +2.0]$

- the mass of the car is $m=0.2 kg$, the gravity is $g=9.8 m/s^2$, the friction is defined by $k=0.3 N$, and the time step is $\Delta t=0.1s$ .

- the position and velocity of the car at $t+1$ are updated using the following equations
  $$
  x_{t+1} = bound \left[x_{t}+\hat{x}_{t+1} \Delta t\right] 
  \\\hat{x}_{t+1} \doteq bound \left[\hat{x}_{t}+ \left( g \; m \cos \left(3 x_{t}\right) + \frac{a_t}{m} - k \; \hat{x}_t\right) \Delta t\right]
  $$
  where the $bound$ operation enforces $-1.2 \leq x_{t+1} \leq 0.5$ and $-1.5 \leq \hat{x}_{t+1} \leq 1.5$. 
  
  In addition, when $x_{t+1}$ reached the left bound, $\hat{x}_{t+1}$ was reset to zero. When it reached the right bound, the goal was reached and the episode was terminated.

There are three possible actions a=a= [-2.0, 0.0, +2.0] which are the values of the force applied to the car (left, no-op, right). The reward obtained is positive 1.0 only if the car reaches the goal. A negative cost of living of -0.01 is applied at every time step. The mass of the car is m=0.2 kgm=0.2 kg, the gravity is g=9.8 m/s2g=9.8 m/s2, the friction is defined by k=0.3 Nk=0.3 N, and the time step is Δt=0.1 sΔt=0.1 s. Given all these parameters the position and velocity of the car at t+1t+1 are updated using the following equations:

Its position, $x_t$, and velocity, $\hat{x}_ t$, are updated by:
$$
x_{t+1} \doteq  bound \left[x_{t}+\hat{x}_{t+1}\right] \\
\hat{x}_{t+1} \doteq bound \left[\hat{x}_{t}+0.001 A_{t}-0.0025 \cos \left(3 x_{t}\right)\right]
$$
where the bound operation enforces $-1.2 \leq x_{t+1} \leq 0.5$ and $-0.07 \leq \hat{x}_{t+1} \leq 0.07$. In addition, when
$x_{t+1}$ reached the left bound, $\hat{x}_{t+1}$ was reset to zero. When it reached the right bound, the goal was reached and the episode was terminated.

Each episode started from a random position $x_t \in [-0.6, -0.4)$ and zero velocity.

To convert the two continuous state variables to binary features, we used grid space discretizing each dimension in 12 bins.

The initial action values were all zero, which was optimistic (all true values are negative in this task), causing extensive exploration to occur even though the exploration parameter, ε, was 0.

### Particle Swarm Optimization

Developed in 1995 by Eberhart and Kennedy, PSO is a biologically inspired optimization routine designed to mimic birds flocking or fish schooling.

PSO is not guaranteed to find the global minimum, but it does a solid job in challenging, high dimensional, non-convex, non-continuous environments.

A **particle** is a individual, or in other words a population member *i*, and it is a 3-tuple $\langle \bar{x}_i, \bar{v}_i, \bar{b}_i \rangle$ of:

- a position vector (location) $\bar{x}_i$

- a velocity vector $\bar{v}_i$

- a best position vector of this particle in the past $\bar{b}_i$

Each triple is replaced by the mutant triple $\langle \bar{x}_i, \bar{v}_i, \bar{b}_i \rangle \rightarrow \langle \bar{x}_i', \bar{v}_i', \bar{b}_i' \rangle$ s.t.

- $\bar{x}_{i}^{\prime}=\bar{x}_i+\bar{v}_{i}^{\prime}$
- $\bar{v}_{i}^{\prime}=w \cdot \bar{v}_{i}+c_{1} R_{1} \cdot\left(\bar{b}_{i}-\bar{x}_{i}\right)+c_{2} R_{2} \cdot\left(\bar{p}-\bar{x}_{i}\right)$
- $\bar{b}_{i}^{\prime}=\left\{\begin{array}{cc}\bar{x}_{i}^{\prime} & \text { if } f\left(\bar{x}_{i}^{\prime}\right)<f\left(\bar{b}_{i}\right) \\ \bar{b}_{i} & \text { otherwise }\end{array}\right.$

where $w$ and $c_{i}$ are the weights and $R_{1}$ and *$R_2$* randomizer matrices and  $\bar{p}$  denotes the populations global best.

From the particle velocity equation, two important groups emerge:

1. social term: $c_{2} R_{2} \cdot\left(\bar{p}-\bar{x}_{i}\right)$
2. cognitive term: $c_{1} R_{1} \cdot\left(\bar{b}_{i}-\bar{x}_{i}\right)$

Instead the value of $w$ can control exploration and exploitation.

The main concept behind PSO, which is evident from the particle velocity equation above, is that there is a constant balance between three distinct forces pulling on each particle:

1. The particles previous velocity (inertia)
2. Distance from the individual particles’ best known position (cognitive force) 
3. Distance from the swarms best known position (social force)

These three forces are then weighted by $w$, $c_1$, $c_2$ and randomly perturbed by $R_1$ and $R_2$.

In vector form, these three forces can be seen below (vector magnitude represents the weight value of that specific force):

![Vector notation](https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/pso-vector-1.png)

### Nelder Mead

The **Nelder–Mead method** is *direct search* method (does not require any information about the gradient of the objective function) and is often applied to nonlinear optimization problems for which derivatives may not be known. 

The method in *n* dimensions maintains a set of *n* + 1 test points arranged as a simplex.  It then extrapolates the behavior of the objective function measured at each test point in order to find a new test point and to replace one of the old test points with the new one, and so the technique progresses. 

The behavior of this method is based on four operations: **reflection**, **expansion**, **contraction**, **shrinkage**. These are usually tried in this order. Basically, what they do is the following:

- **Reflection**: tries moving the simplex away from the sub-optimal area, by computing the reflected point through the centroid of the other points, excluding the worst one;
- **Expansion**: if reflection generated a point which is better than the current best, the point is moved along the same direction for a bit further;
- **Contraction**: if, on the contrary, the reflected point did not improve the result, we'll want to *contract* the simplex towards the best point;
- **Shrinkage**: if none of the previous did work, we'll want to re-calculate all the points except for the best one, computing them through the best one's coordinates.

https://mpatacchiola.github.io/blog/2017/08/14/dissecting-reinforcement-learning-6.html

https://github.com/mpatacchiola/dissecting-reinforcement-learning



Useful

https://github.com/viniciusenari/Q-Learning-and-SARSA-Mountain-Car-v0/tree/2d56eaba7f70c2ede4fc3bd55b7ce55b9006feba
