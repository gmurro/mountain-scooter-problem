import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class MountainCar:
    """
    Class for defining the mountain car problem.
    Observation space is a 2-dim vector, where the 1st element represents the "car position" and the 2nd element represents the "car velocity".
    There are 3 discrete deterministic actions:
    - 0: Accelerate to the Left
    - 1: Don't accelerate
    - 2: Accelerate to the Right
    Reward: Reward of 0 is awarded if the agent reached the flag
    (position = 0.5) on top of the mountain. Reward of -1 is awarded if the position of the agent is less than 0.5.
    Starting State: The position of the car is assigned a uniform random value in [-0.6 , -0.4]. The starting velocity of the car is always assigned to 0.
    Episode Termination: The car position is more than 0.5. Episode length is greater than 200
    """
    def __init__(self, mass=0.5, friction=0.3, delta_t=0.1, initial_position=-0.5, min_position=-1.2, max_position=0.5, max_speed=1.8, goal_position=0.5):
        """
        Create a new mountain car object.

        It is possible to pass the parameter of the simulation.
        :param mass: the mass of the car (default 0.2)
        :param friction:  the friction in Newton (default 0.3)
        :param delta_t: the time step in seconds (default 0.1)
        :param initial_position: the initial position of the car (default -0.5)
        :param min_position: the minimum position of the car (default -1.2)
        :param max_position: the maximum position of the car (default 0.5)
        :param max_speed: the maximum speed of the car (default 0.07)
        :param goal_position: the position of the goal (default 0.5)
        """
        self.position_list = list()
        self.gravity = 9.8
        self.friction = friction
        self.delta_t = delta_t   # time step
        self.mass = mass         # the mass of the car
        self.position_t = initial_position   # initial position
        self.velocity_t = 0.0    # initial velocity
        self.min_position = min_position
        self.max_position = max_position
        self.max_speed = max_speed
        self.goal_position = goal_position

    def reset(self, exploring_starts=False, initial_position=-0.5):
        """
        It reset the car to an initial position [-1.2, 0.5]

        :param exploring_starts: if True a random position is taken (default False)
        :param initial_position: the initial position of the car (requires exploring_starts=False)
        :return: it returns the velocity and the initial position of the car
        """
        if exploring_starts:
            initial_position = np.random.uniform(-0.6, -0.4)
        if initial_position < self.min_position:
            initial_position = self.min_position
        if initial_position > self.max_position:
            initial_position = self.max_position
        self.position_list = []  # clear the list
        self.position_t = initial_position
        self.velocity_t = 0.0
        self.position_list.append(initial_position)
        return self.velocity_t, self.position_t

    def step(self, action):
        """
        Perform one step in the environment following the action.

        :param action: an integer representing one of three actions [0, 1, 2]
                       where 0=move_left, 1=do_not_move, 2=move_right
        :return: (postion_t1, velocity_t1), reward, done
                  where reward is always negative but when the goal is reached, reward is positive and done is True
        """
        if action >= 3:
            raise ValueError("[MOUNTAIN CAR][ERROR] The action value " + str(action) + " is out of range.")
        done = False

        # each state, except for the reached goal, have a negative reward
        reward = -1

        # action used to update the velocity at each time step
        action_list = [-1.0, 0.0, +1.0]
        action_t = action_list[action]

        # equations of motion
        velocity_t1 = self.velocity_t + \
                      (-self.gravity * self.mass * np.cos(3 * self.position_t)
                       + (action_t / self.mass)
                       - (self.friction * self.velocity_t)) * self.delta_t
        velocity_t1 = np.clip(velocity_t1, -self.max_speed, self.max_speed)

        position_t1 = self.position_t + (velocity_t1 * self.delta_t)
        position_t1 = np.clip(position_t1, self.min_position, self.max_position)

        # Check the limit condition (car outside frame)
        if position_t1 == self.min_position and velocity_t1 < 0:
            velocity_t1 = 0

        # Assign the new position and velocity
        self.position_t = position_t1
        self.velocity_t = velocity_t1
        self.position_list.append(position_t1)

        # Reward and done when the car reaches the goal
        if position_t1 >= 0.5:
            reward = +1.0
            done = True

        # Return state_t1, reward, done
        return (position_t1, velocity_t1), reward, done

    def getImage(self, path='assets/scooter.png'):
        return OffsetImage(plt.imread(path, format="jpg"), zoom=.1)

    def render(self, file_path='./mountain_car.mp4', mode='mp4'):
        """ When the method is called it saves an animation
        of what happened until that point in the episode.
        Ideally it should be called at the end of the episode,
        and every k episodes.

        ATTENTION: It requires avconv and/or imagemagick installed.
        @param file_path: the name and path of the video file
        @param mode: the file can be saved as 'gif' or 'mp4'
        """
        # Plot init
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.2, 0.5), ylim=(-1.1, 1.1))
        ax.grid(False)  # disable the grid
        x_sin = np.linspace(start=-1.2, stop=0.5, num=100)
        y_sin = np.sin(3 * x_sin)

        ax.plot(x_sin, y_sin)  # plot the sine wave

        dot, = ax.plot([], [], 'ro')
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        _position_list = self.position_list
        _delta_t = self.delta_t

        def _init():
            dot.set_data([], [])
            time_text.set_text('')
            return dot, time_text

        def _animate(i):
            x = _position_list[i]
            y = np.sin(3 * x)
            dot.set_data(x, y)
            time_text.set_text("Time: " + str(np.round(i * _delta_t, 1)) + "s" + '\n' + "Frame: " + str(i))
            return dot, time_text

        ani = animation.FuncAnimation(fig, _animate, np.arange(1, len(self.position_list)),
                                      blit=True, init_func=_init, repeat=False)

        if mode == 'gif':
            ani.save(file_path, writer='imagemagick', fps=int(1 / self.delta_t))
        elif mode == 'mp4':
            ani.save(file_path, fps=int(1 / self.delta_t), writer='avconv', codec='libx264')
        # Clear the figure
        fig.clear()
        plt.close(fig)