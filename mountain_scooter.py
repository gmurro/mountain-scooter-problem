import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.use('Qt5Agg')


class MountainScooter:
    """
    Class for defining the Mountain Scooter problem (analogy of the well known Mountain Car problem, https://en.wikipedia.org/wiki/Mountain_car_problem).
    Observation space is a 2-dim vector, where the 1st element represents the "scooter position" and the 2nd element represents the "scooter velocity".
    There are 3 discrete deterministic actions:
    - 0: Accelerate to the Left
    - 1: Don't accelerate
    - 2: Accelerate to the Right
    Reward: Reward of 0 is awarded if the agent reached the flag (position = 0.5) on top of the mountain.
    Reward of -1 is awarded if the position of the agent is less than 0.5.
    Starting State: The position of the scooter is assigned a uniform random value in [-0.6 , -0.4] if exploring_starts
    is True, otherwise is deterministically setted to -0.5.
    The starting velocity of the scooter is always assigned to 0.
    Episode Termination: The scooter position is more than 0.5. Episode length is greater than 150
    """

    def __init__(self, mass=0.5, friction=0.3, delta_t=0.1, initial_position=-0.5, min_position=-1.2, max_position=0.5, max_speed=1.8, goal_position=0.5):
        """
        Create a new mountain scooter object.
        It is possible to pass the parameter of the simulation.
            :param mass: the mass of the scooter (default 0.2)
            :param friction:  the friction in Newton (default 0.3)
            :param delta_t: the time step in seconds (default 0.1)
            :param initial_position: the initial position of the scooter (default -0.5)
            :param min_position: the minimum position of the scooter (default -1.2)
            :param max_position: the maximum position of the scooter (default 0.5)
            :param max_speed: the maximum speed of the scooter (default 0.07)
            :param goal_position: the position of the goal (default 0.5)
        """
        self.position_list = list()
        self.gravity = 9.8
        self.friction = friction
        self.delta_t = delta_t
        self.mass = mass
        self.position_t = initial_position
        self.velocity_t = 0.0
        self.min_position = min_position
        self.max_position = max_position
        self.max_speed = max_speed
        self.goal_position = goal_position

    def reset(self, exploring_starts=False, initial_position=-0.5):
        """
        It reset the scooter to an initial position [-1.2, 0.5]
            :param exploring_starts: if True a random position  between [-0.6, -0.4] is taken (default False)
            :param initial_position: the initial position of the scooter (requires exploring_starts=False)
            :return: it returns the velocity and the position the  of the scooter
        """
        if exploring_starts:
            initial_position = np.random.uniform(-0.6, -0.4)
        if initial_position < self.min_position:
            initial_position = self.min_position
        if initial_position > self.max_position:
            initial_position = self.max_position
        # clear the list of positions
        self.position_list = []
        self.position_t = initial_position
        self.velocity_t = 0.0
        self.position_list.append(initial_position)
        return self.velocity_t, self.position_t

    def step(self, action):
        """
        Perform one step in the environment following the action.
            :param action: an integer representing one of three actions [0, 1, 2] where 0=move_left, 1=do_not_move, 2=move_right
            :return: (velocity_t1, position_t1), reward, done
                      where reward is always negative but when the goal is reached, reward is zero and done is True
        """
        if action >= 3 or action < 0:
            raise ValueError("[MOUNTAIN SCOOTER][ERROR] The action value " + str(action) + " is out of range.")
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

        # Check the limit condition (scooter outside frame)
        if position_t1 == self.min_position and velocity_t1 < 0:
            velocity_t1 = 0

        # Assign the new position and velocity
        self.position_t = position_t1
        self.velocity_t = velocity_t1
        self.position_list.append(position_t1)

        # Reward and done when the scooter reaches the goal
        if position_t1 >= 0.5:
            reward = 0
            done = True

        # Return state_t1, reward, done
        return (velocity_t1, position_t1), reward, done

    def evaluate_policy(self, policy, num_bins, max_steps=100, exploring_starts=False):
        """
        Evaluate a policy by running it in the mountain scooter environment.
            :param policy: an array representing a policy that will be reshape as (num_bins x num_bins) matrix with velocity on rows and position on columns
            :param num_bins: the number of bins used to discretize the state space
            :param max_steps: the maximum number of steps in which the policy will be evaluated interacting with the environment. Default: 100
            :param exploring_starts: if True a random position between [-0.6, -0.4] as initial position is taken, otherwise it will be -0.5. Default: False
            :return: the total reward obtained by the policy
        """

        # reshape the policy to be (num_bins x num_bins) matrix with velocity on rows and position on columns
        policy_matrix = policy.reshape(num_bins, num_bins)

        # list of thresholds according to which packing in bins the velocity and the position
        velocity_state_array = np.linspace(-self.max_speed, self.max_speed, num=num_bins - 1, endpoint=False)
        position_state_array = np.linspace(self.min_position, self.max_position, num=num_bins - 1, endpoint=False)

        # Reset and return the first observation
        velocity, position = self.reset(exploring_starts=exploring_starts)

        # The observation is digitized, meaning that an integer corresponding to the bin where the raw float belongs
        state = (np.digitize(velocity, velocity_state_array), np.digitize(position, position_state_array))

        total_reward = 0
        step = 0
        done = False

        # Iterate until the maximum number of steps is reached or the goal is reached
        while not done and step < max_steps:

            # take the correspondent action from the policy
            action = int(policy_matrix[state])

            # Move one step in the environment and get the new state and reward
            (new_velocity, new_position), reward, done = self.step(action)
            state = (np.digitize(new_velocity, velocity_state_array), np.digitize(new_position, position_state_array))
            total_reward += reward
            step += 1
        return total_reward

    def render(self, file_path="./mountain_scooter.gif", figsize=(8, 6), show_plot=False):
        """
        Render the mountain scooter evolution.
            :param file_path: the path where the gif will be saved
            :param figsize: the size of the figure
            :param show_plot: if True the plot will be shown, otherwise it will be saved in the file_path
        """

        # Plot init
        fig, ax = plt.subplots(figsize=figsize)

        x = np.linspace(-1.3, 0.6, 100)
        y = np.sin(3 * x)

        # plot the sin wave
        ax.plot(x, y, color='#525252', linewidth=25.0, zorder=5)
        ax.plot(x, y, color='white', linestyle='dashed', linewidth=1.5, zorder=10)

        # plot the scooter
        img = plt.imread("assets/scooter.png")
        im = ax.imshow(img, zorder=10, aspect='auto')

        # add time annotation
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        _position_list = self.position_list
        _delta_t = self.delta_t

        def _init():
            ax.set_xlim(-1.3, 0.6)
            ax.set_ylim(-1.2, 1.2)
            ax.grid(True)
            return

        def _update(i):
            x = _position_list[i]
            l = x - 0.13
            r = x + 0.13
            b = np.sin(3*x) - 0.25
            t = np.sin(3*x) + 0.25
            im.set_extent([l, r, b, t])

            time_text.set_text("Time: " + str(np.round(i * _delta_t, 1)) + "s" + '\n' + "Frame: " + str(i))
            return

        ani = animation.FuncAnimation(fig, _update, frames=len(self.position_list), init_func=_init, blit=False, repeat=False)
        ani.save(file_path, writer='pillow')

        if show_plot:
            plt.show()
        else:
            ani.save(file_path, writer='pillow')
            print("Animation saved in " + file_path)

        # Clear the figure
        fig.clear()


def main():
    """
    Execute the environment going back and forth as long as the scooter velocity became negative.
    """
    #my_car = MountainCar(mass=0.45, friction=0.4, max_speed=1.8)

    # Initialize the environment
    env = MountainScooter(mass=0.70, friction=0.35, max_speed=2.8)

    total_reward = 0
    done = False
    step = 0
    max_steps = 100
    print("🛵 Starting the MOUNTAIN SCOOTER...")

    # scooter starts going back
    action = 0

    # Iterate until the maximum number of steps is reached or the goal is reached
    while not done and step < max_steps:
        (velocity, position), reward, done = env.step(action)

        if action == 0 and velocity > 0:
            action = 2
            print(f"\t➡️Go on! (Change direction at position {position:.2f})")
        elif action == 2 and velocity < 0:
            print(f"\t⬅️Go back! (Change direction at position {position:.2f})")
            action = 0
        total_reward += reward
        step += 1

    print("Finished after: " + str(step + 1) + " steps")
    print("Total reward: " + str(total_reward))

    env.render(show_plot=False)
    print("✅ Complete!")

if __name__ == "__main__":
    main()