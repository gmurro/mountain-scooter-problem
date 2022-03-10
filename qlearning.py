import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mountain_scooter import MountainScooter
matplotlib.use('Qt5Agg')
np.random.seed(9)


class QLearning(object):
    """
    QLearning is a class that implements the Q-Learning off-policy TD control algorithm
    by Sutton and Barto, Reinforcement Learning: An Introduction (2018, http://incompleteideas.net/book/ebook/node65.html)
    """
    def __init__(self, env, num_bins, alpha, epsilon, decay_factor=200, num_actions=3, discount_factor=1.0, verbose=False):
        """
        Initializes the Q-Learning algorithm
            :param env: Environment object to interact with
            :param num_bins: number of bins used to discretize the space
            :param alpha: Starting value of alpha to compute the learning rate that will be decayed over time
            :param epsilon: Starting value of epsilon for the epsilon-greedy policy (it will be decayed over time)
            :param decay_factor: Denominator used to decay epsilon and alpha over time (default: 200)
            :param num_actions: number of actions allowed (default: 3)
            :param discount_factor: Value of discount factor (default: 1.0)
            :param verbose: If True, print the progress of the algorithm. Default value is False.
        """
        self.env = env
        self.num_bins = num_bins
        self.num_actions = num_actions
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.verbose = verbose
        self.policy = self.initialize_policy_grid()  # Initialize policy randomly

    def policy_to_string(self):
        """
        Convert the policy matrix to string using specific symbol. O noop, < left, > right.
            :return: string representation of the policy
        """
        counter = 0
        shape = self.policy.shape
        policy_string = ""
        for row in range(shape[0]):
            for col in range(shape[1]):
                if self.policy[row, col] == 0:
                    policy_string += " <  "
                elif self.policy[row, col] == 1:
                    policy_string += " O  "
                elif self.policy[row, col] == 2:
                    policy_string += " >  "
                counter += 1
            policy_string += '\n'
        return policy_string

    def initialize_policy_grid(self):
        """
        Creates a grid of random policies based on the number of bins used to discretize the space.
            :return: matrix of policies for each state
        """
        # Random policy as a square matrix of size (num_bins x num_bins)
        # Three possible actions represented by three integers
        policy = np.random.randint(low=0, high=self.num_actions, size=(self.num_bins, self.num_bins))
        return policy

    def greedification(self, q_values):
        """
        Greedification of the value-state function and compute the corresponding policy matrix.
            :param q_values: Value-state function matrix
        """
        # greedify the value-state function
        for i in range(self.policy.shape[0]):
            for j in range(self.policy.shape[1]):
                self.policy[i, j] = np.argmax(q_values[i, j])

    def train(self, num_episodes, max_steps=100, n_episode_print_stats=500, n_episode_render_env=None, render_training=False, return_stats=False):
        """
        Implementation of the Q-learning algorithm.
            :param num_episodes: Number of episodes to use for training
            :param max_steps: Maximum number of steps per episode
            :param n_episode_print_stats: Number of episodes after which to print stats
            :param n_episode_render_env: Number of episodes after which to render the scooter acting in the environment with the policy learned so far. If None, no rendering will be done.
            :param render_training: Boolean to indicate whether to render the evolution of the training or not at the end of the training. Default value is True.
            :param return_stats: Boolean indicating whether to return the stats of the algorithm
            :return: Return the optimal Q-value matrix, the optimal policy is stored in the attribute policy
        """
        # initialize the value-state function randomly
        q_values = np.random.random((self.num_bins, self.num_bins, self.num_actions))

        # list of thresholds according to which packing in bins the velocity and the position
        velocity_state_array = np.linspace(-self.env.max_speed, self.env.max_speed, num=self.num_bins-1, endpoint=False)
        position_state_array = np.linspace(self.env.min_position, self.env.max_position, num=self.num_bins-1, endpoint=False)

        # store the stats of the algorithm in a dictionary
        stats = {
            'v_values_steps': np.zeros((self.num_bins, self.num_bins, num_episodes)),
            'total_rewards': np.zeros(num_episodes),
            'avg_td': np.zeros(num_episodes),
            'std_td': np.zeros(num_episodes)
        }

        for episode in range(num_episodes):
            # compute the epsilon decayed value for the current episode
            # to explore more at the beginning and to exploit at the end
            decayed_value = self.epsilon * np.power(0.9, (episode / self.decay_factor))
            epsilon = decayed_value if decayed_value > 0.01 else 0.01

            # Reset and return the first observation
            velocity, position = self.env.reset()

            # The observation is digitized, meaning that an integer corresponding
            # to the bin where the raw float belongs is obtained and use as replacement.
            state = (np.digitize(velocity, velocity_state_array), np.digitize(position, position_state_array))

            # compute the policy derived from the Q-values
            self.greedification(q_values)

            # store an array of temporal differencing
            td = []

            total_reward = 0
            step = 0
            done = False

            # Iterate until the maximum number of steps is reached or the goal is reached
            while not done and step < max_steps:
                # choose an action based on the policy using epsilon-greedy strategy
                if np.random.random() > 1 - epsilon:
                    action = np.random.randint(low=0, high=self.num_actions)
                else:
                    action = self.policy[state]

                # Move one step in the environment and get the new state and reward
                (new_velocity, new_position), reward, done = self.env.step(action)
                new_state = (np.digitize(new_velocity, velocity_state_array),
                             np.digitize(new_position, position_state_array))

                # compute the temporal difference
                td_t = reward + self.discount_factor * np.max(q_values[new_state[0], new_state[1]]) - q_values[state[0], state[1], action]
                td.append(td_t)

                # update the Q-values
                decayed_value = self.alpha * np.power(0.9, (episode / self.decay_factor))
                lr = decayed_value if decayed_value > 0.01 else 0.01
                q_values[state[0], state[1], action] += lr * td_t

                state = new_state
                total_reward += reward
                step += 1

            # Store the data for statistics
            stats['v_values_steps'][:, :, episode] = np.max(q_values, axis=2)   # store the V-values
            stats['avg_td'][episode] = np.average(td)
            stats['std_td'][episode] = np.std(td)
            stats['total_rewards'][episode] = total_reward

            if episode % n_episode_print_stats == 0:
                print(f"🚀 Performing episode {episode + 1}:\n\t"
                      f"🏆 Total reward={total_reward}\n\t"
                      f"📉 Epsilon={epsilon:.2f}\t"
                      f"Learning rate={lr:.2f}\n\t"
                      f"📊 Average TD={stats['avg_td'][episode]:.2f}\t"
                      f"Standard dev TD={stats['std_td'][episode]:.2f}")
            elif self.verbose:
                print(f"🚀 Performing episode {episode+1}:\n\t"
                      f"🏆 Total reward={total_reward}")

            if n_episode_render_env is not None and episode % n_episode_render_env == 0:
                # Render the scooter acting in the environment with the policy learned so far
                print("🚧 Rendering the scooter acting in the environment with the policy learned so far...")
                self.env.render(show_plot=True)

        # update optimal policy greedified from the Q-values
        self.greedification(q_values)

        # plot the V-values evolution over time
        if render_training:
            print("\n🚧 Rendering the V-values evolution over time...")
            self.render_training(x = np.append(position_state_array, self.env.max_position)
                                 , y = np.append(velocity_state_array, self.env.max_speed)
                                 , v_values=stats['v_values_steps']
                                 , show_plot=True)
        return q_values if not return_stats else q_values, stats

    def render_training(self, x, y, v_values, file_path="./q-learning_training.gif", show_plot=False, figsize=(10, 8)):
        """
        Render evolution of the v-values during the training.
            :param x: x-axis values
            :param y: y-axis values
            :param v_values: V-values matrices obtained at each episode during the training
            :param file_path: the path where the gif will be saved
            :param show_plot: if True the plot will be shown, otherwise it will be saved in the file_path
            :param figsize: the size of the figure
        """

        # Attaching 3D axis to the figure
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection='3d')

        # plot initial surface
        X, Y = np.meshgrid(x, y)
        Z = v_values[:,:,0]
        plot = [ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')]

        def _init():
            # Setting the axes properties
            ax.set_xlim3d([self.env.min_position, self.env.max_position])
            ax.set_xlabel('Position')

            ax.set_ylim3d([-self.env.max_speed, self.env.max_speed])
            ax.set_ylabel('Velocity')

            ax.set_zlabel('V-value')
            return

        def _update(i):
            Z = v_values[:,:,i]

            ax.clear()
            plot[0].remove()
            plot[0] = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            return

        # Creating the Animation object
        ani = animation.FuncAnimation(fig, _update, frames=v_values.shape[2], init_func=_init, blit=False, repeat=True)
        if show_plot:
            plt.show()
        else:
            ani.save(file_path, writer='pillow')
            print("Animation saved in " + file_path)

        # Clear the figure
        fig.clear()


def plot_stats(x, y, x_label, y_label,  labels, title="", std_y=None, y_scale="linear", figsize=(10,5)):
    """
    Plot the given data.
        :param x: Array of x values.
        :param y:  List of array of y values.
        :param x_label:  Label of the x axis.
        :param y_label: Label of the y axis.
        :param title: Title of the plot.
        :param labels: List of labels
        :param std_y: List of standard deviation of the y values.
        :param y_scale: Scale of the y axis.
    """
    plt.figure(figsize=figsize)
    for i in range(len(y)):
        plt.plot(x, y[i], linestyle='solid', lw=2, label=labels[i])
        if std_y:
            plt.fill_between(x, y[i]-std_y[i], y[i]+std_y[i], alpha=0.3)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(linestyle='--', linewidth=1)
    plt.yscale(y_scale)
    plt.show()


def main():
    env = MountainScooter(mass=0.4, friction=0.3, max_speed=1.8)
    
    # -------------------------------------------------------------------------------#
    # ----------------------------- Q-learning method -------------------------------#
    # -------------------------------------------------------------------------------#
    num_episodes = 10000
    optimizer = QLearning(env
                          , num_bins=20
                          , alpha=0.1
                          , epsilon=0.4
                          , verbose=False)
    q_valuse, stats = optimizer.train(num_episodes=num_episodes
                                      , max_steps=100
                                      , n_episode_print_stats=500
                                      , n_episode_render_env=5000
                                      , render_training=True
                                      , return_stats=False)

    # plot statistics
    plot_episodes = range(0, num_episodes, 100)
    plot_stats(
        x=plot_episodes,
        y=[stats['total_rewards'][plot_episodes]],
        x_label="Episode",
        y_label="Total reward",
        title="Total reward per episode",
        labels=["Q-learning"],
    )
    plot_stats(
        x=plot_episodes,
        y=[stats['avg_td'][plot_episodes]],
        x_label="Episode",
        y_label="Avg TD error",
        title="Average Temporal Differencing error per episode",
        labels=["Q-learning"],
        std_y=[stats['std_td'][plot_episodes]]
    )


if __name__ == "__main__":
    main()