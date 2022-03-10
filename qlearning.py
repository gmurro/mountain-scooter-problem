import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from mountain_scooter import MountainScooter
matplotlib.use('Qt5Agg')
np.random.seed(9)


class QLearning(object):
    """
    QLearning is a class that implements the Q-Learning algorithm
    """
    def __init__(self, env, num_bins, alpha, epsilon, num_actions=3, discount_factor=1.0):
        """
        Initializes the Q-Learning algorithm
            :param env: Environment object to interact with
            :param num_bins: number of bins used to discretize the space
            :param num_actions: number of actions allowed (default: 3)
            :param alpha: Learning rate
            :param discount_factor: Value of discount factor (default: 1.0)
            :param epsilon: Starting value of epsilon for the epsilon-greedy policy (it will be decayed over time)
        """
        self.env = env
        self.num_bins = num_bins
        self.num_actions = num_actions
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.epsilon = epsilon
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

    def train(self, num_episodes, max_steps=100, n_episode_print_stats=100, n_episode_save_movie=10000, render_training=True, return_stats=False):
        """
        Implementation of the Q-learning algorithm.
        :param num_episodes: Number of episodes to use for training
        :param max_steps: Maximum number of steps per episode
        :param n_episode_print_stats: Number of episodes after which to print stats
        :param n_episode_save_movie: Number of episodes after which to save the movie
        :param return_stats: Boolean indicating whether to return the stats of the algorithm
        :return: Return the optimal Q-value matrix, the optimal policy is stored in the attribute policy
        """
        # initialize the value-state function to 0
        q_values = np.random.random((self.num_bins, self.num_bins, self.num_actions))

        # list of thresholds according to which packing in bins the velocity and the position
        velocity_state_array = np.linspace(-self.env.max_speed, self.env.max_speed, num=self.num_bins-1, endpoint=False)
        position_state_array = np.linspace(self.env.min_position, self.env.max_position, num=self.num_bins-1, endpoint=False)

        # store the stats of the algorithm in a dictionary
        stats = {
            'visit_counter_state': np.zeros((self.num_bins, self.num_bins, self.num_actions)),
            'cumulated_rewards': np.zeros(num_episodes),
            'episode_steps': np.zeros(num_episodes),
            'avg_td': np.zeros(num_episodes),
            'std_td': np.zeros(num_episodes)
        }

        if render_training:
            # Attaching 3D axis to the figure
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            # Setting the axes properties
            ax.set_xlim3d([self.env.min_position, self.env.max_position])
            ax.set_xlabel('Position')

            ax.set_ylim3d([-self.env.max_speed, self.env.max_speed])
            ax.set_ylabel('Velocity')

            ax.set_zlabel('V-value')

            ax.set_title('Q-learning - Mountain Car')

            # plot initial surface
            x = np.append(position_state_array, self.env.max_position)
            y = np.append(velocity_state_array, self.env.max_speed)

            X, Y = np.meshgrid(x, y)
            Z = np.max(q_values, axis=2)

            plot = [ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            cmap='viridis', edgecolor='none')]

            # Creating the Animation object
            ani = animation.FuncAnimation(fig, self.episodic_training, num_episodes, fargs=(max_steps, n_episode_print_stats, n_episode_save_movie,
                                        position_state_array, q_values, stats, velocity_state_array, plot, ax), interval=1, repeat=False)
            plt.show()
            ani.save("q-learning.gif", writer='imagemagick', fps=120)
        else:
            for episode in tqdm(range(num_episodes), desc='Compute optimal policy using Q-LEARNING algorithm', unit=' episodes'):
                self.episodic_training(episode, max_steps, n_episode_print_stats, n_episode_save_movie,
                                       position_state_array, q_values, stats, velocity_state_array)

        # update optimal policy greedified from the Q-values
        self.greedification(q_values)

        print("Saving the gif in: ./mountain_car.gif")
        self.env.render(show_plot=True)
        print("Complete!")

        return q_values if not return_stats else q_values, stats

    def episodic_training(self, episode, max_steps, n_episode_print_stats, n_episode_save_movie, position_state_array,
                          q_values, stats, velocity_state_array, plot=None, ax=None):
        """
        Implementation of the Q-learning algorithm for one episode.
        :param episode: Number of the episode
        :param max_steps: Maximum number of steps per episode
        :param velocity_state_array: Array of thresholds according to which packing in bins the velocity
        :param position_state_array: Array of thresholds according to which packing in bins the position
        :param q_values: Current Q-value matrix
        :param stats: Dictionary of the stats of the algorithm
        """
        decayed_value = self.epsilon * np.power(0.9, (episode / 2000))
        self.epsilon = decayed_value if decayed_value > 0.01 else 0.01

        # Reset and return the first observation
        velocity, position = self.env.reset(exploring_starts=True)

        # The observation is digitized, meaning that an integer corresponding
        # to the bin where the raw float belongs is obtained and use as replacement.
        state = (np.digitize(velocity, velocity_state_array), np.digitize(position, position_state_array))

        # compute the policy derived from the Q-values
        self.greedification(q_values)

        # store an array of temporal differencing
        td = []
        cumulated_reward = 0
        # for each step in the episode
        for step in range(max_steps):
            # choose an action based on the policy using ε-greedy
            if np.random.random() > 1 - self.epsilon:
                action = np.random.randint(low=0, high=self.num_actions)
            else:
                action = self.policy[state]

            # Move one step in the environment and get the new state and reward
            (new_velocity, new_position), reward, done = self.env.step(action)
            new_state = (np.digitize(new_velocity, velocity_state_array),
                         np.digitize(new_position, position_state_array))

            # Increment the visit counter
            stats['visit_counter_state'][state[0], state[1], action] += 1

            # update the Q-values
            td_t = reward + self.discount_factor * np.max(q_values[new_state[0], new_state[1]]) - q_values[
                state[0], state[1], action]
            td.append(td_t)

            decayed_alpha = self.alpha * np.power(0.9, (episode / 100))
            alpha = decayed_alpha if decayed_alpha > 0.001 else 0.001
            q_values[state[0], state[1], action] += alpha * td_t

            state = new_state
            cumulated_reward += reward

            # if the episode is done, break the loop
            if done: break


        # Store the data for statistics
        stats['avg_td'][episode] = np.average(td)
        stats['std_td'][episode] = np.std(td)
        stats['cumulated_rewards'][episode] = cumulated_reward
        stats['episode_steps'][episode] = step + 1

        if episode % n_episode_print_stats == 0:
            print("\nEpisode: " + str(episode + 1))
            print("Epsilon: " + str(self.epsilon))
            print("Episode steps: " + str(stats['episode_steps'][episode]))
            print("Cumulated Reward: " + str(stats['cumulated_rewards'][episode]))
            print("Policy matrix: \n" + self.policy_to_string())

        """if episode % n_episode_save_movie == 0:
            print("Saving the gif in: ./mountain_car.gif")
            self.env.render(file_path='./mountain_car.gif', mode='gif')
            print("Complete!")"""

        if plot is not None:
            # plot the figure
            x = np.append(position_state_array, self.env.max_position)
            y = np.append(velocity_state_array, self.env.max_speed)

            X, Y = np.meshgrid(x, y)
            Z = np.max(q_values, axis=2)

            ax.clear()
            plot[0].remove()
            plot[0] = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                                    cmap='viridis', edgecolor='none')

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
    optimizer = QLearning(env, num_bins=20, alpha=0.5, epsilon=0.4)
    q_valuse, stats = optimizer.train(num_episodes=num_episodes, return_stats=True)
    print("Policy matrix after " + str(num_episodes) + " episodes:")
    print(optimizer.policy_to_string())

    # plot statistics
    plot_episodes = range(0, num_episodes, 150)
    plot_stats(
        x=plot_episodes,
        y=[stats['episode_steps'][plot_episodes]],
        x_label="Episode",
        y_label="Steps",
        title="Number of steps per episode",
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