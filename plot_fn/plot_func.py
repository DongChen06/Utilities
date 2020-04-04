import numpy as np
import matplotlib.pyplot as plt


def smooth(x, timestamps=9):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - timestamps)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y


def plot(logs_dir):
    episode_rewards = np.load(logs_dir + '{}'.format('epoch_rewards') + '.npy')
    eval_rewards = np.load(logs_dir + '{}'.format('eval_rewards') + '.npy')

    epochs = episode_rewards.size
    eval_epochs = eval_rewards.size
    plt.figure(1)
    plt.title('Epoch Returns')
    plt.xlabel('Training epochs')
    plt.ylabel('Average return')
    plt.xlim([0, epochs])
    # plt.ylim([-2000, 8000])
    # Plot the smoothed returns
    # episode_rewards = smooth(episode_rewards)
    plt.plot(episode_rewards, label='bs')
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.title('Evaluation Returns')
    plt.xlabel('Evaluation epochs')
    plt.ylabel('return per epoch')
    plt.xlim([0, eval_epochs])
    # plt.ylim([-3000, 7000])
    # eval_rewards = smooth(eval_rewards, 1)
    plt.plot(eval_rewards, label='bs')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot("../Data/results/")
