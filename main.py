import matplotlib.pyplot as plt
import argparse
from population import *

parser = argparse.ArgumentParser(description='Policy gradient based predator-prey model')
parser.add_argument('--fig_dir', default='./fig_final',
                    type=str, help='The directory to store the figures')
parser.add_argument('--model_dir', default='./model',
                    type=str, help='The directory to store the weights of neural network')
args = parser.parse_args()

fig_dir = args.fig_dir
model_dir = args.model_dir
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

np.random.seed(666)
torch.manual_seed(666)

use_ratio = np.array([[0., 0.], [0.15, 0.]])
growth_rate = np.array([[0.2, 0], [0, -0.05]])
prey_num1 = np.array([[0, 0]]).T    # Producer's prey distribution, of course it should be zero
init_num = np.array([[5000, 10]]).T
upper_bound_num = 1e5   # Ecosystem capacity

predator = Agent(2)
times = []
final_nums = []

for j in range(100):
    num = init_num
    final_num = 0
    final = False
    num_list = []
    num_list2 = []
    for i in range(1000):
        growth_rate[0][0] = 0.2 * (upper_bound_num - num[0, 0]) / upper_bound_num
        predation_ratio = predator.act(num)
        predation_ratio[predation_ratio < 0] = 0.
        predation_ratio[1][0] = 0.  # suppose that there are no deaths caused by intraspecific competition
        prey_num = predation_ratio * num[1]
        prey_array = np.hstack((prey_num1, prey_num))
        died = np.expand_dims(np.sum(prey_array, axis=1), axis=1)
        num = num + growth_rate.dot(num) + np.expand_dims(np.diagonal(use_ratio.dot(prey_array)), axis=1) - died \
            + np.random.normal(0, 1, (2, 1))    # The effects of environmental fluctuations, which are more severe in small populations

        num = np.ceil(num)
        num[num <= 0] = 0

        num_list.append(np.log(num[0][0] + 1))
        num_list2.append(np.log(num[1][0] + 1))

        final_num = num[1]
        if num[1][0] == 0 or num[0][0] == 0:
            predator.rewards.append(-100)
            times.append(i)
            print(i)
            break
        else:
            predator.rewards.append(num[1][0] / 100)
            if i == 999:
                final = True
                times.append(i)

    predator.learn()
    if final:
        final_nums.append(np.log(final_num + 1))
    else:
        final_nums.append(0)
    print(final_num)
    if j % 10 == 0:
        plt.plot(num_list)
        plt.plot(num_list2)
        plt.xlabel("time")
        plt.ylabel("log(num)")
        plt.title("the number of populations")
        plt.savefig(fig_dir + '/num_%d.png' % j)
        plt.clf()
    if j % 100 == 0:
        predator.save(model_dir)

plt.plot(times)
plt.xlabel("episode")
plt.ylabel("time")
plt.title("exist_time")
# plt.show()
plt.savefig(fig_dir + '/exist_time.png')
plt.clf()

plt.plot(final_nums)
plt.xlabel("episode")
plt.ylabel("number")
plt.title("log(final_num)")
# plt.show()
plt.savefig(fig_dir + '/final_num.png')
plt.clf()
