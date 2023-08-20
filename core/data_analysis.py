import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../../')

import warnings

import utils
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


def load_data(args):
    path = os.path.join(args.data_folder, f'{args.dataset}.npz')

    all_data = np.load(path)
    data = all_data['traffic_demands']

    if len(data.shape) > 2:
        data = np.reshape(data, newshape=(data.shape[0], -1))

    # calculate num node
    T, F = data.shape
    N = int(np.sqrt(F))
    args.num_node = N
    args.num_flow = F
    # print('Data shape', data.shape)

    data[data <= 0] = 1e-4
    data[data == np.nan] = 1e-4
    # Train-test split

    total_steps = utils.get_data_size(dataset=args.dataset, data=data)
    data_traffic = data[:total_steps]

    return data_traffic


def data_time_dynamicity(args):
    save_figs_dir = '../data_analysis/'
    if not os.path.exists(save_figs_dir):
        os.makedirs(save_figs_dir)

    datasets = ['abilene', 'geant', 'gnnet-40', 'germany']
    labels = ['Abilene', 'Geant', 'Gnnet-40', 'Germany']

    for idx, dataset in enumerate(datasets):
        args.dataset = dataset
        args = utils.args_adjust(args)

        data = load_data(args)
        new_data = []
        for r in range(0, data.shape[0], 6):
            new_data.append(np.mean(data[r:r + 6], axis=0))

        data = np.array(new_data)
        print(data.shape)
        change_ratio = []
        large_ratio = []
        for flow_id in range(data.shape[1]):
            for time_step in range(1, data.shape[0], 1):
                ratio = abs(data[time_step, flow_id] - data[time_step - 1, flow_id]) / data[time_step - 1, flow_id]
                if ratio < 2:
                    change_ratio.append(ratio)
                else:
                    large_ratio.append(ratio)

        sorted_data = np.sort(change_ratio)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        # Step 4: Plot the CDF
        plt.plot(sorted_data, cdf, label=labels[idx])
        plt.grid(True)

        print(f'{labels[idx]} - {len(large_ratio) * 100 / (len(large_ratio) + len(change_ratio))}%')

    plt.xlabel('Traffic Dynamicity')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Traffic Dynamicity')
    plt.legend()
    fig_name = os.path.join(save_figs_dir, f'cdf_traffic_dynamicity.svg')
    plt.savefig(fig_name, dpi=300)
    plt.close()


def topk_dynamic(args):
    save_figs_dir = '../data_analysis/'
    if not os.path.exists(save_figs_dir):
        os.makedirs(save_figs_dir)

    datasets = ['abilene', 'geant', 'gnnet-40', 'germany']
    labels = ['Abilene', 'Geant', 'Gnnet-40', 'Germany']

    changing = {}
    min_len = np.inf
    for idx, dataset in enumerate(datasets):
        args.dataset = dataset
        args = utils.args_adjust(args)
        data = load_data(args)

        L = int(0.2 * data.shape[1])
        topk_list = []
        for r in range(0, data.shape[0], 6):
            mean_data = np.mean(data[r:r + 6], axis=0)

            topk_id = np.argsort(mean_data)[::-1][0:L]
            topk_list.append(topk_id)

        changing_percentage = []
        for i in range(1, len(topk_list), 1):
            ratio = (L - len(np.intersect1d(topk_list[i - 1], topk_list[i]))) / L
            changing_percentage.append(ratio)

        # changing[dataset] = np.mean(np.array(changing_percentage))
        changing[dataset] = changing_percentage
        if len(changing_percentage) < min_len:
            min_len = len(changing_percentage)

    x = np.arange(min_len)
    for idx, (k, v) in enumerate(changing.items()):
        v = np.array(v[:min_len])
        plt.plot(x, v * 100, label=labels[idx])

    plt.xlabel('Routing cycle')
    plt.ylabel('Percentage (%)')
    plt.title('The change in the monitored flow set ')
    plt.legend()
    fig_name = os.path.join(save_figs_dir, f'topk_dynamic.svg')
    plt.savefig(fig_name, dpi=300)
    plt.close()


def data_skewness(args):
    save_figs_dir = '../data_analysis/'
    if not os.path.exists(save_figs_dir):
        os.makedirs(save_figs_dir)

    datasets = ['abilene', 'geant', 'gnnet-40', 'germany']
    labels = ['Abilene', 'Geant', 'Gnnet-40', 'Germany']

    for idx, dataset in enumerate(datasets):
        args.dataset = dataset
        args = utils.args_adjust(args)

        data = load_data(args)

        mean_traffic = data.flatten()
        print('data: ', dataset, ' Max:', np.max(data), 'Mean:', np.mean(data), 'min:', np.min(data), 'std:',
              np.std(data))
        mean_traffic = np.expand_dims(mean_traffic, -1)
        scaler = MinMaxScaler()
        mean_traffic = scaler.fit_transform(X=mean_traffic)
        mean_traffic = np.squeeze(mean_traffic, -1)
        sorted_data = np.sort(mean_traffic)
        tail_prob = (np.arange(1, len(sorted_data) + 1) / len(sorted_data))

        # Step 4: Plot the tail distribution
        plt.plot(sorted_data, tail_prob, label=labels[idx])

    plt.xlabel('Normalized traffic volume')
    plt.ylabel('Cumulative Probability')
    # plt.title('Flow distribution')
    plt.legend()
    plt.grid(True)
    fig_name = os.path.join(save_figs_dir, f'flow_distribution.svg')
    plt.savefig(fig_name, dpi=300)
    plt.close()


if __name__ == "__main__":
    args = utils.get_args()

    # data_time_dynamicity(args)
    data_skewness(args)
    # topk_dynamic(args)
