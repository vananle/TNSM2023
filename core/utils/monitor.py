import os

import numpy as np
import torch
import tqdm
from torch.utils.tensorboard.writer import SummaryWriter


class Monitor:

    def __init__(self, args, total_step):
        # save
        self.args = args
        self.label = self.get_label()

        # initialize progress bar
        self.bar = tqdm.tqdm(range(total_step))
        # initialize writer
        self.__create_tensorboard_writer()
        self.__create_csv_writer()
        self.global_step = 0

        print(self.label)

    def get_label(self):
        args = self.args
        if args.method == 'mtsr':
            label = f'{args.method}-{args.model}-{args.dataset}-{args.type}-{args.trunk}-{args.input_len}' \
                    f'-{args.predict_len}-{args.seed}'
        elif 'cs' in args.method:
            label = f'{args.method}-{args.mon_method}-{args.mon_per}-{args.model}-{args.dataset}-' \
                    f'{args.type}-{args.trunk}-{args.input_len}' \
                    f'-{args.predict_len}-{args.seed}'

            self.label_model = f'mtsr_cs-{args.mon_method}-{args.mon_per}-{args.model}-{args.dataset}-' \
                               f'{args.type}-{args.trunk}-{args.input_len}' \
                               f'-{args.predict_len}-{args.seed}'
        else:
            raise NotImplementedError

        return label

    def __create_tensorboard_writer(self):
        # extract parameters
        args = self.args
        # make directory if needed
        folder = f'{args.tensorboard_folder}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        # construct folder name
        folder = f'{folder}/{self.label}'
        # remove old folder
        # os.system(f'rm -rf {folder}')
        # create the tensorboard writer
        self.tensorboard_writer = SummaryWriter(folder)

    def __create_csv_writer(self):
        # extract parameters
        args = self.args
        # make directory if needed
        folder = f'{args.csv_folder}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        # construct csv_path
        path = f'{folder}/{self.label}.csv'
        # remove old path
        # os.system(f'rm -rf {path}')
        # create csv writer
        self.csv_writer = open(path, 'a+')

    def __del__(self):
        self.csv_writer.close()

    def __update_time(self):
        self.bar.update(1)

    def __update_description(self, **kwargs):
        _kwargs = {}
        for key in kwargs:
            if 'test/' in key or '/loss' in key:
                _kwargs[key] = kwargs[key]
        self.bar.set_postfix(**_kwargs)

    def __display(self):
        self.bar.display()

    def __update_tensorboard(self, stats):
        # print(stats)
        for key in stats.keys():
            value = stats[key]
            self.tensorboard_writer.add_scalar(key, value, global_step=self.global_step)

    def __update_csv(self, stats):
        args = self.args
        # line = f'{stats["train/loss"]}'
        try:
            line = f',{stats["val/mse"]}'
            line += f',{stats["val/mae"]}'
            line += f',{stats["test/mse"]}'
            line += f',{stats["test/mae"]}'
        except KeyError:
            line = ''
        line += '\n'
        self.csv_writer.write(line)

    def get_stats(self, info):
        stats = {}
        for key in info:
            if key != 'global_step':
                stats[key] = np.mean(info[key])
        return stats

    def step(self, info):
        # extract stats from all stations
        stats = self.get_stats(info)
        # update progress bar
        self.__update_time()
        self.__update_description(**stats)
        self.__display()
        # log to tensorboard
        self.__update_tensorboard(stats)
        # log to csv
        self.__update_csv(stats)
        self.global_step += 1

    def save_stats(self, stats, tag='final_results'):
        args = self.args

        folder = f'{args.csv_folder}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        # construct csv_path
        path = f'{folder}/{tag}-{self.label}.csv'
        # remove old path
        os.system(f'rm -rf {path}')
        # create csv writer
        csv_writer = open(path, 'a+')

        line = ''
        try:
            line += f',{stats["val/mse"]}'
            line += f',{stats["val/mae"]}'
            line += f',{stats["test/mse"]}'
            line += f',{stats["test/mae"]}'
        except KeyError:
            line = ''

        line += '\n'
        csv_writer.write(line)

    def save_model(self, model, tag=None):
        # extract parameters
        args = self.args
        # make directory if needed
        folder = f'{args.model_folder}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        # construct csv_path
        # path = f'{folder}/{self.label}-{self.global_step}.pkl'
        # # save
        # torch.save(global_model.state_dict(), path)
        if tag is not None:
            path = f'{folder}/{self.label}-{tag}-best.pkl'
        else:
            path = f'{folder}/{self.label}-best.pkl'

        # save
        torch.save(model.state_dict(), path)

    def save_data(self, data):
        args = self.args
        # make directory if needed
        folder = f'{args.model_folder}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = f'{folder}/{self.label}-data.npz'

        np.savez_compressed(path, **data)

    def load_data(self):
        args = self.args
        # make directory if needed
        folder = f'{args.model_folder}'
        if not os.path.exists(folder):
            os.makedirs(folder)

        path = f'{folder}/{self.label}-data.npz'
        if args.method == 'mtsr_nocs' and not os.path.exists(path):
            path = f'{folder}/{self.label_model}-data.npz'
            data = np.load(path)
            self.save_data(data)
        else:
            data = np.load(path)

        return data

    def load_model(self, model, tag=None):
        # extract parameters
        args = self.args
        # make directory if needed
        folder = f'{args.model_folder}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        # construct csv_path
        path = f'{folder}/{self.label}-best.pkl'

        if args.method == 'mtsr_nocs' and not os.path.exists(path):
            path = f'{folder}/{self.label_model}-best.pkl'

            model.load_state_dict(torch.load(path, map_location=torch.device(self.args.device)))
            self.save_model(model)
        else:
            model.load_state_dict(torch.load(path, map_location=torch.device(self.args.device)))

        return model
