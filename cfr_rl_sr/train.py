from __future__ import print_function

import sys

sys.path.append('../')

import multiprocessing as mp

import numpy as np
from absl import app
from tqdm import tqdm

from env import Environment
from game_mssr import CFRRL_Game
from model import Network
from args_cfr_rl import get_args, args_adjust

GRADIENTS_CHECK = False


def central_agent(args, game, model_weights_queues, experience_queues):
    network = Network(args, game.state_dims, game.action_dim, game.max_moves, master=True)
    network.save_hyperparams(args)
    start_step = network.restore_ckpt()
    for step in tqdm(range(start_step, args.max_step), ncols=70, initial=start_step):
        network.ckpt.step.assign_add(1)
        model_weights = network.model.get_weights()

        for i in range(args.num_agents):
            model_weights_queues[i].put(model_weights)

        if args.method == 'actor_critic':
            # assemble experiences from the agents
            s_batch = []
            a_batch = []
            r_batch = []

            for i in range(args.num_agents):
                s_batch_agent, a_batch_agent, r_batch_agent = experience_queues[i].get()

                assert len(s_batch_agent) == args.num_iter, \
                    (len(s_batch_agent), len(a_batch_agent), len(r_batch_agent))

                s_batch += s_batch_agent
                a_batch += a_batch_agent
                r_batch += r_batch_agent

            assert len(s_batch) * game.max_moves == len(a_batch)

            # used shared RMSProp, i.e., shared g
            actions = np.eye(game.action_dim, dtype=np.float32)[np.array(a_batch)]
            value_loss, entropy, actor_gradients, critic_gradients = network.actor_critic_train(np.array(s_batch),
                                                                                                actions,
                                                                                                np.array(
                                                                                                    r_batch).astype(
                                                                                                    np.float32),
                                                                                                args.entropy_weight)

            if GRADIENTS_CHECK:
                for g in range(len(actor_gradients)):
                    assert np.any(np.isnan(actor_gradients[g])) == False, (
                        'actor_gradients', s_batch, a_batch, r_batch, entropy)
                for g in range(len(critic_gradients)):
                    assert np.any(np.isnan(critic_gradients[g])) == False, (
                        'critic_gradients', s_batch, a_batch, r_batch)

            if step % args.save_step == 0:
                network.save_ckpt(_print=True)

                # log training information
                actor_learning_rate = network.lr_schedule(network.actor_optimizer.iterations.numpy()).numpy()
                avg_value_loss = np.mean(value_loss)
                avg_reward = np.mean(r_batch)
                avg_entropy = np.mean(entropy)

                network.inject_summaries({
                    'learning rate': actor_learning_rate,
                    'value loss': avg_value_loss,
                    'avg reward': avg_reward,
                    'avg entropy': avg_entropy
                }, step)
                print('lr:%f, value loss:%f, avg reward:%f, avg entropy:%f' % (
                    actor_learning_rate, avg_value_loss, avg_reward, avg_entropy))

        elif args.method == 'pure_policy':
            # assemble experiences from the agents
            s_batch = []
            a_batch = []
            r_batch = []
            ad_batch = []

            for i in range(args.num_agents):
                s_batch_agent, a_batch_agent, r_batch_agent, ad_batch_agent = experience_queues[i].get()

                assert len(s_batch_agent) == args.num_iter, \
                    (len(s_batch_agent), len(a_batch_agent), len(r_batch_agent), len(ad_batch_agent))

                s_batch += s_batch_agent
                a_batch += a_batch_agent
                r_batch += r_batch_agent
                ad_batch += ad_batch_agent

            assert len(s_batch) * game.max_moves == len(a_batch)
            # used shared RMSProp, i.e., shared g
            actions = np.eye(game.action_dim, dtype=np.float32)[np.array(a_batch)]
            entropy, gradients = network.policy_train(np.array(s_batch),
                                                      actions,
                                                      np.vstack(ad_batch).astype(np.float32),
                                                      args.entropy_weight)

            if GRADIENTS_CHECK:
                for g in range(len(gradients)):
                    assert np.any(np.isnan(gradients[g])) == False, (s_batch, a_batch, r_batch)

            if step % args.save_step == args.save_step - 1:
                network.save_ckpt(_print=True)

                # log training information
                learning_rate = network.lr_schedule(network.optimizer.iterations.numpy()).numpy()
                avg_reward = np.mean(r_batch)
                avg_advantage = np.mean(ad_batch)
                avg_entropy = np.mean(entropy)
                network.inject_summaries({
                    'learning rate': learning_rate,
                    'avg reward': avg_reward,
                    'avg advantage': avg_advantage,
                    'avg entropy': avg_entropy
                }, step)
                print('lr:%f, avg reward:%f, avg advantage:%f, avg entropy:%f' % (
                    learning_rate, avg_reward, avg_advantage, avg_entropy))


def agent(agent_id, args, game, tm_subset, model_weights_queue, experience_queue):
    random_state = np.random.RandomState(seed=agent_id)
    network = Network(args, game.state_dims, game.action_dim, game.max_moves, master=False)

    # initial synchronization of the model weights from the coordinator 
    model_weights = model_weights_queue.get()
    network.model.set_weights(model_weights)

    idx = 0
    s_batch = []
    a_batch = []
    r_batch = []
    if args.method == 'pure_policy':
        ad_batch = []
    run_iteration_idx = 0
    num_tms = len(tm_subset)
    random_state.shuffle(tm_subset)
    run_iterations = args.num_iter

    while True:
        tm_idx = tm_subset[idx]
        # state
        state = game.get_state(tm_idx)
        s_batch.append(state)
        # action
        if args.method == 'actor_critic':
            policy = network.actor_predict(np.expand_dims(state, 0)).numpy()[0]
        elif args.method == 'pure_policy':
            policy = network.policy_predict(np.expand_dims(state, 0)).numpy()[0]
        else:
            raise NotImplementedError
        assert np.count_nonzero(policy) >= game.max_moves, (policy, state)
        if np.any(np.isnan(policy)):
            raise RuntimeError
        actions = random_state.choice(game.action_dim, game.max_moves, p=policy, replace=False)
        for a in actions:
            a_batch.append(a)

        # reward
        reward = game.reward(tm_idx, actions)
        r_batch.append(reward)

        if args.method == 'pure_policy':
            # advantage
            if args.baseline == 'avg':
                ad_batch.append(game.advantage(tm_idx, reward))
                game.update_baseline(tm_idx, reward)
            elif args.baseline == 'best':
                best_actions = policy.argsort()[-game.max_moves:]
                best_reward = game.reward(tm_idx, best_actions)
                ad_batch.append(reward - best_reward)

        run_iteration_idx += 1
        if run_iteration_idx >= run_iterations:
            # Report experience to the coordinator                          
            if args.method == 'actor_critic':
                experience_queue.put([s_batch, a_batch, r_batch])
            elif args.method == 'pure_policy':
                experience_queue.put([s_batch, a_batch, r_batch, ad_batch])
            else:
                raise NotImplementedError

            # print('report', agent_id)

            # synchronize the network parameters from the coordinator
            model_weights = model_weights_queue.get()
            network.model.set_weights(model_weights)

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]
            if args.method == 'pure_policy':
                del ad_batch[:]
            run_iteration_idx = 0

        # Update idx
        idx += 1
        if idx == num_tms:
            random_state.shuffle(tm_subset)
            idx = 0


def main(_):
    args = get_args()
    args = args_adjust(args)
    env = Environment(args, is_training=True)
    game = CFRRL_Game(args, env)
    model_weights_queues = []
    experience_queues = []
    if args.num_agents == 0 or args.num_agents >= mp.cpu_count():
        args.num_agents = mp.cpu_count() - 1
    print('Agent num: %d, iter num: %d\n' % (args.num_agents + 1, args.num_iter))
    for _ in range(args.num_agents):
        model_weights_queues.append(mp.Queue(1))
        experience_queues.append(mp.Queue(1))

    tm_subsets = np.array_split(game.tm_indexes, args.num_agents)

    coordinator = mp.Process(target=central_agent, args=(args, game, model_weights_queues, experience_queues))

    coordinator.start()

    agents = []
    for i in range(args.num_agents):
        agents.append(mp.Process(target=agent,
                                 args=(i, args, game, tm_subsets[i], model_weights_queues[i], experience_queues[i])))

    for i in range(args.num_agents):
        agents[i].start()

    coordinator.join()


if __name__ == '__main__':
    app.run(main)
