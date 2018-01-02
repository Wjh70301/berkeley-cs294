import functools

import numpy as np
import tensorflow as tf
import gym
import logz
import os
import time
import inspect
from multiprocessing import Process

import utils as U
from utils import lazy_property


import logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')


class PGAgent(object):
    # code structre inspired by
    # https://danijar.com/structuring-your-tensorflow-models/
    def __init__(self, env, tf_session):
        self.env = env          # an openai environment
        self.tf_session = tf_session

        self.env_is_discrete = isinstance(
            env.action_space, gym.spaces.Discrete)

        # only consider discrete environment for now
        assert self.env_is_discrete

        self.obs_dim = env.observation_space.shape[0]

        self._define_placeholders()
        self._build_cg()

    def _define_placeholders(self):
        env = self.env

        # observation
        obs_dim = env.observation_space.shape[0]
        self.sy_obs = tf.placeholder(
            shape=[None, obs_dim], name="ob", dtype=tf.float32)

        # action
        self.act_dim = env.action_space.n
        self.sy_act = tf.placeholder(
            shape=[None], name="ac", dtype=tf.int32)

        # advantage (Q)
        self.sy_adv = tf.placeholder(
            shape=[None], name='adv', dtype=tf.float32)

    def _build_cg(self):
        # build computational graph
        self.sy_logits
        # self.sy_sampled_action
        # self.sy_logprob
        self.update_policy
        pass

    @lazy_property
    def sy_logits(self):
        inputs = self.sy_obs
        output_layer_size = self.act_dim

        num_layers = 2
        hidden_layer_size = 64
        activation = tf.tanh
        output_activation = None  # None means linear activation

        # build hidden layer
        for _ in range(num_layers):
            inputs = tf.layers.dense(
                inputs=inputs,
                units=hidden_layer_size,
                activation=activation)

        # build output layer
        return tf.layers.dense(
            inputs=inputs,
            units=output_layer_size,
            activation=output_activation)

    @lazy_property
    def sy_logprob(self):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.sy_act, logits=self.sy_logits
        )

    @lazy_property
    def loss(self):
        return tf.reduce_mean(tf.multiply(self.sy_logprob, self.sy_adv))

    @lazy_property
    def update_policy(self, learning_rate=5e-3):
        return tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    @lazy_property
    def sy_sampled_action(self):
        return tf.multinomial(self.sy_logits, 1)[:, 0]

    def sample_trajectory(self, gamma, reward_to_go,
                          max_traj_len=None):
        if max_traj_len is None:
            max_traj_len = self.env.spec.max_episode_steps

        ob = self.env.reset()
        obs, actions, rewards = [], [], []
        steps = 0

        while True:
            obs.append(ob)

            # ob[None] is equivalent to ob.reshape(1, -1) in this case,
            # i.e. turning ob into a sequence of observations with a length
            # of 1 so that can be fed to the nn
            ac = self.tf_session.run(
                self.sy_sampled_action, feed_dict={self.sy_obs: ob[None]})

            actions.append(ac[0])
            ob, rew, done, _ = self.env.step(ac[0])
            rewards.append(rew)
            steps += 1

            if done or steps >= max_traj_len:
                break

        adv = self.compute_advantage(rewards, gamma, reward_to_go)

        traj = {
            'observation': np.array(obs),
            'reward': np.array(rewards),
            'action': np.array(actions),
            'adv': np.array(adv),
            'len': len(obs)
        }
        return traj

    def compute_advantage(self, traj_rewards, gamma, reward_to_go):
        """compute q for a single trajectory"""
        disc_rew = []
        for k, rew in enumerate(traj_rewards):
            disc_rew.append(gamma ** k * rew)

        num_steps = len(traj_rewards)
        if not reward_to_go:
            return np.repeat(np.sum(disc_rew), num_steps)
        else:
            return np.cumsum(disc_rew[::-1])[::-1]

    def sample_trajectories(self, gamma=1, reward_to_go=False,
                            batch_size=None, num_trajectories=None):
        """
        You could sample time step according to total time steps (i.e.
        batch_size) or number of trajectories. batch_size takes precedence
        """
        if batch_size is None and num_trajectories is None:
            raise ValueError(
                "must specify one of `batch_size` and `num_trajectories`")

        trajs = []

        if batch_size is not None:
            size = 0
            while size < batch_size:
                j = self.sample_trajectory(gamma, reward_to_go)
                size += j['len']
                trajs.append(j)

        if num_trajectories is not None:
            for itr in range(num_trajectories):
                j = self.sample_trajectory(gamma, reward_to_go)
                trajs.append(j)

        return trajs

    def concat_trajectories(self, trajectories):
        trajs = trajectories
        obs = np.concatenate([path["observation"] for path in trajs])
        act = np.concatenate([path["action"] for path in trajs])
        adv = np.concatenate([path["adv"] for path in trajs])
        return [obs, act, adv]


if __name__ == "__main__":
    args = U.get_args()
    # logdir = U.setup_logdir(args.exp_name, args.env_name)

    env = gym.make(args.env_name)

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1
    )

    with tf.Session(config=tf_config) as sess:
        agent = PGAgent(env, sess)

        # initializer has to be after agent creation
        tf.global_variables_initializer().run()

        for itr in range(args.n_iter):
            trajs = agent.sample_trajectories(
                # i.e. total number of timesteps in a batch
                # from multiple trajectories
                gamma=args.discount,
                reward_to_go=args.reward_to_go,
                batch_size=args.batch_size,
            )

            # could also specify number of trajectories to sample instead
            # trajs = agent.sample_trajectories(sess, num_trajectories=10)

            [obs, act, adv] = agent.concat_trajectories(trajs)

            feed_dict = {
                agent.sy_obs: obs,
                agent.sy_act: act,
                agent.sy_adv: adv
            }

            before = agent.loss.eval(feed_dict=feed_dict)
            sess.run(agent.update_policy, feed_dict)
            after = agent.loss.eval(feed_dict=feed_dict)
            # logging.info("loss change per update: {0} => {1}".format(before, after))

            returns = [path["reward"].sum() for path in trajs]
            logging.info('\t'.join(
                [
                    '#{0}'.format(itr + 1),
                    '{0}-timesteps'.format(len(obs)),
                    '{0}-trajs'.format(len(trajs)),
                    '{0:.8f}'.format(np.mean(returns)),
                    '{0:.8f}'.format(np.max(returns)),
                    '{0:.8f} => {1:.8f}'.format(before, after),

                ]
            ))
