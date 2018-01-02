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


class PGAgent(object):
    # code structre inspired by
    # https://danijar.com/structuring-your-tensorflow-models/
    def __init__(self, env):
        self.env = env          # an openai environment

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
        self.logits
        # self.sampled_action
        # self.logprob

    @lazy_property
    def logits(self):
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
    def sampled_action(self):
        return tf.multinomial(self.logits, 1)[:, 0]

    @lazy_property
    def logprob(self):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.sy_act, logits=self.logits
        )

    def sample_trajectory(self, session, max_trajectory_length=1000):
        ob = self.env.reset()
        obs, acs, rewards = [], [], []
        steps = 0

        while True:
            obs.append(ob)
            # ob[None] is equivalent to ob.reshape(1, -1) in this case,
            # i.e. turning ob into a sequence of observations with a length
            # of 1 so that can be fed to the nn
            ac = session.run(
                self.sampled_action,
                feed_dict={
                    self.sy_obs: ob[None]
                }
            )
            ac = ac[0]
            acs.append(ac)
            ob, rew, done, _ = self.env.step(ac)
            rewards.append(rew)
            steps += 1
            if done or steps > max_trajectory_length:
                break

        traj = {
            'observation': np.array(obs),
            'reward': np.array(rewards),
            'action': np.array(acs),
            'len': len(obs)
        }

        return traj

    def sample_trajectories(self, session, num_trajectories):
        trajs = []
        for itr in range(num_trajectories):
            trajs.append(self.sample_trajectory(session=sess))
        return trajs


if __name__ == "__main__":
    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    agent = PGAgent(env)

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1
    )

    with tf.Session(config=tf_config) as sess:
        tf.global_variables_initializer().run()
        # sess.run(tf.initialize_all_variables())

        trajs = agent.sample_trajectories(sess, num_trajectories=10)
