import gym
import tensorflow as tf

from train_pg import build_mlp


def test_gym_CartPole_v0():
    env = gym.make("CartPole-v0")
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 2
    assert env.observation_space.shape == (4,)


def test_build_mlp():
    input_placeholder = tf.placeholder(tf.float32, shape=(None, 2))
    output_size = 10
    hidden_size = 4
    scope_name = 'tiny_nn'
    n_layers = 3
    out = build_mlp(
        input_placeholder,
        output_size=output_size,
        scope=scope_name,
        n_layers=3,
        size=hidden_size,
        activation=tf.sigmoid)

    assert out.dtype == tf.float32
    # +1: including the output layer
    assert out.name == '{0}/dense_{1}/BiasAdd:0'.format(scope_name, n_layers + 1)
    assert out.shape.as_list() == [None, output_size]
