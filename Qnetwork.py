# import pandas as pd
import numpy as np
import tensorflow as tf
import helpers

zeros = tf.zeros_initializer()


def xavier_init(shape):
    xavier = tf.contrib.layers.xavier_initializer()
    return tf.divide(xavier(shape), 10)   


class Qnetwork():
    """
    """
    def __init__(self, num_feat, n_hidden_layers, neuron_mult, num_actions, lr):
        self.global_step = tf.Variable(1, dtype=tf.int32, trainable=False, name='global_step')
        prev_num_nodes = num_feat
        self.input = tf.placeholder(shape=[None, num_feat], dtype=tf.float32)
        # keep track of previous layer in self.layers
        self.layers = [self.input]
        prev_layer = self.input
        self.keep_prob = tf.placeholder(tf.float32)
        #keep track of weights for regularization
        wgt_lst = []
        # loop through to create hidden layers
        for i in range(n_hidden_layers):
            # shrink hidden layer size at each subsequent layer
            num_out = np.int32(prev_num_nodes * neuron_mult[i])
            # ensure correct number of nodes
            remainder = num_out % int(1/neuron_mult[i] + 0.00001)
            if remainder != 0:
                num_out += int(1/neuron_mult[i] + 0.00001) - remainder
            w = tf.Variable(xavier_init((prev_num_nodes, num_out)))
            wgt_lst.append(w)
            b = tf.Variable(zeros((num_out,)))
            layer = tf.nn.relu(tf.add(tf.matmul(prev_layer, w), b))
            dropout_layer = tf.nn.dropout(layer, self.keep_prob)
            self.layers.append(dropout_layer)
            prev_num_nodes = num_out
            prev_layer = layer
        # dueling double Q network architecture
        self.stream_A, self.stream_V = tf.split(layer, 2, 1)
        self.AW = tf.Variable(xavier_init((prev_num_nodes//2, num_actions)))
        wgt_lst.append(self.AW)
        self.AB = tf.Variable(zeros((num_actions,)))
        self.VW = tf.Variable(xavier_init((prev_num_nodes//2, 1)))
        wgt_lst.append(self.VW)
        self.VB = tf.Variable(zeros((1,)))
        self.advantage = tf.add(tf.matmul(self.stream_A, self.AW), self.AB)
        self.value = tf.add(tf.matmul(self.stream_V, self.VW), self.VB)
        self.Qout = self.value + tf.subtract(
            self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
#        self.Q_soft = tf.nn.softmax(tf.divide(self.Qout, 10), 1)
        self.predict = tf.argmax(self.Qout, 1)
        # only main Qnetwork evaluation
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        self.error = tf.square(self.targetQ - self.Q)
        self.beta = tf.placeholder(tf.float32)
        self.reg_penalty = tf.add_n([tf.nn.l2_loss(w) for w in wgt_lst]) * self.beta
        self.loss = tf.reduce_mean(self.error) + self.reg_penalty
        self.trainer = tf.train.AdamOptimizer(learning_rate=lr)
        # # clip gradients to counter exploding gradients
        self.grads_tup = self.trainer.compute_gradients(self.loss)
        self.update_model = self.trainer.apply_gradients(self.grads_tup, global_step=self.global_step)#
        # self.update_model = self.trainer.minimize(self.loss)


class DoubleDuelQ():
    """
    """
    def __init__(self, num_feat, num_actions, params, mod_path, new_mod):
        tf.reset_default_graph()
        self.mainQN = Qnetwork(num_feat, params['n_hidden_layers']
            , params['neuron_mult'], num_actions, params['lr'])
        self.targetQN = Qnetwork(num_feat, params['n_hidden_layers']
            , params['neuron_mult'], num_actions, params['lr'])
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init)
        self.mod_path = mod_path
        if not new_mod and helpers.check_tf_model_exists(self.mod_path):
            print('Loading Model..')
            ckpt = tf.train.get_checkpoint_state(self.mod_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        trainables = tf.trainable_variables()
        self.target_ops = update_target_graph(trainables, params['tau'])
        self.mod_path = mod_path
        self.num_actions = num_actions

    def get_action(self, state, keep_prob, random_prop=0):
        Q1 = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.input: state
                                                    , self.mainQN.keep_prob: keep_prob
                                                    })
        if random_prop < 0.000001:
            return Q1
        is_random = np.random.random_sample(Q1.shape) < random_prop        
        randoms = np.random.randint(self.num_actions, size=Q1.shape)
        Q1[is_random] = randoms[is_random]
        return Q1

    def get_targetQ(self, state, done):
        Q1 = self.get_action(state, self.params['keep_prob'])
        Q2 = self.sess.run(self.targetQN.Qout, feed_dict={self.targetQN.input: state
                                                    , self.targetQN.keep_prob: 1
                                                    })
        end_multiplier = (1 - done)
        doubleQ = Q2[range(Q2.shape[0]), Q1]
        future_reward = self.params['q_discount'] * doubleQ * end_multiplier
        return future_reward

    def get_loss_grads(self, s0, actions, rewards, s1, done, beta, update_model=True):
        targetQ = rewards + self.get_targetQ(s1, done)
        if update_model:
            loss, grads, _ = self.sess.run([self.mainQN.loss, self.mainQN.grads_tup, self.mainQN.update_model]
                                      , feed_dict={self.mainQN.input: s0
                                            , self.mainQN.targetQ: targetQ
                                            , self.mainQN.actions: actions
                                            , self.mainQN.beta: beta})
        else:
            loss, grads = self.sess.run([self.mainQN.loss, self.mainQN.grads_tup]
                                      , feed_dict={self.mainQN.input: s0
                                            , self.mainQN.targetQ: targetQ
                                            , self.mainQN.actions: actions
                                            , self.mainQN.beta: beta})
        return loss, grads, targetQ

    def update_target(self):
        update_target(self.target_ops, self.sess)

    def clean_up(self, save, step):
        if save:
            self.saver.save(self.sess, self.mod_path + 'model.ckpt', global_step=step)
        self.sess.close()


def update_target_graph(tf_vars, tau):
    total_vars = len(tf_vars)
    op_holder = []
    half = total_vars//2
    for idx, var in enumerate(tf_vars[0:half]):
        assignment = var.value()*tau + (1-tau) * tf_vars[idx + half].value()
        op_holder.append(tf_vars[idx+half].assign(assignment))
    return op_holder


def update_target(op_holder, sess):
    for op in op_holder:
        sess.run(op)