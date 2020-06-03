import numpy as np
import os
import math
import tensorflow as tf
from time import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir
from scripts.utils.data_utils import Data

class MLP_DecisionBoundary(object):
    def __init__(self, dataname):
        self.dataname = dataname
        self.model_dir_name = "MLP-" + self.dataname
        self.model_dir = os.path.join(config.model_root,
                                      self.model_dir_name)
        check_dir(self.model_dir)
        self.data = Data(self.dataname)
        self.X_train, self.y_train, self.X_test, self.y_test = self.data.get_data("all")
        self.train_num, self.feature_num = self.X_train.shape
        self.num_class = int(self.y_train.max() + 1)
        self.batch_size = 64
        self.max_iter = 10000
        self.hidden_layer_size = [1200, 1200]

    def vis_data(self):
        assert (self.dataname == config.three_dim)
        color_map = plt.get_cmap("tab10")(self.y_train.astype(int))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.X_train[:,0], self.X_train[:,1], self.X_train[:,2], s=3, c=color_map)
        plt.show()

    def vis_data_2d(self):
        assert (self.dataname == config.two_dim)
        color_map = plt.get_cmap("tab10")(self.y_train.astype(int))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.X_train[:,0], self.X_train[:,1], s=3, c=color_map)
        plt.show()

    def model_training(self):
        self.x = tf.placeholder(tf.float32, [None, 3])
        layer_1_w = tf.Variable(tf.zeros([3,10]))
        layer_1_b = tf.Variable(tf.zeros([10]))
        layer_2_w = tf.Variable(tf.zeros([10,3]))
        layer_2_b = tf.Variable(tf.zeros([3]))
        init = tf.global_variables_initializer()

        t = tf.matmul(self.x, layer_1_w) + layer_1_b
        y_pred = tf.matmul(t, layer_2_w) + layer_2_b
        y_true = tf.placeholder(tf.float32, [None, 3])

        # loss function
        mse = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(0.003)
        train_step = optimizer.minimize(mse)

        #train
        sess = tf.Session()
        sess.run(init)

        onehot_encoder = OneHotEncoder()
        y_onehot = onehot_encoder.fit_transform(self.y_train.reshape(-1,1))
        y_onehot = y_onehot.toarray()

        for i in range(100):
            train_data = {
                self.x: self.X_train,
                y_true: y_onehot
            }
            sess.run(train_step, feed_dict=train_data)

    def run_training(self):
        hidden_layer_size = self.hidden_layer_size
        learning_rate = 0.01
        with tf.Graph().as_default():
            input_placeholder, labels_placeholder = self.placeholder_inputs(self.feature_num)
            logits = self.inference(input_placeholder,
                                    hidden_layer_size,
                                    self.num_class)

            loss = self.loss(logits, labels_placeholder)

            train_op = self._training(loss, learning_rate)

            eval_correct = self.evaluation(logits, labels_placeholder)

            # build the summary tensor based on the tf collection of summaries
            summary = tf.summary.merge_all()

            init = tf.global_variables_initializer()

            saver = tf.train.Saver()

            sess = tf.Session()

            # Instantiate a SummaryWriter to output summaries and the Graph
            summary_writer = tf.summary.FileWriter(self.model_dir, sess.graph)

            sess.run(init)

            max_steps = self.max_iter

            for step in range(max_steps):
                start_time = time()

                feed_dict = {
                    input_placeholder: self.X_train,
                    labels_placeholder: self.y_train
                }
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict=feed_dict)

                duration = time() - start_time

                # write the summaries and print an overview fairly often
                if step % 100 == 0:
                    # print status
                    print("step {}: loss={} ({} sec)".format(step, loss_value, duration))
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # save a checkpoint and evaluate the model periodically
                if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
                    checkpoint_file = os.path.join(self.model_dir, "model.ckpt")
                    saver.save(sess, checkpoint_file, global_step=step)
                    # evaluate against the training set
                    print("training data eval:")
                    self.do_eval(sess,
                            eval_correct,
                            input_placeholder,
                            labels_placeholder,
                            self.X_train, self.y_train)

                    # evaluate against the validation set
                    None

                    # evaluate against the test set
                    print("test data eval:")
                    self.do_eval(sess,
                                 eval_correct,
                                 input_placeholder,
                                 labels_placeholder,
                                 self.X_test, self.y_test)

    def do_eval(self, sess,
                eval_correct,
                input_placeholder,
                labels_placeholder,
                X, y):
        """
        runs one evaluation against the full epoch of data
        :param sess:
        :param eval_correct:
        :param input_placeholder:
        :param labels_placeholder:
        :param dataset:
        :return:
        """
        true_count = 0
        steps_per_epoch = 1
        num_examples = X.shape[0]
        for step in range(steps_per_epoch):
            feed_dict = {
                input_placeholder: X,
                labels_placeholder: y
            }
            true_count += sess.run(eval_correct, feed_dict=feed_dict)
        acc = float(true_count) / num_examples
        print("num examples: {}, num correct: {}, accuracy: {}".format(num_examples, true_count, acc))

    def placeholder_inputs(self, size, batch_size=None):
        """

        :param batch_size:
        :return:
        """
        input_placeholder = tf.placeholder(tf.float32, shape=(batch_size, size))
        # TODO: do not understand
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
        return input_placeholder, labels_placeholder

    def inference(self, input, hidden_layer_sizes, num_class):
        hidden_layer_out = input
        pre_feature_num = input.shape[1].value
        for idx, hidden_layer_size in enumerate(hidden_layer_sizes):
            with tf.name_scope("layer_" + str(idx)):
                weights = tf.Variable(
                    tf.truncated_normal([pre_feature_num, hidden_layer_size],
                        stddev= 1.0 / math.sqrt(float(pre_feature_num))),
                    name="weights")
                biases = tf.Variable(tf.zeros([hidden_layer_size]),
                                     name="biases")
                hidden_layer_out = tf.nn.relu(
                    tf.matmul(hidden_layer_out, weights) + biases
                )
            pre_feature_num = hidden_layer_size

        # with tf.name_scope("layer_2"):
        #     weights = tf.Variable(
        #         tf.truncated_normal([hidden_layer_size, hidden_layer_size],
        #             stddev= 1.0 / math.sqrt(float(self.feature_num))),
        #         name="weights")
        #     biases = tf.Variable(tf.zeros([hidden_layer_size]),
        #                          name="biases")
        #     hidden_layer_out = tf.nn.relu(
        #         tf.matmul(hidden_layer_out, weights) + biases
        #     )

        with tf.name_scope("softmax_linear"):
            weights = tf.Variable(
                tf.truncated_normal([pre_feature_num, num_class],
                    stddev = 1.0 / math.sqrt(float(pre_feature_num))),
                name="weights"
            )
            biases = tf.Variable(tf.zeros([num_class]),
                                 name="biases")
            # TODO: missing a soft-max
            logits = tf.matmul(hidden_layer_out, weights) + biases
        return logits

    def loss(self, logits, labels):
        """
        calculates the loss from the logits and the labels (ground truth).
        :param logits: tensor - [batch_size, num_class]
        :param labels: int32 - [batch_size]
        :return:
        """
        labels = tf.to_int64(labels)
        return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    def _training(self, loss, learning_rate):
        """
        set up the training Ops.
            1. creates a summarizer to track the loss over time in TensorBoard
            2. creates an optimizer and applies the gradients to all trainable variables
        The Op returned by this function is what must be passed to the
        "sessioin.run()" call to cause the model to train
        :param loss: loss tensor, from self.loss
        :param learning_rate:
        :return:
        """
        tf.summary.scalar("loss", loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def evaluation(self, logits, labels):
        """
        evaluate the quality of the logits at predicting the label
        :param logits:
        :param labels_placeholder:
        :return:
        """
        correct = tf.nn.in_top_k(logits, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def gradient_y_to_x(self, x):
        hidden_layer_size = self.hidden_layer_size

        tf.reset_default_graph()

        input_placeholder, labels_placeholder = self.placeholder_inputs(self.feature_num)
        logits = self.inference(input_placeholder,
                                hidden_layer_size,
                                self.num_class)

        pred_softmax = tf.nn.softmax(logits)
        val_top_2, idx_top_2 = tf.nn.top_k(pred_softmax, k=2, sorted=True)
        margin = val_top_2[:,0] - val_top_2[:,1]
        gradient = tf.gradients(margin, input_placeholder)

        # model_path = os.path.join(self.model_dir,
        #                           "model.ckpt-5999.meta")
        sess = tf.Session()
        # saver = tf.train.import_meta_graph(model_path)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))

        # test
        eval_correct = self.evaluation(logits, labels_placeholder)
        self.do_eval(sess,
                     eval_correct,
                     input_placeholder,
                     labels_placeholder,
                     self.X_train,
                     self.y_train)

        for i in range(500):
            feed_dict = {
                input_placeholder: x
            }
            l = sess.run(logits, feed_dict=feed_dict)
            v2 = sess.run(val_top_2, feed_dict=feed_dict)
            m = sess.run(margin, feed_dict=feed_dict)
            g = sess.run(gradient, feed_dict=feed_dict)
            x = x - g[0] * 0.01
            # print(l, m, g, x)
        return x

    def prediction_vis(self):
        hidden_layer_size = self.hidden_layer_size

        tf.reset_default_graph()

        input_placeholder, labels_placeholder = self.placeholder_inputs(self.feature_num)
        logits = self.inference(input_placeholder,
                                hidden_layer_size,
                                self.num_class)

        pred_val = tf.argmax(logits, 1)

        # model_path = os.path.join(self.model_dir,
        #                           "model.ckpt-5999.meta")
        sess = tf.Session()
        # saver = tf.train.import_meta_graph(model_path)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))



        # test
        eval_correct = self.evaluation(logits, labels_placeholder)
        self.do_eval(sess,
                     eval_correct,
                     input_placeholder,
                     labels_placeholder,
                     self.X_train,
                     self.y_train)

        feed_dict = {
            input_placeholder: self.X_train
        }
        p = sess.run(pred_val, feed_dict=feed_dict)

        x = self.gradient_y_to_x(self.X_train)

        color_map = plt.get_cmap("tab10")(self.y_train.astype(int))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.X_train[:,0], self.X_train[:,1], s=6, marker="o", c="", edgecolor=color_map)
        ax.scatter(x[:,0], x[:,1], s=6, marker="o", c="black")

        plt.show()

        None


if __name__ == '__main__':
    m = MLP_DecisionBoundary(config.two_dim)
    # m.vis_data()
    m.run_training()
    m.gradient_y_to_x(m.X_train)
    # m.prediction_vis()