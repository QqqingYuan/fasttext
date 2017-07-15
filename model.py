__author__ = 'PC-LiNing'

import tensorflow as tf
import load_data
import huffman_tree
import numpy
import data_helpers
import datetime
import argparse

label_size = 736
vocab_size = 10186
MAX_NUM_TARGET = 5
MIN_NUM_TARGET = 1
sentence_length = 33
MIN_LENGTH = 2
embedding_size = 100
max_path_length = 15
min_path_length = 5

# training
NUM_EPOCHS = 100
BATCH_SIZE = 1
EVAL_FREQUENCY = 500
Test_Size = 3000
Train_Size = 12404
FLAGS = None


def train():
    # load data
    train_data, train_label,train_label_num,train_path_length,test_data, test_label,test_label_num,test_path_length, counts = load_data.load_cooking_data()
    # tree = [TreeNode]
    # paths = [label_size,max_path_length],padding 0
    # codes = [label_size,max_path_length],padding 0 , false = 0,true = 1
    # paths_length = [label_1_num1,label_2_num,...]
    tree, paths,codes,paths_length = huffman_tree.build_huffman_tree(counts)

    # input is a sentence
    train_data_node = tf.placeholder(tf.int32, shape=(None,))
    # train_length = tf.placeholder(tf.int32,None)
    train_labels_node = tf.placeholder(tf.int32, shape=(None,))
    train_label_num_node = tf.placeholder(tf.int32,None)
    # corresponding to labels
    label_path_length_node = tf.placeholder(tf.int32, (None,))

    # paths
    target_paths = tf.constant(paths,dtype=tf.int32)
    # paths_length
    target_paths_length = tf.constant(paths_length,dtype=tf.int32)
    # codes
    target_codes = tf.constant(codes,dtype=tf.int32)
    # word embedding
    words_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
    # label embedding, store (label_size - 1) non-leaf node
    labels_embedding = tf.Variable(tf.random_uniform([label_size - 1, embedding_size], -1.0, 1.0), name="L")

    # sent = [sentence_length]
    # label = [label_size]
    def computeHidden(sent):
        # embed_data = [sent_length,embedding_size]
        embeded_data = tf.nn.embedding_lookup(words_embedding, sent)
        # hidden = [embedding_size]
        hidden = tf.reduce_mean(embeded_data, axis=0)
        return hidden

    # hidden = [embedding_size]
    # target = [label_length]
    # label_num = [1,5]
    # path_length = [label_length], corresponding to target
    def hierarchical_Softmax(hidden,target,label_num,path_length):
        # target_label = target
        # random sampling
        pos = tf.random_uniform((1,), 0, label_num, tf.int32)
        target_label = target[pos[0]]
        target_path_length = path_length[pos[0]]
        # get path
        # path = [max_path_length,embedding_size], path_length is useful.
        path = tf.nn.embedding_lookup(labels_embedding,target_paths[target_label])
        # flag = [max_path_length], path_length is useful
        flag = tf.cast(target_codes[target_label], tf.float32)
        # score = [max_path_length]
        score = tf.squeeze(tf.nn.sigmoid(tf.matmul(path,tf.expand_dims(hidden,-1))))
        # score_ = -log(score)
        # score_ = [max_path_length]
        score_ = tf.negative(tf.log(tf.multiply(flag,score) + tf.multiply(1-flag,1-score)))
        # reduce sum of path_length
        # path_score = [path_length]
        path_score = tf.gather(score_,tf.range(target_path_length))
        return tf.reduce_sum(path_score)

    def predict_target(hidden,target_label,target_path_length):
        # path = [max_path_length,embedding_size], target_path_length is useful.
        path = tf.nn.embedding_lookup(labels_embedding,target_paths[target_label])
        # flag = [max_path_length], target_path_length is useful
        flag = tf.cast(target_codes[target_label],tf.float32)
        # score = [max_path_length]
        score = tf.squeeze(tf.nn.sigmoid(tf.matmul(path,tf.expand_dims(hidden,-1))))
        # score_ = log(score) or log(1-score)
        # score_ = [max_path_length]
        score_ = tf.log(tf.multiply(flag,score) + tf.multiply(1-flag,1-score))
        # reduce sum of path_length
        # path_score = [path_length]
        path_score = tf.gather(score_, tf.range(target_path_length))
        return tf.reduce_sum(path_score)

    # hidden = [embedding_size]
    # K = 1
    def find_K_best(hidden):
        labels_score = []
        for ids in range(label_size):
            ids_path_length = target_paths_length[ids]
            ids_score = predict_target(hidden,ids,ids_path_length)
            labels_score.append(ids_score)
        # labels_score = [label_size]
        return tf.argmax(labels_score,0)

    hidden_ = computeHidden(train_data_node)
    loss = hierarchical_Softmax(hidden_,train_labels_node,train_label_num_node,label_path_length_node)
    p_target = find_K_best(hidden_)
    # train
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # runing the training
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print('Initialized!')
        # generate batches
        batches = data_helpers.batch_iter(list(zip(train_data,train_label,train_label_num,train_path_length)),BATCH_SIZE,NUM_EPOCHS)
        # batch count
        batch_count = 0
        epoch = 1
        print("Epoch "+str(epoch)+":")
        for batch in batches:
            batch_count += 1
            # train process
            x_batch, y_batch, label_num_batch,path_length_batch = zip(*batch)
            feed_dict = {train_data_node: x_batch[0],train_labels_node: y_batch[0],train_label_num_node:label_num_batch[0],label_path_length_node:path_length_batch[0]}
            _,step,losses = sess.run([train_op, global_step,loss],feed_dict=feed_dict)
            # time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}".format(time_str, step, losses))
            # test process
            if (batch_count * BATCH_SIZE) % Train_Size == 0:
                epoch += 1
                print("Epoch "+str(epoch)+":")
            if batch_count % EVAL_FREQUENCY == 0:
                predict_counts = 0
                sum_loss = 0.0
                for i in range(Test_Size):
                    feed_dict = {train_data_node: test_data[i],train_labels_node: test_label[i],train_label_num_node:test_label_num[i],label_path_length_node:test_path_length[i]}
                    step,losses,predict_i = sess.run([global_step,loss,p_target],feed_dict=feed_dict)
                    sum_loss += losses
                    i_labels = test_label[i]
                    if predict_i in i_labels:
                        predict_counts += 1

                time_str = datetime.datetime.now().isoformat()
                acc = float(predict_counts * 100 / Test_Size)
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, float(sum_loss/Test_Size), acc))


def main(_):
    # if tf.gfile.Exists(FLAGS.summaries_dir):
    #    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    # tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_dir', type=str, default='/tmp/fasttext',help='Summaries directory')
    FLAGS = parser.parse_args()
    tf.app.run()











