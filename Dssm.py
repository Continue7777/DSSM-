import pandas as pd
from scipy.sparse import coo_matrix
import collections
import random
import time
import numpy as np
import tensorflow as tf
# import data_input

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', 'Summaries', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('epoch_num', 10, 'Number of epoch.')
flags.DEFINE_bool('gpu', 0, "Enable GPU or not")

#制作一个test数据集
def get_random_data(size,length):
    result = np.zeros((size,length))
    for i in range(size):
        j = random.randint(0,length-1)
        result[i,j] = 1
    return result

def get_input_data(train_size,test_size):
    """
    return dict['query_train_data'] = np.array
    """
    data_sets = {}
    
    query_train = get_random_data(train_size,TRIGRAM_D)
    doc_neg_train = get_random_data(train_size*NEG,TRIGRAM_D)
    data_sets['query_train_data'] = query_train
    data_sets['doc_train_positive'] = query_train
    data_sets['doc_train_negative'] = doc_neg_train
    
    query_test = get_random_data(test_size,TRIGRAM_D)
    doc_neg_test = get_random_data(test_size*NEG,TRIGRAM_D)
    data_sets['query_test_data'] = query_test
    data_sets['doc_test_positive'] = query_test
    data_sets['doc_test_negative'] = doc_neg_test
    
    return data_sets

start = time.time()
# 是否加BN层
norm, epsilon = False, 0.001

TRIGRAM_D = 4
# negative sample
NEG = 4
# query batch size
query_BS = 100
# batch size
BS = query_BS * NEG
L1_N = 400
L2_N = 120

# 读取数据
train_size, test_size = 1000, 200
# data_path = 'D:\data\dssm/hy_test.csv'
data_sets = get_input_data(train_size, test_size)


def mean_var_with_update(ema, fc_mean, fc_var):
    ema_apply_op = ema.apply([fc_mean, fc_var])
    with tf.control_dependencies([ema_apply_op]):
        return tf.identity(fc_mean), tf.identity(fc_var)


def batch_normalization(x, phase_train, out_size):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        out_size:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[out_size]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[out_size]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


with tf.name_scope('input'):
    query_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='QueryBatch')
    doc_positive_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='DocBatch')
    doc_negative_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='DocBatch')
    on_train = tf.placeholder(tf.bool)

with tf.name_scope('FC1'):
    l1_par_range = np.sqrt(6.0 / (TRIGRAM_D + L1_N))
    weight1 = tf.Variable(tf.random_uniform([TRIGRAM_D, L1_N], -l1_par_range, l1_par_range))
    bias1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))
    variable_summaries(weight1, 'L1_weights')
    variable_summaries(bias1, 'L1_biases')

    query_l1 = tf.sparse_tensor_dense_matmul(query_batch, weight1) + bias1
    doc_positive_l1 = tf.sparse_tensor_dense_matmul(doc_positive_batch, weight1) + bias1
    doc_negative_l1 = tf.sparse_tensor_dense_matmul(doc_negative_batch, weight1) + bias1

with tf.name_scope('BN1'):
    query_l1 = batch_normalization(query_l1, on_train, L1_N)
    doc_l1 = batch_normalization(tf.concat([doc_positive_l1, doc_negative_l1], axis=0), on_train, L1_N)
    doc_positive_l1 = tf.slice(doc_l1, [0, 0], [query_BS, -1])
    doc_negative_l1 = tf.slice(doc_l1, [query_BS, 0], [-1, -1])

    query_l1_out = tf.nn.relu(query_l1)
    doc_positive_l1_out = tf.nn.relu(doc_positive_l1)
    doc_negative_l1_out = tf.nn.relu(doc_negative_l1)

    # query_l1_out = tf.contrib.slim.batch_norm(query_l1, activation_fn=tf.nn.relu)

# with tf.name_scope('Drop_out'):
#     keep_prob = tf.placeholder("float")
#     query_l1_out = tf.nn.dropout(query_l1_out, keep_prob)
#     doc_positive_l1_out = tf.nn.dropout(doc_positive_l1_out, keep_prob)
#     doc_negative_l1_out = tf.nn.dropout(doc_positive_l1_out, keep_prob)


with tf.name_scope('FC2'):
    l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))

    weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
    bias2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))
    variable_summaries(weight2, 'L2_weights')
    variable_summaries(bias2, 'L2_biases')

    query_l2 = tf.matmul(query_l1_out, weight2) + bias2
    doc_positive_l2 = tf.matmul(doc_positive_l1_out, weight2) + bias2
    doc_negative_l2 = tf.matmul(doc_negative_l1_out, weight2) + bias2

    query_l2 = batch_normalization(query_l2, on_train, L2_N)

with tf.name_scope('BN2'):
    doc_l2 = batch_normalization(tf.concat([doc_positive_l2, doc_negative_l2], axis=0), on_train, L2_N)
    doc_positive_l2 = tf.slice(doc_l2, [0, 0], [query_BS, -1])
    doc_negative_l2 = tf.slice(doc_l2, [query_BS, 0], [-1, -1])

    query_y = tf.nn.relu(query_l2)
    doc_positive_y = tf.nn.relu(doc_positive_l2)
    doc_negative_y = tf.nn.relu(doc_negative_l2)
    # query_y = tf.contrib.slim.batch_norm(query_l2, activation_fn=tf.nn.relu)

with tf.name_scope('Merge_Negative_Doc'):
    # 合并负样本，tile可选择是否扩展负样本。
    doc_y = tf.tile(doc_positive_y, [1, 1])

    for i in range(NEG):
        for j in range(query_BS):
            # slice(input_, begin, size)切片API
            doc_y = tf.concat([doc_y, tf.slice(doc_negative_y, [j * NEG + i, 0], [1, -1])], 0)

with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    # query_norm = sqrt(sum(each x^2))
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [NEG + 1, 1])
    # doc_norm = sqrt(sum(each x^2))
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

    prod = tf.reduce_sum(tf.multiply(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
    norm_prod = tf.multiply(query_norm, doc_norm)

    # cos_sim_raw = query * doc / (||query|| * ||doc||)
    cos_sim_raw = tf.truediv(prod, norm_prod)
    # gamma = 20
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, query_BS])) * 20

with tf.name_scope('Loss'):
    # Train Loss
    # 转化为softmax概率矩阵。
    prob = tf.nn.softmax(cos_sim)
    # 只取第一列，即正样本列概率。
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_sum(tf.log(hit_prob))
    tf.summary.scalar('loss', loss)

with tf.name_scope('Training'):
    # Optimizer
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

    
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.name_scope('Test'):
    average_loss = tf.placeholder(tf.float32)
    loss_summary = tf.summary.scalar('average_loss', average_loss)

with tf.name_scope('Train'):
    train_average_loss = tf.placeholder(tf.float32)
    train_loss_summary = tf.summary.scalar('train_average_loss', train_average_loss)


def pull_all(query_in, doc_positive_in, doc_negative_in):
    #the input type is numpy
    #query_in = query_in.tocoo()
    #doc_positive_in = doc_positive_in.tocoo()
    #doc_negative_in = doc_negative_in.tocoo()
    query_in = coo_matrix(query_in)
    doc_positive_in = coo_matrix(doc_positive_in)
    doc_negative_in = coo_matrix(doc_negative_in)
    
    query_in = tf.SparseTensorValue(
        np.transpose([np.array(query_in.row, dtype=np.int64), np.array(query_in.col, dtype=np.int64)]),
        np.array(query_in.data, dtype=np.float),
        np.array(query_in.shape, dtype=np.int64))
    doc_positive_in = tf.SparseTensorValue(
        np.transpose([np.array(doc_positive_in.row, dtype=np.int64), np.array(doc_positive_in.col, dtype=np.int64)]),
        np.array(doc_positive_in.data, dtype=np.float),
        np.array(doc_positive_in.shape, dtype=np.int64))
    doc_negative_in = tf.SparseTensorValue(
        np.transpose([np.array(doc_negative_in.row, dtype=np.int64), np.array(doc_negative_in.col, dtype=np.int64)]),
        np.array(doc_negative_in.data, dtype=np.float),
        np.array(doc_negative_in.shape, dtype=np.int64))

    return query_in, doc_positive_in, doc_negative_in


def pull_batch(query_data, doc_positive, doc_negative, batch_id):
    query_in = query_data[batch_id * query_BS:(batch_id + 1) * query_BS, :]
    doc_positive_in = doc_positive[batch_id * query_BS:(batch_id + 1) * query_BS, :]
    doc_negative_in = doc_negative[batch_id * query_BS * NEG:(batch_id + 1) * query_BS * NEG, :]

    query_in, doc_positive_in, doc_negative_in = pull_all(query_in, doc_positive_in, doc_negative_in)
    return query_in, doc_positive_in, doc_negative_in


def feed_dict(data_sets,on_training, Train, batch_id):
    """
    input: data_sets is a dict and the value type is numpy
    """
    if Train:
        query_in, doc_positive_in, doc_negative_in = pull_batch(data_sets['query_train_data'],
                                                                data_sets['doc_train_positive'],
                                                                data_sets['doc_train_negative'], 
                                                                batch_id)
        
    else:
        query_in, doc_positive_in, doc_negative_in = pull_batch(data_sets['query_test_data'],
                                                                data_sets['doc_test_positive'],
                                                                data_sets['doc_test_negative'], 
                                                                batch_id)
        
    return {query_batch: query_in, doc_positive_batch: doc_positive_in, doc_negative_batch: doc_negative_in,
            on_train: on_training}

config = tf.ConfigProto() 
config.gpu_options.allow_growth = True
if not FLAGS.gpu:
    config = tf.ConfigProto(device_count= {'GPU' : 0},allow_soft_placement=True)

# 创建一个Saver对象，选择性保存变量或者模型。
# saver = tf.train.Saver()
with tf.Session(config=config) as sess:
# with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test', sess.graph)
    
    print "start traing"
    for epoch_id in range(FLAGS.epoch_num):
        epoch_loss = 0
        start = time.time()
        for batch_id in range(int(train_size/query_BS)):
            _,loss_v = sess.run([train_step,loss], feed_dict=feed_dict(data_sets,True, True, batch_id))
            epoch_loss += loss_v
      
        end = time.time()
        epoch_loss /= int(train_size/query_BS)
        train_loss = sess.run(train_loss_summary, feed_dict={train_average_loss: epoch_loss})
        train_writer.add_summary(train_loss, epoch_id + 1)

        print("\nEpoch #%-5d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %
              (epoch_id, epoch_loss, end - start))

        # test loss
        start = time.time()
        epoch_loss = 0
        for batch_id in range(int(test_size/query_BS)):
            loss_v = sess.run(loss, feed_dict=feed_dict(data_sets,False, False, batch_id))
            epoch_loss += loss_v
        end = time.time()
        epoch_loss /= int(test_size/query_BS)
        test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
        train_writer.add_summary(test_loss, epoch_id + 1)
        # test_writer.add_summary(test_loss, step + 1)
        print("Epoch #%-5d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs" %
              (epoch_id, epoch_loss,end - start))

    # 保存模型
#     save_path = saver.save(sess, "model/model_1.ckpt")
#     print("Model saved in file: ", save_path)