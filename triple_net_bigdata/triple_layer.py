import pandas as pd
from scipy.sparse import coo_matrix
import collections
import random
import time
import numpy as np
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', 'Summaries', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epoch_num', 5, 'Number of epoch.')
flags.DEFINE_bool('gpu', 0, "Enable GPU or not")

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
        
def get_loss_summary(name):
    with tf.name_scope(name):
        average_loss = tf.placeholder(tf.float32)
        loss_summary = tf.summary.scalar(name + 'average_loss', average_loss)
    return average_loss,loss_summary

def input_layer():
    """
    global var:TRIGRAM_D
    """
    with tf.name_scope('input'):
        query_in = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='QueryBatch')
        doc_positive_in = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='DocBatch')
        doc_negative_in = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='DocBatch')
        on_train = tf.placeholder(tf.bool)
    return query_in,doc_positive_in,doc_negative_in,on_train

def batch_layer(query,doc_pos,doc_neg,next_layer_len,on_train,name):
    with tf.name_scope(name):
        query_layer = batch_normalization(query, on_train, next_layer_len)
        doc_positive_layer = batch_normalization(doc_pos, on_train, next_layer_len)
        doc_negative_layer = batch_normalization(doc_neg, on_train, next_layer_len)

        query_layer_out = tf.nn.relu(query_layer)
        doc_positive_layer_out = tf.nn.relu(doc_positive_layer)
        doc_negative_layer_out = tf.nn.relu(doc_negative_layer)
    return query_layer_out,doc_positive_layer_out,doc_negative_layer_out

def fc_layer(query,doc_positive,doc_negative,layer_in_len,layer_out_len,name,first_layer,batch_norm):
    with tf.name_scope(name):
        layer_par_range = np.sqrt(6.0 / (layer_in_len + layer_out_len))
        weight = tf.Variable(tf.random_uniform([layer_in_len, layer_out_len], -layer_par_range, layer_par_range))
        bias = tf.Variable(tf.random_uniform([layer_out_len], -layer_par_range, layer_par_range))
        variable_summaries(weight, name+'_weights')
        variable_summaries(bias, name+'_biases')
        
        if first_layer:
            query_out = tf.sparse_tensor_dense_matmul(query, weight) + bias
            doc_positive_out = tf.sparse_tensor_dense_matmul(doc_positive, weight) + bias
            doc_negative_out = tf.sparse_tensor_dense_matmul(doc_negative, weight) + bias
        else:
            query_out = tf.matmul(query, weight) + bias
            doc_positive_out = tf.matmul(doc_positive, weight) + bias
            doc_negative_out = tf.matmul(doc_negative, weight) + bias
        
        if batch_norm:
            query_out,doc_positive_out,doc_negative_out = batch_layer(query_out,doc_positive_out,doc_negative_out,layer_out_len,True,name+'BN')
    return query_out,doc_positive_out,doc_negative_out

    
def train_loss_layer(query_y,doc_positive_y,doc_negative_y):
    """
    describe: give batch query,doc+,doc- 
    query_y shape:query_BS,l2_len
    doc_positive_y shape:query_BS,l2_len
    doc_negative_y shape:query_BS,l2_len
    return：
        cos_sim : [2,query_BS]
        loss: float
    """
    with tf.name_scope('train_Cosine_Similarity'):
        
            doc_y = tf.concat([doc_positive_y, doc_negative_y], axis=0)

            query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [2, 1])
            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

            prod = tf.reduce_sum(tf.multiply(tf.tile(query_y, [2, 1]), doc_y), 1, True)
            norm_prod = tf.multiply(query_norm, doc_norm)

            # cos_sim_raw = query * doc / (||query|| * ||doc||)
            cos_sim_raw = tf.truediv(prod, norm_prod)
            cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [2, query_BS])) 
          
    with tf.name_scope('train_Loss'):
        # Train Loss
        # 转化为softmax概率矩阵。
        prob = tf.nn.softmax(cos_sim)
        # 只取第一列，即正样本列概率。
        hit_prob = tf.slice(prob, [0, 0], [-1, 1])
        loss = -tf.reduce_sum(tf.log(hit_prob))
        tf.summary.scalar('loss', loss)
    return cos_sim,prob,loss

def triple_loss_layer(query_y,doc_positive_y,doc_negative_y):
    """
    describe: give batch query,doc+,doc- 
    query_y shape:1,l2_len
    doc_positive_y shape:1,l2_len
    doc_negative_y shape:1,l2_len
    return：
        cos_sim : [2,1]
        loss: float
    """
    doc_y = tf.concat([doc_positive_y, doc_negative_y], axis=0)
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [2, 1])
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))
    
    prod = tf.reduce_sum(tf.multiply(tf.tile(query_y, [2, 1]), doc_y), 1, True)
    norm_prod = tf.multiply(query_norm, doc_norm)

    # cos_sim_raw = query * doc / (||query|| * ||doc||)
    cos_sim_raw = tf.truediv(prod, norm_prod)
    
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [2, 1])) 
    
    prob = tf.nn.softmax(cos_sim)
    # 只取第一列，即正样本列概率。
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_sum(tf.log(hit_prob))
    return loss

def accuracy_layer(prob):
    correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

def predict_layer(query_y,doc_positive_y):
    """
    describe: give batch query,doc+
    query_y shape:1,l2_len
    doc_positive_y shape:main_question_len,l2_len
    return：
        cos_sim : [main_len,1]
        loss: float
    """
    # Cosine similarity
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [MAIN_QUESTION_NUM, 1])
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_positive_y), 1, True))

    prod = tf.reduce_sum(tf.multiply(tf.tile(query_y, [MAIN_QUESTION_NUM, 1]), doc_positive_y), 1, True)
    norm_prod = tf.multiply(query_norm, doc_norm)

    # cos_sim_raw = query * doc / (||query|| * ||doc||)
    cos_sim_raw = tf.truediv(prod, norm_prod)

    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [MAIN_QUESTION_NUM, 1])) 
    
    prob = tf.nn.softmax(cos_sim)
    
    label = tf.argmax(prob,1)[0]
    return prob,label


def pull_all(index_list):
    #该地方插入函数，把query_iin，doc_positive_in,doc_negative_in转化成one_hot，再转化成coo_matrix
    query_in = data_set.get_one_hot_from_batch(index_list,'query')
    doc_positive_in = data_set.get_one_hot_from_batch(index_list,'main_question')
    doc_negative_in = data_set.get_one_hot_from_batch(index_list,'wrong_question')
    
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


def pull_batch(index_list,batch_id):
    
    if (batch_id + 1) * query_BS >= len(index_list):
        print "batch outof index"
        return None
    
    batch_index_list = index_list[batch_id * query_BS:(batch_id + 1) * query_BS]
    query_in, doc_positive_in, doc_negative_in = pull_all(batch_index_list)
    return query_in, doc_positive_in, doc_negative_in


def feed_dict(train_index_list,test_index_list,on_training, Train, batch_id):
    """
    input: data_sets is a dict and the value type is numpy
    describe: to match the text classification the data_sets's content is the doc in df
    """
    if Train:
        query, doc_positive, doc_negative = pull_batch(train_index_list,batch_id)
        
    else:
        query, doc_positive, doc_negative = pull_batch(test_index_list,batch_id)
        
    return {query_in: query, doc_positive_in: doc_positive, doc_negative_in: doc_negative,
            on_train: on_training}

def feed_evaluate_dict(sentence,on_training=True):
    """
    input: data_sets is a dict and the value type is numpy
    describe: to match the text classification the data_sets's content is the doc in df
    """
    #该地方插入函数，把query_iin，doc_positive_in,doc_negative_in转化成one_hot，再转化成coo_matrix
    query = data_set.get_one_hot_from_sentence(sentence)
    doc_positive = data_set.get_one_hot_from_main_question()
    doc_negative = np.ones((1,data_set.get_word_num()))
    
    query = coo_matrix(query)
    doc_positive = coo_matrix(doc_positive)
    doc_negative = coo_matrix(doc_negative)
    
    query = tf.SparseTensorValue(
        np.transpose([np.array(query.row, dtype=np.int64), np.array(query.col, dtype=np.int64)]),
        np.array(query.data, dtype=np.float),
        np.array(query.shape, dtype=np.int64))
    doc_positive = tf.SparseTensorValue(
        np.transpose([np.array(doc_positive.row, dtype=np.int64), np.array(doc_positive.col, dtype=np.int64)]),
        np.array(doc_positive.data, dtype=np.float),
        np.array(doc_positive.shape, dtype=np.int64))
#     doc_negative = tf.SparseTensorValue(
#         np.transpose([np.array(doc_negative.row, dtype=np.int64), np.array(doc_negative.col, dtype=np.int64)]),
#         np.array(doc_negative.data, dtype=np.float),
#         np.array(doc_negative.shape, dtype=np.int64))
    
    return {query_in: query, doc_positive_in: doc_positive,on_train: on_training}

def feed_triple_dict(query,doc_pos,doc_neg,on_training=True):
    """
    input: data_sets is a dict and the value type is numpy
    describe: to match the text classification the data_sets's content is the doc in df
    """
    #该地方插入函数，把query_iin，doc_positive_in,doc_negative_in转化成one_hot，再转化成coo_matrix
    query = data_set.get_one_hot_from_sentence(query)
    doc_positive = data_set.get_one_hot_from_sentence(doc_pos)
    doc_negative = data_set.get_one_hot_from_sentence(doc_neg)
    
    query = coo_matrix(query)
    doc_positive = coo_matrix(doc_positive)
    doc_negative = coo_matrix(doc_negative)
    
    query = tf.SparseTensorValue(
        np.transpose([np.array(query.row, dtype=np.int64), np.array(query.col, dtype=np.int64)]),
        np.array(query.data, dtype=np.float),
        np.array(query.shape, dtype=np.int64))
    doc_positive = tf.SparseTensorValue(
        np.transpose([np.array(doc_positive.row, dtype=np.int64), np.array(doc_positive.col, dtype=np.int64)]),
        np.array(doc_positive.data, dtype=np.float),
        np.array(doc_positive.shape, dtype=np.int64))
    doc_negative = tf.SparseTensorValue(
        np.transpose([np.array(doc_negative.row, dtype=np.int64), np.array(doc_negative.col, dtype=np.int64)]),
        np.array(doc_negative.data, dtype=np.float),
        np.array(doc_negative.shape, dtype=np.int64))
    
    return {query_in: query, doc_positive_in: doc_positive, doc_negative_in: doc_negative,on_train: on_training}


def train():
    """
    global var : epoch_num train_index_list test_index_list train_size test_size query_BS
    """
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True
    if not FLAGS.gpu:
        print "we use gpu"
        config = tf.ConfigProto(device_count= {'GPU' : 0},allow_soft_placement=True)

    # 创建一个Saver对象，选择性保存变量或者模型。
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print "variable initial"
        sess.run(tf.global_variables_initializer())
        print "variable initial ok!"
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test', sess.graph)

        print "start training"
        for epoch_id in range(FLAGS.epoch_num):
            epoch_loss = 0
            epoch_acc = 0
            start = time.time()
            for batch_id in range(int(train_size/query_BS)):
                _,loss_v,acc_v = sess.run([train_step,loss,accuracy], feed_dict=feed_dict(train_index_list,test_index_list,True, True, batch_id))
                epoch_loss += loss_v
                epoch_acc += acc_v

            end = time.time()
            epoch_loss /= int(train_size/query_BS)
            epoch_acc /= int(train_size/query_BS)
            average_loss,train_loss_summary = get_loss_summary('train')
            train_loss = sess.run(train_loss_summary, feed_dict={average_loss: epoch_loss})
            train_writer.add_summary(train_loss, epoch_id + 1)

            print("\nEpoch #%-5d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs | acc: %f" %
                  (epoch_id, epoch_loss, end - start,epoch_acc))

            # test loss
            start = time.time()
            epoch_loss = 0
            epoch_acc = 0
            for batch_id in range(int(test_size/query_BS)):
                loss_v,acc_v = sess.run([loss,accuracy], feed_dict=feed_dict(train_index_list,test_index_list,False, False, batch_id))
                epoch_loss += loss_v
                epoch_acc += acc_v
            end = time.time()
            epoch_loss /= int(test_size/query_BS)
            epoch_acc /= int(test_size/query_BS)
            average_loss,test_loss_summary = get_loss_summary('test')
            test_loss = sess.run(test_loss_summary, feed_dict={average_loss: epoch_loss})
            test_writer.add_summary(test_loss, epoch_id + 1)
            print("Epoch #%-5d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs | acc: %f" %
                  (epoch_id, epoch_loss,end - start,epoch_acc))

        # 保存模型
        save_path = saver.save(sess, "model/model_1.ckpt")
        print("Model saved in file: ", save_path)


#这里之后必要时写成类，现在还不能当库用，里面很多默认的全局。

#根据句子，预测主问题
def predict_label(sentence):
    """
    class fun flag
    global var: pred_prob,pred_label dataset
    """
    saver = tf.train.Saver()

    #写一个函数查看输入query和输出类别
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True
    if not FLAGS.gpu:
        config = tf.ConfigProto(device_count= {'GPU' : 0},allow_soft_placement=True)
    with tf.Session(config=config) as sess:     
        saver.restore(sess, "model/model_1.ckpt")
        print "Model restored."
        pred_prob_v,pred_label_v = sess.run([pred_prob,pred_label],feed_dict=feed_evaluate_dict(sentence))
        pred_main_question = data_set.get_main_question_from_label_index(pred_label_v)
        print sentence,pred_main_question,pred_label_v
        
#测试主问题的正确匹配度
def evaluate_main_question():
    saver = tf.train.Saver()

    #写一个函数查看输入query和输出类别
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True
    if not FLAGS.gpu:
        config = tf.ConfigProto(device_count= {'GPU' : 0},allow_soft_placement=True)
    with tf.Session(config=config) as sess:     
        saver.restore(sess, "model/model_1.ckpt")
        print "Model restored."
        
        count = 0
        acc = 0
        for i,sentence in enumerate(data_set.get_main_question_list()):
            pred_prob_v,pred_label_v = sess.run([pred_prob,pred_label],feed_dict=feed_evaluate_dict(sentence))
            pred_main_question = data_set.get_main_question_from_label_index(pred_label_v)
            if sentence == pred_main_question:
                acc += 1
            count += 1
        print acc/float(count),count
        
#查看一个triple的loss
def show_triple_loss(query,doc_pos,doc_neg):
    """
    class flag
    global var: query_y,doc_pos doc_neg
    """
    saver = tf.train.Saver()

    #写一个函数查看输入query和输出类别
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True
    if not FLAGS.gpu:
        config = tf.ConfigProto(device_count= {'GPU' : 0},allow_soft_placement=True)
    with tf.Session(config=config) as sess:     
        saver.restore(sess, "model/model_1.ckpt")
        loss = triple_loss_layer(query_y,doc_positive_y,doc_negative_y)
        return  sess.run(loss,feed_dict=feed_triple_dict(query,doc_pos,doc_neg))

#对所有log进行测试
#写一个测评脚本，测试真实情况与可视化真实情况
def evaluate_test():
    saver = tf.train.Saver()

    #写一个函数查看输入query和输出类别
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True
    if not FLAGS.gpu:
        config = tf.ConfigProto(device_count= {'GPU' : 0},allow_soft_placement=True)
        print "use_gpu"
    with tf.Session(config=config) as sess:     
        saver.restore(sess, "model/model_1.ckpt")
        print "Model restored."
        
        count = 0
        acc = 0
        df_test = pd.read_excel('data_part1.xlsx')
        test_question_query_list = list(df_test['query'])
        test_question_label_list = list(df_test['main_question'])
        for i,sentence in enumerate(test_question_query_list):
            pred_prob_v,pred_label_v = sess.run([pred_prob,pred_label],feed_dict=feed_evaluate_dict(sentence))
            pred_main_question = data_set.get_main_question_from_label_index(pred_label_v)
            if pred_main_question == test_question_label_list[i]:
                acc += 1
#             else:
#                 print sentence,pred_main_question,test_question_label_list[i]
            count += 1
            if i % 1000 == 0:
                print i
            
        print acc/float(count),count
        
#查看中间层
def show_var_from_sentence(sentence):
    saver = tf.train.Saver()

    #写一个函数查看输入query和输出类别
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True
    if not FLAGS.gpu:
        config = tf.ConfigProto(device_count= {'GPU' : 0},allow_soft_placement=True)
    with tf.Session(config=config) as sess:     
        saver.restore(sess, "model/model_1.ckpt")
        return  sess.run(query_l1_out,feed_dict=feed_evaluate_dict(sentence))


#测试dataset
from data_input_fast import Data_set
data_set = Data_set(data_path='data/train_data.csv')
train_size, test_size = data_set.get_train_test_size()
train_index_list = data_set.train_index_list
test_index_list = data_set.test_index_list

TRIGRAM_D = data_set.get_word_num()
MAIN_QUESTION_NUM = data_set.get_main_question_num()

query_BS = 100
L1_N = 400
L2_N = 120

#input
query_in,doc_positive_in,doc_negative_in,on_train = input_layer()
#fc1 - bn?
query_layer1_out,doc_pos_layer1_out,doc_neg_layer1_out = fc_layer(query_in,doc_positive_in,doc_negative_in,TRIGRAM_D,L1_N,'FC1',True,False)
#fc2 - bn?
query_y,doc_positive_y,doc_negative_y = fc_layer(query_layer1_out,doc_pos_layer1_out,doc_neg_layer1_out,L1_N,L2_N,'FC2',False,False)
#loss
cos_sim,prob,loss = train_loss_layer(query_y,doc_positive_y,doc_negative_y)
#acc
accuracy = accuracy_layer(prob)
#pred_label
pred_prob,pred_label = predict_layer(query_y,doc_positive_y)
# Optimizer
train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

merged = tf.summary.merge_all()

train()

print predict_label('   请问金卡能退吗')

#主问题测试
evaluate_main_question()

#测试集测试
evaluate_test()

#测试triple-loss
query = '会员卡'
doc_pos = '会员卡规则'
doc_neg = '很高兴认识你'
print show_triple_loss(query,doc_pos,doc_neg)

query = '会员卡规则'
doc_pos = '会员卡规则'
doc_neg = '很高兴认识你'
print show_triple_loss(query,doc_pos,doc_neg)

query = '会员卡规则'
doc_pos = '很高兴认识你'
doc_neg = '会员卡规则'
print show_triple_loss(query,doc_pos,doc_neg)