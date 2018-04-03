#-*- coding:utf-8 -*-
import pandas as pd
from scipy.sparse import coo_matrix
import collections
import random
import time
import numpy as np
import tensorflow as tf
from data_input_fast_random import Data_set
from utils_multi_gpu import *

#**************************************feed_dict***********************************************

def pull_all():
    #该地方插入函数，把query_iin，doc_positive_in,doc_negative_in转化成one_hot，再转化成coo_matrix
    query_in,doc_positive_in,doc_negative_in = train_data_set.get_one_hot_from_batch(query_BS)
    
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


def feed_dict_train(on_training):
    """
    input: data_sets is a dict and the value type is numpy
    describe: to match the text classification the data_sets's content is the doc in df
    """
    query, doc_positive, doc_negative = pull_all()
        
    return {query_in: query, doc_positive_in: doc_positive, doc_negative_in: doc_negative,
            on_train: on_training}

def feed_dict_train_multi_gpu():
    """
    input: data_sets is a dict and the value type is numpy
    describe: to match the text classification the data_sets's content is the doc in df
    """
    result_dict = {}
    result_dict[on_train]=True
    for i in range(FLAGS.gpu_num):
        query, doc_positive, doc_negative = pull_all()
        result_dict[query_input_list[i]] = query
        result_dict[doc_positive_input_list[i]] = doc_positive
        result_dict[doc_negative_input_list[i]] = doc_negative
        
    return result_dict

def feed_dict_predict(sentence,doc_positive_spt,on_training=True):
    """
    input: data_sets is a dict and the value type is numpy
    describe: to match the text classification the data_sets's content is the doc in df
    """
    #该地方插入函数，把query_iin，doc_positive_in,doc_negative_in转化成one_hot，再转化成coo_matrix
    query = train_data_set.get_one_hot_from_sentence(sentence)
    
    query = coo_matrix(query)

    query = tf.SparseTensorValue(
        np.transpose([np.array(query.row, dtype=np.int64), np.array(query.col, dtype=np.int64)]),
        np.array(query.data, dtype=np.float),
        np.array(query.shape, dtype=np.int64))

    return {query_in: query, doc_positive_in: doc_positive_spt,on_train: on_training}

def feed_dict_triple(query,doc_pos,doc_neg,on_training=True):
    """
    input: data_sets is a dict and the value type is numpy
    describe: to match the text classification the data_sets's content is the doc in df
    """
    #该地方插入函数，把query_iin，doc_positive_in,doc_negative_in转化成one_hot，再转化成coo_matrix
    query = train_data_set.get_one_hot_from_sentence(query)
    doc_positive = train_data_set.get_one_hot_from_sentence(doc_pos)
    doc_negative = train_data_set.get_one_hot_from_sentence(doc_neg)
    
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

def predict_label_n_with_sess(sess,sentence_list):
    with tf.variable_scope(tf.get_variable_scope(),reuse=True):
        #fc1 
        query_layer1_out,doc_pos_layer1_out,doc_neg_layer1_out = fc_layer(query_in,doc_positive_in,doc_negative_in,input_layer_num,layer1_len,'FC1',True,is_norm)
        #fc2 
        query_y,doc_positive_y,doc_negative_y = fc_layer(query_layer1_out,doc_pos_layer1_out,doc_neg_layer1_out,layer1_len,layer2_len,'FC2',False,is_norm)
        #loss
        cos_sim,prob,loss = train_loss_layer(query_y,doc_positive_y,doc_negative_y,query_BS)
        #acc
        accuracy = accuracy_layer(prob)
        #pred_label
        pred_prob,pred_label = predict_layer(query_y,doc_positive_y,main_question_num)
    result_list = []
    for i,sentence in enumerate(sentence_list):
        pred_prob_v,pred_label_v = sess.run([pred_prob,pred_label],feed_dict=feed_dict_predict(sentence,doc_main_question_spt))
        pred_main_question = train_data_set.get_main_question_from_label_index(pred_label_v)
        result_list.append(sentence + ":" +pred_main_question)
    return result_list

def evaluate_test_with_sess(sess,test_question_query_list,test_question_label_list):
    with tf.variable_scope(tf.get_variable_scope(),reuse=True):
        #fc1 
        query_layer1_out,doc_pos_layer1_out,doc_neg_layer1_out = fc_layer(query_in,doc_positive_in,doc_negative_in,input_layer_num,layer1_len,'FC1',True,is_norm)
        #fc2 
        query_y,doc_positive_y,doc_negative_y = fc_layer(query_layer1_out,doc_pos_layer1_out,doc_neg_layer1_out,layer1_len,layer2_len,'FC2',False,is_norm)
        #loss
        cos_sim,prob,loss = train_loss_layer(query_y,doc_positive_y,doc_negative_y,query_BS)
        #acc
        accuracy = accuracy_layer(prob)
        #pred_label
        pred_prob,pred_label = predict_layer(query_y,doc_positive_y,main_question_num)
    count = 0
    acc = 0
    for i,sentence in enumerate(test_question_query_list):
        pred_prob_v,pred_label_v = sess.run([pred_prob,pred_label],feed_dict=feed_dict_predict(sentence,doc_main_question_spt))
        pred_main_question = train_data_set.get_main_question_from_label_index(pred_label_v)
        if pred_main_question == test_question_label_list[i]:
            acc += 1
        count += 1
    return acc/float(count)


def get_loss(query_in,doc_positive_in,doc_negative_in):
    #fc1 
    query_layer1_out,doc_pos_layer1_out,doc_neg_layer1_out = fc_layer(query_in,doc_positive_in,doc_negative_in,input_layer_num,layer1_len,'FC1',True,is_norm)
    #fc2 
    query_y,doc_positive_y,doc_negative_y = fc_layer(query_layer1_out,doc_pos_layer1_out,doc_neg_layer1_out,layer1_len,layer2_len,'FC2',False,is_norm)
    #loss
    cos_sim,prob,loss = train_loss_layer(query_y,doc_positive_y,doc_negative_y,query_BS)
    return loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
	    # Note that each grad_and_vars looks like the following:
	    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
    	for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

	    # Average over the 'tower' dimension.
	    grad = tf.concat(axis=0, values=grads)
	    grad = tf.reduce_mean(grad, 0)

	    # Keep in mind that the Variables are redundant because they are shared
	    # across towers. So .. we will just return the first tower's pointer to
	    # the Variable.
	    v = grad_and_vars[0][1]
	    grad_and_var = (grad, v)
	    average_grads.append(grad_and_var)
  	return average_grads


# the constant
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summaries_dir', 'Summaries/', 'Summaries directory')
flags.DEFINE_string('train_write_name', 'train_fc*2(multi_gpu)', 'Summaries directory')
flags.DEFINE_string('checkpoint_name', '"model_fc*2(multi_gpu).ckpt".', 'Summaries directory')
flags.DEFINE_string('model_dir', 'model/', 'model directory')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('step_num', 100000, 'batch_step')
flags.DEFINE_bool('gpu', 0, "Enable GPU or not")
flags.DEFINE_integer('print_cycle', 100, "how many batches to print")
flags.DEFINE_integer('gpu_num', 2, "how many gpus to use")

# the data_set and dataframe
train_data_set = Data_set(data_path='data/train_data.csv') #the train dataset
test_data_df = pd.read_csv('data/test_data.csv',encoding='utf-8')
test_question_query_list = list(test_data_df['query'])
test_question_label_list = list(test_data_df['main_question'])

# coo fisrt
doc_main_question = train_data_set.get_one_hot_from_main_question()
doc_main_question = coo_matrix(doc_main_question)
doc_main_question_spt = tf.SparseTensorValue(
    np.transpose([np.array(doc_main_question.row, dtype=np.int64), np.array(doc_main_question.col, dtype=np.int64)]),
    np.array(doc_main_question.data, dtype=np.float),
    np.array(doc_main_question.shape, dtype=np.int64))

# the arg of triple-net
input_layer_num = train_data_set.get_word_num()
main_question_num =  train_data_set.get_main_question_num()
query_BS = 100

# the architecture of the triple-net
is_norm = False
layer1_len = 400
layer2_len = 120

# input
query_in,doc_positive_in,doc_negative_in,on_train = input_layer(input_layer_num)
# Optimizer
train_opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

# 定义训练轮数和指数衰减的学习率。
global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), 
     trainable=False)

tower_grads = []
query_input_list = []
doc_positive_input_list = []
doc_negative_input_list = []
# 将神经网络的优化过程跑在不同的GPU上。
with tf.variable_scope(tf.get_variable_scope()):
    for i in xrange(FLAGS.gpu_num):
        with tf.device('/gpu:%d' % i):
            input_result_ = input_layer(input_layer_num)
            query_input_list.append(input_result_[0])
            doc_positive_input_list.append(input_result_[1])
            doc_negative_input_list.append(input_result_[2])
            cur_loss = get_loss(query_input_list[i],doc_positive_input_list[i],doc_negative_input_list[i])
            # 在第一次声明变量之后，将控制变量重用的参数设置为True。这样可以
            # 让不同的GPU更新同一组参数。注意tf.name_scope函数并不会影响
            # tf.get_ variable的命名空间。
            tf.get_variable_scope().reuse_variables()
            # 使用当前GPU计算所有变量的梯度。
            grads = train_opt.compute_gradients(cur_loss)
            tower_grads.append(grads)

# 计算变量的平均梯度，并输出到TensorBoard日志中。
grads = average_gradients(tower_grads)
for grad, var in grads:
    if grad is not None:
        tf.summary.histogram('gradients_on_average/%s' % var.op.name, grad)

# 使用平均梯度更新参数。
apply_gradient_op = train_opt.apply_gradients(grads, global_step=global_step)


train_step = apply_gradient_op

merged = tf.summary.merge_all()

#evaluate
evaluate_on_test_acc,evaluae_summary = get_evaluate_test_summary()
#record predict text
predict_strings,text_summary = get_text_summaries()

#train
config = tf.ConfigProto() 
if not FLAGS.gpu:
    print "here we use gpu"
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

# 创建一个Saver对象，选择性保存变量或者模型。
saver = tf.train.Saver()
with tf.Session(config=config) as sess:
 
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + FLAGS.train_write_name, sess.graph)

    print "start training"
    for batch_id in range(FLAGS.step_num):       
        if batch_id % FLAGS.print_cycle == 0 and batch_id != 0:
            #add text_summary
            query_list = random.sample(list(train_data_set.df['query']),10)
            predict_strings_v = predict_label_n_with_sess(sess,query_list)
            text_summary_t = sess.run(text_summary,feed_dict={predict_strings:predict_strings_v})
            train_writer.add_summary(text_summary_t,batch_id)
            #add evaluate_test()
            evaluae_summary_t = sess.run(evaluae_summary,feed_dict={evaluate_on_test_acc:evaluate_test_with_sess(sess,test_question_query_list,test_question_label_list)})
            train_writer.add_summary(evaluae_summary_t,batch_id)   
        elif batch_id % 100 == 0:
            summary_v,_ = sess.run([merged,train_step], feed_dict=feed_dict_train_multi_gpu()) 
            train_writer.add_summary(summary_v, batch_id)
        else:
            sess.run(train_step, feed_dict=feed_dict_train_multi_gpu())


    #保存模型,每个epoch保存一次
    save_path = saver.save(sess, FLAGS.model_dir+FLAGS.checkpoint_name)
    print("Model saved in file: ", save_path)