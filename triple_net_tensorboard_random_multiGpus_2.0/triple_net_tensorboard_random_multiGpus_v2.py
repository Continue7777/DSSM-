#-*- coding:utf-8 -*-
import pandas as pd
from scipy.sparse import coo_matrix
import collections
import random
import time
import numpy as np
import tensorflow as tf
from data_input_fast_random_v2 import Data_set
from utils_multi_gpu_v2 import *

#**************************************feed_dict***********************************************

def pull_all(hard_df=None,is_retrain=False):
    #该地方插入函数，把query_iin，doc_positive_in,doc_negative_in转化成one_hot，再转化成coo_matrix
    if is_retrain:
        query_in,doc_positive_in,doc_negative_in = train_data_set.get_one_hot_from_df_in(hard_df,query_BS)
    else:
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

def feed_dict_train_multi_gpu():
    """
    input: data_sets is a dict and the value type is numpy
    describe: to match the text classification the data_sets's content is the doc in df
    """
    result_dict = {}
    result_dict[on_train]=True
    for i in range(FLAGS_gpu_num):
        query, doc_positive, doc_negative = pull_all()
        result_dict[query_input_list[i]] = query
        result_dict[doc_positive_input_list[i]] = doc_positive
        result_dict[doc_negative_input_list[i]] = doc_negative
        
    return result_dict

def feed_dict_retrain_hard_multi_gpu(hard_df):
    """
    input: data_sets is a dict and the value type is numpy
    describe: to match the text classification the data_sets's content is the doc in df
    """
    result_dict = {}
    result_dict[on_train]=True
    for i in range(FLAGS_gpu_num):
        query, doc_positive, doc_negative = pull_all(hard_df,True)
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
    query_in = query_input_list[0]
    doc_positive_in  = doc_positive_input_list[0]
    doc_negative_in = doc_negative_input_list[0]

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
        pred_label = model_pred_label()
    result_list = []
    for i,sentence in enumerate(sentence_list):
        pred_label_v = sess.run(pred_label,feed_dict=feed_dict_predict(sentence,doc_main_question_spt))
        pred_main_question = train_data_set.get_main_question_from_label_index(pred_label_v)
        result_list.append(sentence + ":" +pred_main_question)
    return result_list

def evaluate_test_with_sess(sess,test_question_query_list,test_question_label_list):
    with tf.variable_scope(tf.get_variable_scope(),reuse=True):
        pred_label = model_pred_label()
    count = 0
    acc = 0
    for i,sentence in enumerate(test_question_query_list):
        pred_label_v = sess.run(pred_label,feed_dict=feed_dict_predict(sentence,doc_main_question_spt))
        pred_main_question = train_data_set.get_main_question_from_label_index(pred_label_v)
        if pred_main_question == test_question_label_list[i]:
            acc += 1
        count += 1
    return acc/float(count)

def evaluate_train_with_sess(sess,train_question_query_list,train_question_label_list):
    with tf.variable_scope(tf.get_variable_scope(),reuse=True):
        pred_label = model_pred_label()
    count = 0
    acc = 0
    for i,sentence in enumerate(train_question_query_list):
        pred_label_v = sess.run(pred_label,feed_dict=feed_dict_predict(sentence,doc_main_question_spt))
        pred_main_question = train_data_set.get_main_question_from_label_index(pred_label_v)
        if pred_main_question == train_question_label_list[i]:
            acc += 1
        count += 1
    return acc/float(count)

def model_input():
    query_input_list = []
    doc_positive_input_list = []
    doc_negative_input_list = []
    #获取不同gpu上的input函数变量
    for i in xrange(FLAGS_gpu_num):
        #不知道这里加tf.device有没好处，但是加了没毛病。
        with tf.device('/gpu:%d' % i):
            input_result_ = input_layer(input_layer_num)
            query_input_list.append(input_result_[0])
            doc_positive_input_list.append(input_result_[1])
            doc_negative_input_list.append(input_result_[2])
    return query_input_list,doc_positive_input_list,doc_negative_input_list

def model(query_in,doc_positive_in,doc_negative_in,is_first):
    #fc1 
    query_layer1_out,doc_pos_layer1_out,doc_neg_layer1_out = fc_layer(query_in,doc_positive_in,doc_negative_in,input_layer_num,layer1_len,'FC1',True,is_norm,is_first)
    #fc2 
    query_y,doc_positive_y,doc_negative_y = fc_layer(query_layer1_out,doc_pos_layer1_out,doc_neg_layer1_out,layer1_len,layer2_len,'FC2',False,is_norm,is_first)
    return query_y,doc_positive_y,doc_negative_y

def model_loss(query_in,doc_positive_in,doc_negative_in,is_first):
    """
    描述：为了获取函数loss管道出口
    """
    query_y,doc_positive_y,doc_negative_y =  model(query_in,doc_positive_in,doc_negative_in,is_first)
    _,loss = train_loss_layer(query_y,doc_positive_y,doc_negative_y,query_BS,FLAGS_distance_type,FLAGS_loss_type,is_first)
    return loss

def model_pred_label():
    """
    描述：为了获取函数pred_label管道出口
    """
    query_in = query_input_list[0]
    doc_positive_in  = doc_positive_input_list[0]
    doc_negative_in = doc_negative_input_list[0]
    query_y,doc_positive_y,doc_negative_y = model(query_in,doc_positive_in,doc_negative_in,is_first=False)
    pred_label = predict_layer(query_y,doc_positive_y,main_question_num,FLAGS_distance_type,FLAGS_loss_type)
    return pred_label

def get_hard_negative_df_with_sess(sess,train_question_query_list,train_question_label_list):
    query_list = []
    doc_positive_list = []
    doc_hard_negative_list = []
    saver = tf.train.Saver()

    with tf.variable_scope(tf.get_variable_scope(),reuse=True):
        pred_label = model_pred_label()
        for i,sentence in enumerate(train_question_query_list):
            pred_label_v = sess.run(pred_label,feed_dict=feed_dict_predict(sentence,doc_main_question_spt))
            pred_main_question = train_data_set.get_main_question_from_label_index(pred_label_v)
            if pred_main_question != train_question_label_list[i]:
                query_list.append(sentence)
                doc_positive_list.append(train_question_label_list[i])
                doc_hard_negative_list.append(pred_main_question)
    df = pd.DataFrame(data={'query':query_list,'main_question':doc_positive_list,'other_question':doc_hard_negative_list})
    return df

FLAGS_summaries_dir = 'Summaries/'      #Summaries directory
FLAGS_model_dir =  'model/'             #model directory
FLAGS_learning_rate = 0.01              #Initial learning rate
FLAGS_step_num = 100000                 #batch_step
FLAGS_restep_num = 5000                 #hard train
FLAGS_gpu = 0                           #Enable GPU or not
FLAGS_print_cycle = 200                 #how many batches to print
FLAGS_gpu_num = 1                       #how many gpus to use
FLAGS_wfreq_flag = False                #input not use frequence information
FLAGS_ngram_flag = False                #input not use ngram information
FLAGS_loss_type = 'softmax'             #softmax or triplet_loss
FLAGS_distance_type = 'cos'             #distance type eular or cos
FLAGS_opt_type = 'Adam'            #type of optimizer
FLAGS_many_hard = True                  #many hard negative train
name = "fc*2_" + str(FLAGS_step_num) + "_" + str(FLAGS_learning_rate) + '_' + 'wf:' + str(FLAGS_wfreq_flag) + '_ngram_flag:' + str(FLAGS_ngram_flag)
FLAGS_train_write_name =  name          #tensorboard_name
FLAGS_checkpoint_name = name+'.ckpt'    #Summaries directory


# the data_set and dataframe
train_data_set = Data_set(data_path='data/train_data.csv',word_frequence_flag = FLAGS_wfreq_flag,ngram_flag = FLAGS_ngram_flag)
train_question_query_list = list(train_data_set.df['query'])
train_question_label_list = list(train_data_set.df['main_question'])
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
if FLAGS_opt_type == 'Adam':
    train_opt = tf.train.AdamOptimizer(FLAGS_learning_rate)
elif FLAGS_opt_type == 'Moment':
    train_opt = tf.train.MomentumOptimizer(FLAGS_learning_rate,momentum=0.1)
elif FLAGS_opt_type == 'SGD' :
    train_opt ==  tf.train.GradientDescentOptimizer(FLAGS_learning_rate)
else:
    raise RuntimeError("no right optimizer") 

#获取输入变量
query_input_list,doc_positive_input_list,doc_negative_input_list = model_input()

#选择是否并行化
if FLAGS_gpu_num > 1:

    # 定义训练轮数和指数衰减的学习率。
    global_step = tf.get_variable(
        'global_step', [], initializer=tf.constant_initializer(0), 
         trainable=False)

    tower_grads = []

    with tf.variable_scope(tf.get_variable_scope()):
        for i in xrange(FLAGS_gpu_num):
            with tf.device('/gpu:%d' % i):
                is_first = True if i == 0 else False
                cur_loss = model_loss(query_input_list[i],doc_positive_input_list[i],doc_negative_input_list[i],is_first)
                #注意这之前一定没有使用过变量，此处为第一次创建，否则报错
                tf.get_variable_scope().reuse_variables()
                # 使用当前GPU计算所有变量的梯度。
                grads = train_opt.compute_gradients(cur_loss)
                tower_grads.append(grads)

    #计算变量的平均梯度，并输出到TensorBoard日志中。
    grads = average_gradients(tower_grads)
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram('gradients_on_average/%s' % var.op.name, grad)

    #使用平均梯度更新参数。
    apply_gradient_op = train_opt.apply_gradients(grads, global_step=global_step)

    #设置训练step
    train_step = apply_gradient_op
else:
    loss = model_loss(query_input_list[0],doc_positive_input_list[0],doc_negative_input_list[0],True)
    train_step = train_opt.minimize(loss)

#合并所有可视化操作
merged = tf.summary.merge_all()

#测试集/训练集评估可视化
evaluate_on_test_acc,evaluae_test_summary,evaluate_on_train_acc,evaluae_train_summary = get_evaluate_test_train_summary()
#抽样文本预测可视化
predict_strings,text_summary = get_text_summaries()

#训练
config = tf.ConfigProto() 
if not FLAGS_gpu:
    print "here we use gpu"
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

#创建一个Saver对象，选择性保存变量或者模型。
saver = tf.train.Saver()
sess = tf.Session(config=config)
 
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter(FLAGS_summaries_dir + FLAGS_train_write_name, sess.graph)

print "start training"
for batch_id in range(FLAGS_step_num):       
    if batch_id % FLAGS_print_cycle == 0 and batch_id != 0:
        #add text_summary
        query_list = random.sample(list(train_data_set.df['query']),10)
        predict_strings_v = predict_label_n_with_sess(sess,query_list)
        text_summary_t = sess.run(text_summary,feed_dict={predict_strings:predict_strings_v})
        train_writer.add_summary(text_summary_t,batch_id)
        #add evaluate_test
        evaluae_test_summary_t = sess.run(evaluae_test_summary,feed_dict={evaluate_on_test_acc:evaluate_test_with_sess(sess,test_question_query_list,test_question_label_list)})
        train_writer.add_summary(evaluae_test_summary_t,batch_id)   
        #add evaluate on train
        evaluae_train_summary_t = sess.run(evaluae_train_summary,feed_dict={evaluate_on_train_acc:evaluate_train_with_sess(sess,train_question_query_list,train_question_label_list)})
        train_writer.add_summary(evaluae_train_summary_t,batch_id)   
    elif batch_id % 100 == 0:
        summary_v,_ = sess.run([merged,train_step], feed_dict=feed_dict_train_multi_gpu()) 
        train_writer.add_summary(summary_v, batch_id)
    else:
        sess.run(train_step, feed_dict=feed_dict_train_multi_gpu())

#开始many hard negative
if FLAGS_many_hard:
    print "start hard retraining"
    for batch_id in range(FLAGS_restep_num):   
        hard_df = get_hard_negative_df_with_sess(sess,train_question_query_list,train_question_label_list)   
        for k in range(int(len(hard_df)/query_BS)):
            sess.run(train_step, feed_dict=feed_dict_retrain_hard_multi_gpu(hard_df)) 
        #add evaluate_test
        evaluae_test_summary_t = sess.run(evaluae_test_summary,feed_dict={evaluate_on_test_acc:evaluate_test_with_sess(sess,test_question_query_list,test_question_label_list)})
        train_writer.add_summary(evaluae_test_summary_t,batch_id+FLAGS_step_num)   
        #add evaluate on train
        evaluae_train_summary_t = sess.run(evaluae_train_summary,feed_dict={evaluate_on_train_acc:evaluate_train_with_sess(sess,train_question_query_list,train_question_label_list)})
        train_writer.add_summary(evaluae_train_summary_t,batch_id+FLAGS_step_num)   
 

 #保存模型,每个epoch保存一次
sess.close()
save_path = saver.save(sess, FLAGS_model_dir+FLAGS_checkpoint_name)
print("Model saved in file: ", save_path)