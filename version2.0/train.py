#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import time
from textcnn_triple import *
from data_input_fast_random import *


#**************************************summary***********************************************
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
        
def get_text_summaries():
    """
    desribe:this summary should not be merged
    """
    with tf.name_scope('predict_text'):
        predict_strings = tf.placeholder(tf.string,name='predict')
        text_summary = tf.summary.text(name='pair',tensor=predict_strings)
    return predict_strings,text_summary

def get_log_summaries():
    """
    desribe:this summary should not be merged
    """
    with tf.name_scope('log'):
        log_strings = tf.placeholder(tf.string,name='log_info')
        log_summary = tf.summary.text(name='log_info',tensor=log_strings)
    return log_strings,log_summary
    

def get_evaluate_test_train_summary():
    """
    desribe:this summary should not be merged
    """
    with tf.name_scope('evaluate'):
        evaluate_on_test_acc = tf.placeholder(tf.float32,name='evaluateOnTest')
        evaluate_on_train_acc = tf.placeholder(tf.float32,name='evaluateOnTrain')
        return evaluate_on_test_acc,tf.summary.scalar('evaluate_on_test',evaluate_on_test_acc),evaluate_on_train_acc,tf.summary.scalar('evaluate_on_train',evaluate_on_train_acc)
   
 #**************************************evluate***********************************************
def get_pred_label_with_sess(sess,query):
    query_index = train_data_set.get_query_index(query)
    pred_label_v = sess.run(textcnn.h_pool_flat,feed_dict={textcnn.query_in:query_index})
    w_matrix = sess.run(textcnn.W_title)
    distance = w_matrix - pred_label_v
    return np.argmax(np.sum(np.square(distance), 1))
    
    
    
def evaluate_test_with_sess(sess,test_question_query_list,test_question_label_list):
    count = 0
    acc = 0
    for i,sentence in enumerate(test_question_query_list):
        max_index = get_pred_label_with_sess(sess,sentence)
        pred_main_question = train_data_set.get_main_question_from_label_index(max_index)
        if pred_main_question == test_question_label_list[i]:
            acc += 1
        count += 1
    return acc/float(count)

def evaluate_train_with_sess(sess,train_question_query_list,train_question_label_list):
    count = 0
    acc = 0
    for i,sentence in enumerate(train_question_query_list):
        max_index = get_pred_label_with_sess()
        pred_main_question = train_data_set.get_main_question_from_label_index(max_index)
        if pred_main_question == train_question_label_list[i]:
            acc += 1
        count += 1
    return acc/float(count)

FLAGS_summaries_dir = 'Summaries/'      #Summaries directory
FLAGS_model_dir =  'model/'             #model directory
FLAGS_learning_rate = 0.01              #Initial learning rate
FLAGS_gpu = 0                           #Enable GPU or not
FLAGS_gpu_num = 1                       #how many gpus to use
FLAGS_step_num = 300000                 #batch_step
FLAGS_many_hard = False                  #many hard negative train
FLAGS_restep_num = 500                 #hard train
FLAGS_print_cycle = 2000                 #how many batches to print
FLAGS_opt_type = 'Adam'            #type of optimizer
name = "textCNN_triplet"
FLAGS_train_write_name = name          #tensorboard_name
FLAGS_checkpoint_name = name+'.ckpt'    #Summaries directory


# the data_set and dataframe
train_data_set = Data_set('data/train_data.csv',20)
train_question_query_list = list(train_data_set.df['query'])
train_question_label_list = list(train_data_set.df['main_question'])
test_data_df = pd.read_csv('data/test_data.csv',encoding='utf-8')
test_question_query_list = list(test_data_df['query'])
test_question_label_list = list(test_data_df['main_question'])

# the arg of triple-net
vocab_size = train_data_set.get_word_num()
main_question_size =  train_data_set.get_main_question_num()
filter_size_list = [2,3,4,5]
filter_num = 10
sequence_length = 20
embedding_size = 100
query_BS = 100

textcnn = TextCNN_triple(sequence_length, vocab_size,embedding_size,main_question_size, filter_size_list, filter_num)

# Optimizer
if FLAGS_opt_type == 'Adam':
    train_opt = tf.train.AdamOptimizer(FLAGS_learning_rate)
elif FLAGS_opt_type == 'Moment':
    train_opt = tf.train.MomentumOptimizer(FLAGS_learning_rate,momentum=0.1)
elif FLAGS_opt_type == 'SGD' :
    train_opt ==  tf.train.GradientDescentOptimizer(FLAGS_learning_rate)
else:
    raise RuntimeError("no right optimizer") 
    
with tf.device('/gpu:0'):
    train_step = train_opt.minimize(textcnn.loss)

#测试集/训练集评估可视化
evaluate_on_test_acc,evaluae_test_summary,evaluate_on_train_acc,evaluae_train_summary = get_evaluate_test_train_summary()

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
        #add evaluate_test
        value_ = evaluate_test_with_sess(sess,test_question_query_list,test_question_label_list)
        evaluae_test_summary_t = sess.run(evaluae_test_summary,feed_dict={evaluate_on_test_acc:value_})
        train_writer.add_summary(evaluae_test_summary_t,batch_id)  
    else:
        query_,doc_pos_,doc_neg_ = train_data_set.get_batch_index(query_BS)
        sess.run(train_step, feed_dict={textcnn.query_in:query_,textcnn.doc_positive_in:doc_pos_,textcnn.doc_negative_in:doc_neg_})
