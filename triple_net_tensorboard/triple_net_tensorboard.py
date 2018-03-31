#-*- coding:utf-8 -*-
import pandas as pd
from scipy.sparse import coo_matrix
import collections
import random
import time
import numpy as np
import tensorflow as tf
from data_input_fast import Data_set
from utils import *

#**************************************feed_dict***********************************************

def pull_all(index_list):
    #该地方插入函数，把query_iin，doc_positive_in,doc_negative_in转化成one_hot，再转化成coo_matrix
    query_in = train_data_set.get_one_hot_from_batch(index_list,'query')
    doc_positive_in = train_data_set.get_one_hot_from_batch(index_list,'main_question')
    doc_negative_in = train_data_set.get_one_hot_from_batch(index_list,'other_question')
    
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


def feed_dict_train(train_index_list,test_index_list,on_training, Train, batch_id):
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
    result_list = []
    for i,sentence in enumerate(sentence_list):
        pred_prob_v,pred_label_v = sess.run([pred_prob,pred_label],feed_dict=feed_dict_predict(sentence,doc_main_question_spt))
        pred_main_question = train_data_set.get_main_question_from_label_index(pred_label_v)
        result_list.append(sentence + ":" +pred_main_question)
    return result_list

def evaluate_test_with_sess(sess,test_question_query_list,test_question_label_list):
    count = 0
    acc = 0
    for i,sentence in enumerate(test_question_query_list):
        pred_prob_v,pred_label_v = sess.run([pred_prob,pred_label],feed_dict=feed_dict_predict(sentence,doc_main_question_spt))
        pred_main_question = train_data_set.get_main_question_from_label_index(pred_label_v)
        if pred_main_question == test_question_label_list[i]:
            acc += 1
        count += 1
    return acc/float(count)


# the constant
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summaries_dir', 'Summaries/', 'Summaries directory')
flags.DEFINE_string('train_write_name', 'train_fc*2', 'Summaries directory')
flags.DEFINE_string('test_write_name', 'test_fc*2', 'Summaries directory')
flags.DEFINE_string('checkpoint_name', '"model_full.ckpt".', 'Summaries directory')
flags.DEFINE_string('model_dir', 'model/', 'model directory')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('epoch_num', 5, 'Number of epoch.')
flags.DEFINE_bool('gpu', 0, "Enable GPU or not")
flags.DEFINE_integer('print_cycle', 15, "how many batches to print")

# the data_set and dataframe
train_data_set = Data_set(data_path='data/train_data.csv',data_percent=0.4,train_percent=1) #the train dataset
test_data_df = pd.read_csv('data/test_data.csv',encoding='utf-8')

train_size, test_size = train_data_set.get_train_test_size()
train_index_list = train_data_set.train_index_list
test_index_list = train_data_set.test_index_list
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

#input
query_in,doc_positive_in,doc_negative_in,on_train = input_layer(input_layer_num)
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
# Optimizer
train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

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
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir +FLAGS.test_write_name, sess.graph)

    print "start training"
    for epoch_id in range(FLAGS.epoch_num):
        for batch_id in range(int(train_size/query_BS)):
            summary_v,_,loss_v,acc_v = sess.run([merged,train_step,loss,accuracy], feed_dict=feed_dict_train(train_index_list,test_index_list,True, True, batch_id))    
            train_writer.add_summary(summary_v, batch_id + 1)
            if batch_id % FLAGS.print_cycle == 0:
                #add text_summary
                query_list = random.sample(list(train_data_set.df['query']),10)
                predict_strings_v = predict_label_n_with_sess(sess,query_list)
                text_summary_t = sess.run(text_summary,feed_dict={predict_strings:predict_strings_v})
                train_writer.add_summary(text_summary_t,int(train_size/query_BS) * epoch_id + batch_id+1)
                #add evaluate_test()
                evaluae_summary_t = sess.run(evaluae_summary,feed_dict={evaluate_on_test_acc:evaluate_test_with_sess(sess,test_question_query_list,test_question_label_list)})
                train_writer.add_summary(evaluae_summary_t,batch_id+1)   

        #保存模型,每个epoch保存一次
        save_path = saver.save(sess, FLAGS.model_dir+FLAGS.checkpoint_name)
        print("Model saved in file: ", save_path)