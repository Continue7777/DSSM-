#-*- coding:utf-8 -*-
import pandas as pd
import random
import time
import numpy as np
import tensorflow as tf

'''
the main part of the code：
+ layer of dssm
+ sess feed_dict function
+ tools of summary
'''

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
    

def get_evaluate_test_summary():
    """
    desribe:this summary should not be merged
    """
    with tf.name_scope('evaluate'):
        evaluate_on_test_acc = tf.placeholder(tf.float32,name='evaluateOnTest')
        return evaluate_on_test_acc,tf.summary.scalar('evaluate_on_test',evaluate_on_test_acc)
   
 #**************************************layer***********************************************

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

def input_layer(input_len):
    with tf.name_scope('input'):
        query_in = tf.sparse_placeholder(tf.float32, shape=[None, input_len], name='QueryBatch')
        doc_positive_in = tf.sparse_placeholder(tf.float32, shape=[None, input_len], name='DocBatch')
        doc_negative_in = tf.sparse_placeholder(tf.float32, shape=[None, input_len], name='DocBatch')
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
            query_out,doc_positive_out,doc_negative_out = batch_layer(query_out,doc_positive_out,doc_negative_out,layer_out_len,tf.convert_to_tensor(True),name+'BN')
    return query_out,doc_positive_out,doc_negative_out

    
def train_loss_layer(query_y,doc_positive_y,doc_negative_y,query_BS):
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
            cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [2, query_BS]))  * 20
          
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
    
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [2, 1])) * 20
    
    prob = tf.nn.softmax(cos_sim)
    # 只取第一列，即正样本列概率。
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_sum(tf.log(hit_prob))
    return cos_sim,loss

def accuracy_layer(prob):
    correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

def predict_layer(query_y,doc_positive_y,main_question_num):
    """
    describe: give batch query,doc+
    query_y shape:1,l2_len
    doc_positive_y shape:main_question_len,l2_len
    return：
        cos_sim : [main_len,1]
        loss: float
    """
    # Cosine similarity
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [main_question_num, 1])
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_positive_y), 1, True))

    prod = tf.reduce_sum(tf.multiply(tf.tile(query_y, [main_question_num, 1]), doc_positive_y), 1, True)
    norm_prod = tf.multiply(query_norm, doc_norm)

    # cos_sim_raw = query * doc / (||query|| * ||doc||)
    cos_sim_raw = tf.truediv(prod, norm_prod)

    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [main_question_num, 1]))
    
    prob = tf.nn.softmax(cos_sim)
    
    label = tf.argmax(prob,1)[0]
    return prob,label

