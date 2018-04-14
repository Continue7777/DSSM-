import tensorflow as tf
import numpy as np


class TextCNN_triple(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, vocab_size,embedding_size,main_question_sizes, filter_sizes, num_filters,margin=1.0):

        # Placeholders for input, output and dropout
        self.query_in = tf.placeholder(tf.int32, shape=[None, sequence_length], name='QueryBatch')
        self.doc_positive_in = tf.placeholder(tf.int32, shape=[None, 1], name='DocPosBatch')
        self.doc_negative_in = tf.placeholder(tf.int32, shape=[None, 1], name='DocNegBatch')
        
        # Embedding layer_query
        with tf.name_scope("query_embedding"):
            self.W_query = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W_query, self.query_in)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


        # Create a convolution + max-pooling layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Embedding layer_title
        with tf.name_scope("title_embedding"):
            self.W_title = tf.Variable(
                tf.random_uniform([main_question_sizes, num_filters_total], -1.0, 1.0),
                name="W")
            self.doc_positive_embedded_chars = tf.nn.embedding_lookup(self.W_title, self.doc_positive_in)
            self.doc_negative_embedded_chars = tf.nn.embedding_lookup(self.W_title, self.doc_negative_in)

        with tf.name_scope('triplet_loss'):
            positive_distance = tf.reduce_sum(tf.square(self.h_pool_flat - self.doc_positive_embedded_chars), 1)
            negative_distance = tf.reduce_sum(tf.square(self.h_pool_flat - self.doc_negative_embedded_chars), 1)

            self.loss = tf.maximum(0., margin + positive_distance - negative_distance)
            self.loss = tf.reduce_mean(self.loss)