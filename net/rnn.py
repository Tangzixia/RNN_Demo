#coding=utf-8

import tensorflow as tf
import os
import numpy as np
import argparse
from tensorflow.contrib.layers import fully_connected

class RNNModel(object):
    def __init__(self,batch_size,seq_length,hidden_size,vocab_size,size_layers,learning_rate):
        self.batch_size=batch_size
        self.seq_length=seq_length
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.size_layers=size_layers
        self.learning_rate=learning_rate


    def rnn_model(self,input,target,phase):
        cell=tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        cell=tf.nn.rnn_cell.MultiRNNCell([cell]*self.size_layers,state_is_tuple=True)

        embeddings=tf.get_variable(name="embed",shape=(self.vocab_size,self.hidden_size),dtype=tf.float32)
        #(?,?,)
        input=tf.nn.embedding_lookup(embeddings,input)
        if phase=="TEST":
            self.batch_size=1

        initial_state=cell.zero_state(batch_size=self.batch_size,dtype=tf.float32)
        outputs,state=tf.nn.dynamic_rnn(cell,input,initial_state=initial_state)
        w=tf.get_variable(name="w",shape=[self.hidden_size,self.vocab_size],dtype=tf.float32)
        b=tf.get_variable(name="b",shape=[self.vocab_size,],dtype=tf.float32)
        outputs=tf.nn.bias_add(tf.matmul(tf.reshape(outputs,(-1,self.hidden_size)),w),b)
        predicts=tf.nn.softmax(outputs)

        pred=tf.argmax(predicts,1)
        self.pred=pred
        pred=tf.cast(tf.reshape(pred,(self.batch_size,-1)),tf.int32)

        acc=tf.reduce_mean(tf.cast(tf.equal(pred,target),tf.int32))
        self._outputs=outputs
        self._pre=predicts
        self._acc=acc
        self.state=state

        if phase=="TRAIN":
            return self._acc

        elif phase=="TEST":
            return self.pred,self.state

    def build_loss(self,target):
        target=tf.reshape(target,[-1])
        target=tf.one_hot(target,self.vocab_size)
        cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,logits=self._outputs)
        loss=tf.reduce_mean(cross_entropy)
        optim=tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op=optim.minimize(loss)
        return loss,train_op