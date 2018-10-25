#coding=utf-8
import tensorflow as tf
import argparse
import os
from net.rnn import RNNModel
from data.poems_data import DataLoader
from common_flags import *

'''
也可以使用FLAGS=tf.app.flags.FLAGS
'''

def rnn_training(args):
    if os.path.exists(args.model_dir)==False:
        os.mkdir(args.model_dir)
    rnn=RNNModel(args.batch_size,args.seq_length,args.hidden_size,args.vocab_size,args.size_layers,args.learning_rate)
    x=tf.placeholder(dtype=tf.int32,shape=(None,None))
    y=tf.placeholder(dtype=tf.int32,shape=(None,None))
    acc=rnn.rnn_model(x,y)
    loss,train_op=rnn.build_loss(y)
    init_op=tf.global_variables_initializer()
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(args.epoches):
            data = DataLoader(args.batch_size, args.seq_length,args.file)
            data.read_data()
            batches = data.generate_batch()


            for i,batch in enumerate(batches):
                feed_dict={
                    x:batch[0],
                    y:batch[1]
                }
                train_acc,train_loss,_=sess.run([acc,loss,train_op],feed_dict=feed_dict)
                if(i%400==0):
                    print('轮数: {}/{}... '.format(i + 1, args.epoches),
                          '训练步数: {}... '.format(i),
                          '训练误差: {:.4f}... '.format(train_loss))
                if(i%args.steps_save==0):
                    saver.save(sess,"{}/checkpoints_{}_{}.ckpt".format(args.model_dir,epoch,i))

def main(args):
    args = parse_args()
    rnn_training(args)
if __name__=="__main__":
    tf.app.run()
