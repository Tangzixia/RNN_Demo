# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from net.rnn import RNNModel
from data.data_batch import gener_batch
import numpy as np
from tensorflow.python.training import checkpoint_ops
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
    acc=rnn.rnn_model(x,y,phase="TRAIN")
    loss,train_op=rnn.build_loss(y)
    init_op=tf.global_variables_initializer()
    saver=tf.train.Saver(tf.global_variables())
    batch_data, batch_label = gener_batch(data_set=args.data_dir, batch_size=args.batch_size*15, num_epoches=None)
    with tf.Session() as sess:
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        # saver.restore(sess,ckpt.model_checkpoint_path)
        saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir=args.model_dir))
        # saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        for epoch in range(args.epoches):
                for i in range(args.iter):
                    batch_data_np,batch_label_np=sess.run([batch_data,batch_label])
                    batch_data_np=np.reshape(batch_data_np,(args.batch_size,-1))
                    batch_label_np=np.reshape(batch_label_np,(args.batch_size,-1))
                    feed_dict={
                        x:batch_data_np,
                        y:batch_label_np,
                    }
                    train_acc,train_loss,_=sess.run([acc,loss,train_op],feed_dict=feed_dict)
                    if(i%70==0):
                        print('epches: {}/{} '.decode("utf-8").encode("gbk").format(epoch, args.epoches),
                              'iterations: {} '.decode("utf-8").encode("gbk").format(i),
                              'training loss: {:.4f} '.decode("utf-8").encode("gbk").format(train_loss))
                    if(i%args.steps_save==0):
                        saver.save(sess,"{}/checkpoints_{}_{}.ckpt".format(args.model_dir,epoch,i))

        coord.request_stop()
        coord.join(threads=threads)

def main():
    args = parse_args()
    rnn_training(args)
if __name__=="__main__":
    # tf.app.run()
    main()