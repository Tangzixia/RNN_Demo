#coding=utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from data.poems_data import DataLoader
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def createTFRecord(filename_queue,data):
    writer=tf.python_io.TFRecordWriter(filename_queue)
    for index,label in enumerate(data):
        example=tf.train.Example(features=tf.train.Features(feature={
            'img_raw':tf.train.Feature(int64_list=tf.train.Int64List(value=[data[index][0]])),
            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[data[index][1]]))
        }
        ))
        writer.write(example.SerializeToString())
    writer.close()

def gener_batch(data_set,batch_size,num_epoches):

    filename_queue=tf.train.string_input_producer([data_set],shuffle=True,num_epochs=num_epoches)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.int64),
                                           'label': tf.FixedLenFeature([], tf.int64)
                                       })
    # print(features['img_raw'])
    # data=tf.decode_raw(features['img_raw'],tf.int64)
    # label=tf.decode_raw(features['label'],tf.int64)
    data = features['img_raw']
    label = features['label']
    min_after_deque=10
    capacity=min_after_deque+3*batch_size
    batch_data,batch_label=tf.train.shuffle_batch([data,label],
                                         batch_size=batch_size,
                                         num_threads=64,
                                         capacity=capacity,
                                         min_after_dequeue=min_after_deque)
    return batch_data,batch_label

if __name__=="__main__":
    batch_size=64
    seq_length=20
    file_path="./poems.txt"
    filename="/home/jobs/code/self/VQA_Demo/data/data.tfrecords"

    # data=DataLoader(batch_size,seq_length,file_path)
    # key2val,val2key,data_x=data.read_data()
    # data_y=data_x[1:]+[val2key["EOS"]]
    # data=list(zip(data_x,data_y))
    # createTFRecord(filename,data)

    batch_data,batch_label=gener_batch(filename, 64, None)

    init_op=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        try:
            for i in range(20):
                batch_data_np, batch_label_np = sess.run([batch_data, batch_label])
                print(batch_data_np)
                print(batch_label_np)
                print("------------")

            coord.request_stop()
            coord.join(threads=threads)
        except tf.errors.OutOfRangeError as e:
            print("Done writing!")
        finally:
            coord.request_stop()





