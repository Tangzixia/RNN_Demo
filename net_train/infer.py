#coding=utf-8

from net.rnn import *
from common_flags import *


args=parse_args()
# dataLoader=DataLoader(args.batch_size,args.seq_length,args.file)
# key2val,val2key,data=dataLoader.read_data()
# with open("../key2val.txt",'w') as f:
#     for k,v in key2val.items():
#         f.write("{0} {1}\n".format(k,v))
# with open("../val2key.txt",'w') as f:
#     for k,v in val2key.items():
#         f.write("{0} {1}\n".format(k,v))

key2val={}
val2key={}
with open("../key2val.txt",'r') as f:
    lines=f.readlines()
    lines=[line.strip().split(" ") for line in lines]
for line in lines:
    key2val[int(line[0])]=line[1]
    val2key[line[1]]=int(line[0])
print(key2val)
print(val2key)

input_data=tf.placeholder(dtype=tf.int32,shape=(1,None))
output_data=tf.placeholder(dtype=tf.int32,shape=(1,None))

model=RNNModel(args.batch_size,args.seq_length,args.hidden_size,args.vocab_size,args.size_layers,args.learning_rate)
pred,state=model.rnn_model(input=input_data,target=output_data,phase="TEST")

variables_to_restore=tf.global_variables()
init_op=tf.global_variables_initializer()
saver=tf.train.Saver(var_list=variables_to_restore)

startToken=":"

x=np.array([list(map(val2key.get,startToken))])

ckpt_path=tf.train.latest_checkpoint(args.model_dir)

poem=''
with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess,ckpt_path)


    poem=poem+startToken
    # feed_data=[val2key[itm] for itm in genSentence]
    pred_,state_=sess.run([pred,state],feed_dict={input_data:x})
    pred_=pred_.tolist()[-1]
    word=key2val[pred_]

    i=0
    while i<1000:
        poem=poem+word
        x=np.zeros((1,1))
        x[0][0]=val2key[word]
        pred_, state_ = sess.run([pred, state], feed_dict={input_data: x})
        pred_ = pred_.tolist()[-1]
        word=key2val[pred_]
        i=i+1
    print(poem)



