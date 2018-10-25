#coding=utf-8
import numpy as np
class DataLoader(object):
    def __init__(self,batch_size,seq_length,file="./poems.txt"):
        self.file=file
        self.batch_size=batch_size
        self.seq_length=seq_length

    def read_data(self):
        poem=[]
        with open(self.file,'r') as f:
            lines=f.readlines()
            lines=[line.strip() for line in lines]
            for line in lines:
                for cont in line:
                    poem.append(cont)
        key2val=dict(enumerate(set(poem)))
        key2val[len(key2val)]="EOS"
        self.key2val=key2val

        self.val2key=dict(zip(key2val.values(),range(len(key2val))))
        # self.val2key=dict((v,k) for k,v in key2val.items())
        self.data_x=[self.val2key[itm] for itm in poem]
        return self.key2val,self.val2key,self.data_x

    # def generate_batch(self):
    #     n_chunk=len(self.data_x)//(self.batch_size*self.seq_length)
    #     x_batches=[]
    #     y_batches=[]
    #
    #     for i in range(n_chunk):
    #         start_index=i*self.batch_size*self.seq_length
    #         end_index=start_index+self.batch_size*self.seq_length
    #         x_data=np.array(self.data_x[start_index:end_index])
    #         y_data=np.array(self.data_x[start_index+1:end_index]+[self.val2key["EOS"]])
    #         x_data=np.reshape(x_data,(self.batch_size,self.seq_length))
    #         y_data=np.reshape(y_data,(self.batch_size,self.seq_length))
    #         x_batches.append(x_data)
    #         y_batches.append(y_data)
    #     batches=np.array(list(zip(x_batches,y_batches)))
    #     return batches

if __name__=="__main__":
    data=DataLoader(64,20)
    key2val,val2key,data_x=data.read_data()
    print(key2val)
    print(val2key)