#coding=utf-8
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epoches",type=int,default=200)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--seq_length", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=7648)
    parser.add_argument("--size_layers", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--steps_save",type=int,default=3000)
    parser.add_argument("--model_dir",type=str,default="../model/")
    parser.add_argument("--file", type=str, default="../data/poems.txt")
    parser.add_argument("--data_dir", type=str, default="../data/data.tfrecords")
    parser.add_argument("--iter", type=int, default=400)
    args=parser.parse_args()
    return args