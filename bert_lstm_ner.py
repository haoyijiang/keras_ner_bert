from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras_bert import Tokenizer
from keras_bert import load_trained_model_from_checkpoint
import codecs
from keras_contrib.layers import CRF
from keras_contrib.metrics import crf_accuracy
from keras_contrib.losses import crf_loss
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing import sequence
## 参数
maxlen = 100
Batch_size=16
Epoch =10
config_path="/home/thefair/haoyi/ad_classification/chinese_L-12_H-768_A-12/bert_config.json"
checkpoint_path="/home/thefair/haoyi/ad_classification/chinese_L-12_H-768_A-12/bert_model.ckpt"
dict_path="/home/thefair/haoyi/ad_classification/chinese_L-12_H-768_A-12/vocab.txt"
label={"O":"0","B-PER":"1","I-PER":"2","B-LOC":"3","I-LOC":"4","B-ORG":"5","I-ORG":"6"}

def get_token_dict(dict_path):
    """

    :param dict_path:bert模型的vocab.txt文件
    :return: 将文件中字进行编码
    """
    token_dict={}
    with codecs.open(dict_path,"r","utf-8") as reader:
        for line in reader:
            token=line.strip()
            token_dict[token]=len(token_dict)
    return token_dict

def PreProcessData(path):
    sentences = []
    tags = []
    with open(path, encoding="utf-8") as data_file:
        for sentence in data_file.read().strip().split('\n\n'):
            _sentence = ""
            tag = []
            for word in sentence.strip().split('\n'):
                content = word.strip().split()
                if len(content)==2:
                    _sentence += content[0]
                    tag.append(content[1])
            sentences.append(_sentence)
            tags.append(tag)
    data = [sentences, tags]
    return data

def PreProcessOutputData(text):
    tags = []
    for line in text:
        tag = [0]
        for item in line:
            tag.append(int(label[item.strip()]))
        tag.append(0)
        tags.append(tag)

    pad_tags = pad_sequences(tags, maxlen=100, padding="post", truncating="post")
    result_tags = np.expand_dims(pad_tags,2)
    return result_tags

def PreProcessInputData(text,vocab):
    tokenizer=Tokenizer(vocab)
    word_labels=[]
    seq_types=[]
    for line in text:
        code=tokenizer.encode(first=line,max_len=maxlen)
        word_labels.append(code[0])
        seq_types.append(code[1])
    word_labels=pad_sequences(word_labels,maxlen=maxlen,padding='post',truncating='post')
    seq_types=pad_sequences(seq_types,maxlen=maxlen,padding="post",truncating='post')
    return [word_labels,seq_types]

def build_bert_model(X1,X2):
    """
    :param X1,X2:编码后的结果
    :return: 构建bert第一种模型
    """
    bert_model=load_trained_model_from_checkpoint(config_file=config_path,checkpoint_file=checkpoint_path,seq_len=maxlen)

    wordvec=bert_model.predict([X1,X2])
    return wordvec

def build_model():
    model=Sequential()
    model.add(Bidirectional(LSTM(128,return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(CRF(len(label),sparse_target=True))
    model.compile(loss=crf_loss,optimizer=Adam(1e-5),metrics=[crf_accuracy])
    return model

def train_model(word2vec_train,result_train,word2vec_test,result_test):
    our_model=build_model()
    early_stopping=EarlyStopping(monitor='val_loss',patience=20,verbose=2)
    #filepath="weights-title-attention2_best.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
    history=our_model.fit(x=word2vec_train,y=result_train,batch_size=Batch_size,epochs=Epoch,validation_data=[word2vec_test,result_test],verbose=1,callbacks=[early_stopping])
    #yaml_string=model.to_yaml()
    #with open("test_keras_bert3.yml","w") as f:
    #    f.write(yaml.dump(yaml_string,default_flow_style=True))
    our_model.save('keras_bert_ner.h5')
if __name__=="__main__":
    label={"O":"0","B-PER":"1","I-PER":"2","B-LOC":"3","I-LOC":"4","B-ORG":"5","I-ORG":"6"}
    #转化数据
    train="./data/train.txt"
    test="./data/test.txt"
    vocab=get_token_dict(dict_path)
    input_train,result_train=PreProcessData(train)
    result_train=PreProcessOutputData(result_train)
    input_train_labels,input_train_type=PreProcessInputData(input_train,vocab=vocab)
    input_test,result_test=PreProcessData(test)
    result_test=PreProcessOutputData(result_test)
    input_test_labels,input_test_type=PreProcessInputData(input_test,vocab=vocab)
    #bert模型进行
    word2vec_train=build_bert_model(input_train_labels,input_train_type)
    word2vec_test=build_bert_model(input_test_labels,input_test_type)
    ##训练
    train_model(word2vec_train,result_train,word2vec_test,result_test)
    print(input_train_labels)

