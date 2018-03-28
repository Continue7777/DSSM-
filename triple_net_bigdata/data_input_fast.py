import numpy as np
import pandas as pd
import random
import time

class Data_set:
    def __init__(self,data_path,train_percent=0.7):
        start = time.time()
        self.data_path = data_path
        self.df = pd.read_csv(data_path,encoding='utf-8')
        end = time.time()
        print "read over,time:" + str(end-start)
        self.word_dict = self._get_word_dict()
        end1 = time.time()
        print "build dict over,time:" + str(end1-end)
        self.main_question_len = self.get_main_question_num()
        self.train_index_list,self.test_index_list = self._spilt_train_test_index(train_percent)
        end2 = time.time()
        print "split over,time:" + str(end2-end1)
        self.main_question_list = self.get_main_question_list(train_percent)
        self.query_triple_list,self.main_question_triple_list,self.other_question_triple_list = self.generate_triple_fast()
        end3 = time.time()
        print "generate_triple_over,time:" + str(end3-end2)
        
    #获取词数
    def get_word_num(self):
        return len(self.word_dict)
     
    #获取训练集测试集大小
    def get_train_test_size(self):
        return len(self.train_index_list),len(self.test_index_list)
    
    #获取主问题数:wrong+main的并集总数
    def get_main_question_num(self):
        length = len(set(list(self.df['main_question'])))
        return length
    
    #根据数据集生成triple
    def generate_triple_fast(self,train_percent):
        query = list(self.df['query'])
        main_question = list(self.df['main_question'])
        query_final = query * self.main_question_len
        main_question_final = main_question * self.main_question_len
        other_question = list(set(self.df['main_question'])) * len(self.df)
        return query_final,main_question_final,other_question
    
    def _spilt_train_test_index(self,train_percent):
        data_size = len(self.df) * self.main_question_len
        full_list = [i for i in range(data_size)]
        train_index_list = random.sample(full_list,int(data_size*train_percent))
        test_index_list = list(set(full_list) - set(train_index_list))
        return train_index_list,test_index_list
        
    #构建字典索引
    def _get_word_dict(self):
        l = list(self.df['query'])
        l.extend(list(set(self.df['main_question'])))
        string = "".join(l)
        s = set(string)
        word_dict = {}
        for i,w in enumerate(s):
            word_dict[w] = i
        return word_dict
    
    #通过索引把df转成词向量
    def get_one_hot_from_batch(self,index_list,column_name):
        if column_name == 'query':
            doc_list = self.query_triple_list
        elif column_name == 'main_question':
            doc_list = self.main_question_triple_list
        elif column_name == 'other_question':
            doc_list = self.other_question_triple_list
        else:
            print "your column name is wrong"

        result = np.zeros((len(index_list),len(self.word_dict)))
        for i,s in enumerate(doc_list):
            for j,w in enumerate(s):
                if w in self.word_dict:
                    result[i,self.word_dict[w]] = 1

        return result 
    
    # 获取一个句子的onehot
    def get_one_hot_from_sentence(self,sentence):
        #转化到unicode编码
        if isinstance(sentence,str):
            sentence = sentence.decode('utf-8')
        
        result = np.zeros((1,len(self.word_dict)))
    
        for w in sentence:
            if w in self.word_dict:
                result[0,self.word_dict[w]] = 1
        return result
    
    # 获取所有主问题的onehot
    def get_one_hot_from_main_question(self):
        l = self.main_question_list
        
        result = np.zeros((len(l),len(self.word_dict)))
        for i,s in enumerate(l):
            for j,w in enumerate(s):
                if w in self.word_dict:
                    result[i,self.word_dict[w]] = 1
        return result
    
    #根据label_Index获取main_question
    def get_main_question_from_label_index(self,label_index):
        return self.main_question_list[label_index]
        
        
    #获取该pair下所有的主问题
    def get_main_question_list(self):
        s = set(self.df['main_question'])
        l = list(s)
        return l