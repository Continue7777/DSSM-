#-*- coding:utf-8 -*-
import pandas as pd
import random
import time
import numpy as np

class Data_set:
    def __init__(self,data_path,sequence_size):
        self.sequence_size = sequence_size
        self.data_path = data_path
        self.df = pd.read_csv(data_path,encoding='utf-8')
        self.word_dict = self._get_word_dict()
        self.main_question_len = self.get_main_question_num()
        self.main_question_list = self.get_main_question_list()

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
        
    #构建字典索引,0表示无
    def _get_word_dict(self):
        l = list(self.df['query'])
        l.extend(list(set(self.df['main_question'])))
        string = "".join(l)
        s = set(string)
        word_dict = {}
        for i,w in enumerate(s):
            word_dict[w] = i+1
        return word_dict
    
    #获取query_index
    def get_query_index(self,query):
        result_array = np.zeros((1,self.sequence_size))
        if type(query) == str:
            query=query.decode('utf-8')
        for i,w in enumerate(query):
            if i >= self.sequence_size -1:
                break
            if w in self.word_dict:
                result_array[0,i] = self.word_dict[w]
        return result_array

    #获取一批query_index
    def get_batch_index(self,query_BS):
        random_index_list = self.get_random_list(query_BS,len(self.df))

        query = np.zeros((query_BS,self.sequence_size))
        for i,index in enumerate(random_index_list):
            sentence = self.df.iloc[index]['query']
            for j,w in enumerate(sentence):
                if j < self.sequence_size:
                    query[i,j] = self.word_dict[w]

        doc_pos = np.zeros((query_BS,1))
        for i,index in enumerate(random_index_list):
            doc_pos[i,0] = self.main_question_list.index(self.df.iloc[index]['main_question'])

        doc_neg = np.zeros((query_BS,1))
        for i in range(query_BS):
            index = random.randint(0,self.main_question_len-1)
            doc_neg[i,0] = index
        return query,doc_pos,doc_neg

    #根据label_Index获取main_question
    def get_main_question_from_label_index(self,label_index):
        return self.main_question_list[label_index]
        
        
    #获取该pair下所有的主问题
    def get_main_question_list(self):
        s = set(self.df['main_question'])
        l = list(s)
        return l

    def get_random_list(self,index_num,index_max):
        result = []
        while len(result) < index_num:
            r_int = random.randint(0,index_max-1)
            if r_int not in result:
                result.append(r_int)
        return result