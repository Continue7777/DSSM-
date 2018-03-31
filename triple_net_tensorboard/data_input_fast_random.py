    #-*- coding:utf-8 -*-
    import pandas as pd
    import random
    import time
    import numpy as np

    class Data_set:
        def __init__(self,data_path):
            start = time.time()
            self.data_path = data_path
            self.df = pd.read_csv(data_path,encoding='utf-8')
            end = time.time()
            print "read over,time:" + str(end-start)
            self.word_dict = self._get_word_dict()
            end1 = time.time()
            print "word_dict_num:" + str(len(self.word_dict))
            print "build dict over,time:" + str(end1-end)
            self.main_question_len = self.get_main_question_num()

            self.main_question_list = self.get_main_question_list()
            self.main_question_one_hot_dict = self.get_one_hot_dict_from_main_question()

            
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
        def get_one_hot_from_batch(self,query_BS):
            #随机抽取100个pair对
            random_index_list = self.get_random_list(query_BS,len(self.df))
            

            #query
            query = np.zeros((query_BS,len(self.word_dict)))
            for i,index in enumerate(random_index_list):
                for w in self.df.iloc[index]['query']:
                    if w in self.word_dict:
                        query[i,self.word_dict[w]] = 1

            #doc_pos
            doc_pos = []
            for i,index in enumerate(random_index_list):
                doc_pos.append(self.main_question_one_hot_dict[self.df.iloc[index]['main_question']])
            doc_pos = np.array(doc_pos)


            #doc_neg
            doc_neg = []
            for i in range(query_BS):
                index = random.randint(0,self.main_question_len)
                doc_neg.append(self.main_question_one_hot_dict[self.df.iloc[index]['main_question']])
            doc_neg = np.array(doc_neg)

            return query,doc_pos,doc_neg
        
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

        #所有主问题的One_hot预处理加载
        def get_one_hot_dict_from_main_question(self):
            l = self.main_question_list
            
            result_dict = {}
            one_hot_result = np.zeros(len(self.word_dict))
            for i,s in enumerate(l):
                for j,w in enumerate(s):
                    if w in self.word_dict:
                        one_hot_result[self.word_dict[w]] = 1
                result_dict[s] = one_hot_result
            return result_dict
        
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
            for i in range(index_num):
                result.append(random.randint(0,index_max-1))
            return result
