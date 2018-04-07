#-*- coding:utf-8 -*-
import pandas as pd
import random
import time
import numpy as np

class Data_set:
    """
    注意：word_frequence_flag,ngram_flag需要在生成数据集时候确认。
    """
    def __init__(self,data_path,word_frequence_flag,ngram_flag):
        start = time.time()
        self.word_frequence_flag = word_frequence_flag
        self.ngram_flag = ngram_flag
        self.topn_2gram_num = 1000
        self.data_path = data_path
        self.df = pd.read_csv(data_path,encoding='utf-8')
        self.word_dict = self._get_word_dict()
        self.main_question_len = self.get_main_question_num()
        self.main_question_list = self.get_main_question_list()
        self.main_question_one_hot_dict = self.get_one_hot_dict_from_main_question()
        print "data_set_build over,use:"+str(time.time()-start)

        
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
        
    #获取top1000的2gram:获取2元词频字典
    def _get_topn_2gram_word_freq_dict(self):
        l = list(self.df['query'])
        l.extend(list(set(self.df['main_question'])))
        word_dict = {}
        for sentence in l:
            for i in range(len(sentence)-1):
                ngram_word = sentence[i:i+2]
                if ' ' in ngram_word:
                    continue
                if ngram_word in word_dict:
                    word_dict[ngram_word] += 1
                else:
                    word_dict[ngram_word] = 1

        sort_list = sorted(word_dict.items(),key = lambda x:x[1],reverse = True)
        return dict(sort_list[:self.topn_2gram_num ])
      

    #获取词频字典
    def _get_1gram_word_freq_dict(self):
        l = list(self.df['query'])
        l.extend(list(set(self.df['main_question'])))
        string = "".join(l)
        word_dict = {}
        for w in string:
            if w in word_dict:
                word_dict[w] += 1
            else:
                word_dict[w] = 1
        return word_dict

    #建立one_hot编码词典：
    def _get_word_dict(self):
        word_dict = {}
        one_gram_dict = self._get_1gram_word_freq_dict()
        two_gram_dict = self._get_topn_2gram_word_freq_dict()
        index = 0
        for key in one_gram_dict:
            word_dict[key] = (index,one_gram_dict[key])
            index += 1
        for key in two_gram_dict:
            word_dict[key] = (index,two_gram_dict[key])
            index += 1
        return word_dict

    
    #通过索引把df转成词向量
    def get_one_hot_from_batch(self,query_BS):
        #随机抽取100个pair对
        random_index_list = self.get_random_list(query_BS,len(self.df))
        

        #query
        if self.word_frequence_flag and self.ngram_flag:
            query = np.zeros((query_BS,len(self.word_dict)))
            for i,index in enumerate(random_index_list):
                sentence = self.df.iloc[index]['query']
                for j,w in enumerate(sentence):
                    ngram_word = sentence[j:j+2]
                    if w in self.word_dict:
                        query[i,self.word_dict[w][0]] = self.word_dict[w][1]
                    if ngram_word in self.word_dict:
                        query[i,self.word_dict[ngram_word][0]] = self.word_dict[ngram_word][1]
        elif not self.word_frequence_flag and self.ngram_flag:
            query = np.zeros((query_BS,len(self.word_dict)))
            for i,index in enumerate(random_index_list):
                sentence = self.df.iloc[index]['query']
                for j,w in enumerate(sentence):
                    ngram_word = sentence[j:j+2]
                    if w in self.word_dict:
                        query[i,self.word_dict[w][0]] = 1
                    if ngram_word in self.word_dict:
                        query[i,self.word_dict[ngram_word][0]] = 1
        elif self.word_frequence_flag and not self.ngram_flag:
            query = np.zeros((query_BS,len(self.word_dict)))
            for i,index in enumerate(random_index_list):
                sentence = self.df.iloc[index]['query']
                for w in sentence:
                    if w in self.word_dict:
                        query[i,self.word_dict[w][0]] = self.word_dict[w][1]
        elif not self.word_frequence_flag and not self.ngram_flag:
            query = np.zeros((query_BS,len(self.word_dict)))
            for i,index in enumerate(random_index_list):
                sentence = self.df.iloc[index]['query']
                for w in sentence:
                    if w in self.word_dict:
                        query[i,self.word_dict[w][0]] = 1

        #doc_pos
        doc_pos = []
        for i,index in enumerate(random_index_list):
            doc_pos.append(self.main_question_one_hot_dict[self.df.iloc[index]['main_question']])
        doc_pos = np.array(doc_pos)


        #doc_neg
        doc_neg = []
        for i in range(query_BS):
            index = random.randint(0,self.main_question_len-1)
            doc_neg.append(self.main_question_one_hot_dict[self.main_question_list[index]])
        doc_neg = np.array(doc_neg)

        return query,doc_pos,doc_neg
    
 	#外部传入一个df，然后获取batch输出
    def get_one_hot_from_df_in(self,df_in,query_BS):
        #随机抽取100个pair对
        random_index_list = self.get_random_list(query_BS,len(df_in))
        

        #query
        if self.word_frequence_flag and self.ngram_flag:
            query = np.zeros((query_BS,len(self.word_dict)))
            for i,index in enumerate(random_index_list):
                sentence = df_in.iloc[index]['query']
                for j,w in enumerate(sentence):
                    ngram_word = sentence[j:j+2]
                    if w in self.word_dict:
                        query[i,self.word_dict[w][0]] = self.word_dict[w][1]
                    if ngram_word in self.word_dict:
                        query[i,self.word_dict[ngram_word][0]] = self.word_dict[ngram_word][1]
        elif not self.word_frequence_flag and self.ngram_flag:
            query = np.zeros((query_BS,len(self.word_dict)))
            for i,index in enumerate(random_index_list):
                sentence = df_in.iloc[index]['query']
                for j,w in enumerate(sentence):
                    ngram_word = sentence[j:j+2]
                    if w in self.word_dict:
                        query[i,self.word_dict[w][0]] = 1
                    if ngram_word in self.word_dict:
                        query[i,self.word_dict[ngram_word][0]] = 1
        elif self.word_frequence_flag and not self.ngram_flag:
            query = np.zeros((query_BS,len(self.word_dict)))
            for i,index in enumerate(random_index_list):
                sentence = df_in.iloc[index]['query']
                for w in sentence:
                    if w in self.word_dict:
                        query[i,self.word_dict[w][0]] = self.word_dict[w][1]
        elif not self.word_frequence_flag and not self.ngram_flag:
            query = np.zeros((query_BS,len(self.word_dict)))
            for i,index in enumerate(random_index_list):
                sentence = df_in.iloc[index]['query']
                for w in sentence:
                    if w in self.word_dict:
                        query[i,self.word_dict[w][0]] = 1

        #doc_pos
        doc_pos = []
        for i,index in enumerate(random_index_list):
            doc_pos.append(self.main_question_one_hot_dict[df_in.iloc[index]['main_question']])
        doc_pos = np.array(doc_pos)


        #doc_neg
        doc_neg = []
        for i,index in enumerate(random_index_list):
            doc_pos.append(self.main_question_one_hot_dict[df_in.iloc[index]['other_question']])
        doc_neg = np.array(doc_neg)

        return query,doc_pos,doc_neg

    # 获取一个句子的onehot
    def get_one_hot_from_sentence(self,sentence):
        #转化到unicode编码
        if isinstance(sentence,str):
            sentence = sentence.decode('utf-8')
          
        if self.word_frequence_flag and self.ngram_flag:
            query = np.zeros((1,len(self.word_dict)))
            for j,w in enumerate(sentence):
                ngram_word = sentence[j:j+2]
                if w in self.word_dict:
                    query[0,self.word_dict[w][0]] = self.word_dict[w][1]
                if ngram_word in self.word_dict:
                    query[0,self.word_dict[ngram_word][0]] = self.word_dict[ngram_word][1]
        elif not self.word_frequence_flag and self.ngram_flag:
            query = np.zeros((1,len(self.word_dict)))
            for j,w in enumerate(sentence):
                ngram_word = sentence[j:j+2]
                if w in self.word_dict:
                    query[0,self.word_dict[w][0]] = 1
                if ngram_word in self.word_dict:
                    query[0,self.word_dict[ngram_word][0]] = 1
        elif self.word_frequence_flag and not self.ngram_flag:
            query = np.zeros((1,len(self.word_dict)))
            for w in sentence:
                if w in self.word_dict:
                    query[0,self.word_dict[w][0]] = self.word_dict[w][1]
        elif not self.word_frequence_flag and not self.ngram_flag:
            query = np.zeros((1,len(self.word_dict)))
            for w in sentence:
                if w in self.word_dict:
                    query[0,self.word_dict[w][0]] = 1
        return query
    
    # 获取所有主问题的onehot
    def get_one_hot_from_main_question(self):
        l = self.main_question_list

        if self.word_frequence_flag and self.ngram_flag:
            query = np.zeros((len(l),len(self.word_dict)))
            for i,sentence in enumerate(l):
                for j,w in enumerate(sentence):
                    ngram_word = sentence[j:j+2]
                    if w in self.word_dict:
                        query[i,self.word_dict[w][0]] = self.word_dict[w][1]
                    if ngram_word in self.word_dict:
                        query[i,self.word_dict[ngram_word][0]] = self.word_dict[ngram_word][1]
        elif not self.word_frequence_flag and self.ngram_flag:
            query = np.zeros((len(l),len(self.word_dict)))
            for i,sentence in enumerate(l):
                for j,w in enumerate(sentence):
                    ngram_word = sentence[j:j+2]
                    if w in self.word_dict:
                        query[i,self.word_dict[w][0]] = 1
                    if ngram_word in self.word_dict:
                        query[i,self.word_dict[ngram_word][0]] = 1
        elif self.word_frequence_flag and not self.ngram_flag:
            query = np.zeros((len(l),len(self.word_dict)))
            for i,sentence in enumerate(l):
                for w in sentence:
                    if w in self.word_dict:
                        query[i,self.word_dict[w][0]] = self.word_dict[w][1]
        elif not self.word_frequence_flag and not self.ngram_flag:
            query = np.zeros((len(l),len(self.word_dict)))
            for i,sentence in enumerate(l):
                for w in sentence:
                    if w in self.word_dict:
                        query[i,self.word_dict[w][0]] = 1
        return query

    #所有主问题的One_hot预处理加载
    def get_one_hot_dict_from_main_question(self):
        l = self.main_question_list
        
        result_dict = {}
        for sentence in l:
            result_dict[sentence] = self.get_one_hot_from_sentence(sentence).reshape((-1,))
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
        while len(result) < index_num:
            r_int = random.randint(0,index_max-1)
            if r_int not in result:
                result.append(r_int)
        return result
