import pandas as pd
import random
import numpy as np

#把pair对写入一个新文件
log_path = 'data_part1.xlsx'
def generate_triple_pair(log_path):
    df = pd.read_excel(log_path)
    new_df = pd.DataFrame(columns=['query','main_question','wrong_question'])
    index = 0
    main_question_list = list(set(df['main_question']))
    main_question_len = len(main_question_list)
    for i in range(len(df)):
        query,main_question = df.iloc[i,:]
        other_question = random.sample(main_question_list,int(main_question_len * 0.3))
        if main_question in other_question:
            other_question.remove(main_question)
        for w_q in other_question:
            new_df.loc[index]={'query':query,'main_question':main_question,'wrong_question':w_q}
            index += 1
        if i%100 == 0:
            print i
    #写到excel里面
    # new_df.to_excel('triple_pair.xlsx')

class Data_set:
    def __init__(self,data_path,train_percent=0.7):
        self.data_path = data_path
        self.df_triple = pd.read_excel(data_path)
        self.word_dict = self._get_word_dict()
        self.train_index_list,self.test_index_list = self._spilt_train_test_index(train_percent)
    
    #获取词数
    def get_word_num(self):
        return len(self.word_dict)
     
    #获取训练集测试集大小
    def get_train_test_size(self):
        return len(self.train_index_list),len(self.test_index_list)
    
    #获取主问题数:wrong+main的并集总数
    def get_main_question_num(self):
        l = list(self.df_triple['main_question'])
        l.extend(list(self.df_triple['wrong_question']))
        s = set(l)
        return len(s)
    
    def _spilt_train_test_index(self,train_percent):
        train_index_list = []
        test_index_list = []
        for i in range(len(self.df_triple)):
            if random.random() < train_percent:
                train_index_list.append(i)
            else:
                test_index_list.append(i)
        return train_index_list,test_index_list
        
    #构建字典索引
    def _get_word_dict(self):
        l = list(self.df_triple['query'])
        l.extend(list(set(self.df_triple['main_question'])))
        l.extend(list(set(self.df_triple['wrong_question'])))
        string = "".join(l)
        s = set(string)
        word_dict = {}
        for i,w in enumerate(s):
            word_dict[w] = i
        return word_dict
    
    #通过索引把df转成词向量
    def get_onehot_from_batch(self,index_list,column_name):
        batch_doc = list(self.df_triple.iloc[index_list][column_name])
        
        result = np.zeros((len(batch_doc),len(self.word_dict)))
        for i,s in enumerate(batch_doc):
            for j,w in enumerate(s):
                if w in self.word_dict:
                    result[i,self.word_dict[w]] = 1

        return result 