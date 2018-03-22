import numpy as np
import pandas as pd
import random

#把pair对写入一个新文件
def generate_triple_pair(log_path):
	#处理log文件
	df = pd.read_excel(log_path)
    new_df = pd.DataFrame(columns=['query','main_question','wrong_question'])
    index = 0
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
        self.train_percent = train_percent
        self.df_triple = df
        self.word_dict = self.get_word_dict()
    
    #计算词数
    def get_wordnum(self):
        l = list(self.df_triple['query'])
        l.extend(list(set(self.df_triple['main_question'])))
        l.extend(list(set(self.df_triple['wrong_question'])))
        string = "".join(l)
        s = set(string)
        return len(s)
                         

    #构建字典索引
    def get_word_dict(self):
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
        batch_doc = list(new_df.iloc[index_list][column_name])
        
        result = np.zeros((len(batch_doc),len(self.word_dict)))
        for i,s in enumerate(batch_doc):
            for j,w in enumerate(s):
                if w in self.word_dict:
                    result[i,self.word_dict[w]] = 1

        return result   