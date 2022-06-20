# -*- coding: utf-8 -*-
"""
Created on 17 June 2022

@author: Shubham
"""

import numpy as np
import pickle
from keras.models import model_from_json
import pandas as pd
import datetime
import re
import warnings
warnings.filterwarnings('ignore')
class final:
    def load_files(self):
        #https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("model.h5")

        with open('access_enc.pkl','rb') as file:
            self.access_enc=pickle.load(file)

        with open('lang_enc.pkl','rb') as file:
            self.lang_enc=pickle.load(file)

        with open('spider_enc.pkl','rb') as file:
            self.agent_enc=pickle.load(file)
        self.new_data=pd.read_csv('final_data.csv')    
    def find_access(self,page):
        k=max([i.start() for i in re.finditer('org_',page)])   #https://www.geeksforgeeks.org/python-all-occurrences-of-substring-in-string/
        if('all-access' in page[k:]):
            access='all_access'
        if('desktop' in page[k:]):
            access='desktop'
        if('mobile' in page[k:]): 
            access='mobile'
        k=access    
        access=self.access_enc.transform([access]).reshape(1,1)    
        return access,k 
    def find_lang(self,page):
        index=page.find('.wikipedia')
        lang=page[index-1:index-3:-1][::-1]
        lang_dict={'de':'German','en':'English', 'es':'Spanish', 'fr':'French', 'ja':'Japanese', 'nt':'Media', 'ru':'Russian', 'zh':'Chinese'}
        language=lang_dict[lang]
        lang=self.lang_enc.transform([lang]).reshape(1,1)
        return lang,language
    def find_agent(self,page):
        if('spider' in page):
            spider='spider'
        else:
            spider='non-spider' 
        k=spider  
        agent=self.agent_enc.transform([spider]).reshape(1,1)
        return agent,k
    def find_data(self,ind,date):
        data=self.new_data.iloc[ind].values
        date1=datetime.date(2015,7,6)
        k=date.split('-')
        date2=datetime.date(int(k[0]),int(k[1]),int(k[2]))
        dif=(date2-date1).days
        data=np.log1p(data[dif+1:dif+6].astype(int))
        data=np.array(data).reshape(1,5,1)
        return data     
    def predict(self,ind,date):
        self.load_files()
        start=datetime.datetime.now()
        self.page=self.new_data['Page'].values[int(ind)]
        access,access1=self.find_access(self.page)
        lang,language=self.find_lang(self.page)
        agent,agent1=self.find_agent(self.page)
        data=self.find_data(int(ind),date)
        time=datetime.datetime.now()-start
        predicted=int(np.round(np.expm1(self.model.predict([data,access,lang,agent])[0])[0]))
        return(access1,agent1,language,predicted,time,self.page)
