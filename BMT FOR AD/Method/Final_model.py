#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#导入函数
import numpy as np
from tensorflow.keras.optimizers import Adam   #加载优化器
import cv2   #计算机视觉库
from tensorflow.keras.preprocessing.image import img_to_array  #整数转化为浮点数
from sklearn.model_selection import train_test_split   #用来划分训练集和测试集

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
#回调函数训练可视化    ModelCheckpoint：该回调函数将在每个epoch后保存模型到filepath
# ReduceLROnPlateau：更新学习率

import os  #os模块：Python 程序与操作系统进行交互的接口

import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten  #导入全连接层
from tensorflow.keras.models import Sequential  #作为容器包装各层
from tensorflow.keras.models import Model 
from tensorflow.keras.models import load_model
from tensorflow import keras
import albumentations  #数据增强
import pandas as pd
from collections import namedtuple
from tensorflow.keras.initializers import glorot_uniform
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense,Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler


# In[ ]:


import math
import shutil
from keras.models import Sequential
from keras.layers import Dense
import time
from scipy.stats import pearsonr
import random
from tensorflow.keras.models import load_model
from keras.layers import concatenate
from keras.models import save_model
from sklearn.metrics import mean_squared_error


# In[ ]:


#ljq
tasks = ['CDRSB_bl','ADAS13_bl','MMSE_bl','FAQ_bl','MOCA_bl','FDG_bl','AV45_bl']
#target_size = (218,178)
epochs = 30
batch_size =12
INIT_LR=0.001
optimizer = Adam(learning_rate=INIT_LR)
#input_shape = 13574
kernel_initializer='glorot_uniform'


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


target_folder='C:/Users/ASUS/100seed_model_weight/concat分组后LD0.9sigmoid_54/'#最终模型参数储存位置
num_iterations = range(1,101)

start_time = time.time()
pearson_all_100test_pearsonr = []
pearson_all_100test_mse = []
for fff in num_iterations:
    #target_folder= f'C:/100seed/seed_{fff+1}/'
    concatenate_trainX = []
    concatenate_valX = []
    concatenate_testX = []
    for a in range(1,57):
        def load_dataset(file_path,phenotype_excel_path):
            # 加载基因数据
            file_path = file_path
            gene_data = pd.read_csv(file_path, sep=' ')
            gene_ids = gene_data.iloc[:, 1].astype(str).tolist() #ID
            train_gene_data = gene_data.iloc[:, 6:].to_numpy()    #数据
        
            # 加载表型数据
            phenotype_excel_path = phenotype_excel_path
            phenotype_data = pd.read_csv(phenotype_excel_path)
            phe_ids = phenotype_data.iloc[:, 0].astype(str).tolist()  #ID
            labels = phenotype_data.iloc[:, 1:].to_numpy()   #数据
            #scaler = MinMaxScaler()
            scaler = StandardScaler()
            train_labels = scaler.fit_transform(labels)
            indexed_data = {phe_ids[i]:train_labels[i] for i in range(len(phe_ids))}
            #indexed_data = {phe_ids[i]:labels[i] for i in range(len(phe_ids))}
            # 匹配基因和表型数据相同的ID
            train_select_data = []
            for i in gene_ids:
                train_data = indexed_data[i]
                train_select_data.append(train_data)
        
            
            #train_data = pd.DataFrame(train_select_data, columns=train_gene_data.columns)
            
            #对表型数据进行Min-Max归一化
        
            
            #final_train = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(buffer_size=10, seed=0).batch(batch_size, drop_remainder=True)    
            
            Dataset = namedtuple('Dataset', ['train_data','train_labels'])
            return Dataset(train_gene_data,train_select_data)
        dataset = load_dataset(file_path =f'C:/Users/ASUS/Desktop/genedata/real_data/LD0.9/qiegene_ld0.9/txt_file/gene_{a}.txt',
                              phenotype_excel_path = 'C:/Users/ASUS/Desktop/genedata/real_data/57gene/snp/7任务完整表型463样本.csv')
        print(a)
        data = np.array(dataset.train_data)
        #labels = np.array(dataset.train_labels)
        labels = pd.DataFrame(dataset.train_labels, columns=tasks)
        has_nan = np.isnan(labels).any(axis=1)
        filtered_labels = labels[~has_nan]
        filtered_data = data[~has_nan]
        input_shape = len(filtered_data[0])
        #print(len(filtered_data))
        #print(len(filtered_labels))
        #unit1 = min(64,math.ceil(input_shape/2))
        #unit2 = min(32,math.ceil(input_shape/4))
        trainX,result_list1,trainY,result_list2 = train_test_split(filtered_data, filtered_labels, test_size=0.2,random_state = fff)
        valX,test_X,valY,test_Y = train_test_split(result_list1, result_list2, test_size=0.5,random_state=fff)
        concatenate_trainX.append(trainX)
        concatenate_valX.append(valX)
        concatenate_testX.append(test_X)
        def gene_model_Multi(input_shape):
            #nunit1=50
            #nunit2=10
            
            if input_shape > 32:
                nunit1 = min(16,math.ceil(input_shape/2))
                #nunit2 = min(16,math.ceil(input_shape/4))
                #nunit1 = max(input_shape // 2, nunit1)
                #nunit2 = max(input_shape // 4, nunit2)
                #nunit1 = min(200, nunit1)  # 将nunit1范围控制在[50,200]， >400，则为200；100-400，则为值的1/2；50-100，则为50。
                #nunit2 = min(80, nunit2)  # 将nunit2范围控制在[12,80]
            elif input_shape < 11:
                nunit1 = input_shape
                #nunit2 = input_shape
            else:
                nunit1 = input_shape // 2  # 将nunit1范围控制在[5,25]
                #nunit2 = input_shape // 4  # 将nunit2范围控制在[3,12]
                
            #print(nunit1)
            #print(nunit2)
            #unit1 = min(64,math.ceil(input_shape/2))
            #unit2 = min(32,math.ceil(input_shape/4))
            inputs = Input(shape=input_shape)
            if input_shape > 100:
                x = Dropout(0.5)(inputs)
                x = Dense(units=nunit1,activation='sigmoid')(x)
            else:
                x = Dense(units=nunit1,activation='sigmoid')(inputs)
            #if nunit1 >20:
            #    x = Dropout(0.2)(x)
            #x = Dense(units=nunit2,activation='sigmoid')(x)
            outputs = Dense(7)(x)
            model = Model(inputs=inputs, outputs = outputs, name=f'gene_{a}_model')
            return model
        model = gene_model_Multi(input_shape=input_shape)
        #model.summary()
        save_model(model, f'gene_{a}_model.h5')
    models = [load_model(f'gene_{a}_model.h5') for a in range(1,57)]
    outputs = [model.layers[-2].output for model in models]
    inputs = [model.input for model in models]
    merged = concatenate(outputs)
    x = Dropout(0.3)(merged)
    #x = Dense(512,activation='sigmoid')(x)
    #x = Dropout(0.2)(x)
    x = Dense(128,activation='sigmoid')(x)
    x = Dropout(0.2)(x)
    
    
    
    x1_1= Dense(32,activation='sigmoid')(x)
    #x1_2= Dense(32,activation='sigmoid')(x)
    #x1_3= Dense(32,activation='sigmoid')(x)
    #x1_4= Dense(32,activation='sigmoid')(x)
    #x1_5= Dense(32,activation='sigmoid')(x)
    
    x2_1= Dense(32,activation='sigmoid')(x1_1)
    x2_2= Dense(32,activation='sigmoid')(x1_1)
    x2_3= Dense(32,activation='sigmoid')(x1_1)
    #x2_4= Dense(32,activation='sigmoid')(x1_3)
    #x2_5= Dense(32,activation='sigmoid')(x1_2)
    #x2_6= Dense(32,activation='sigmoid')(x1_1)
    #x2_7= Dense(32,activation='sigmoid')(x1_2)
    
    output_1 = Dense(1)(x2_1)
    output_2 = Dense(1)(x2_1)
    output_3 = Dense(1)(x2_2)     
    output_4 = Dense(1)(x2_1)
    output_5 = Dense(1)(x2_1)
    output_6 = Dense(1)(x2_3) 
    output_7 = Dense(1)(x2_1)
    outputs = [output_1, output_2, output_3, output_4, output_5, output_6, output_7]
    outputs = concatenate(outputs, name='concatenated_outputs')
    new_model = Model(inputs=inputs, outputs=outputs)
    new_model.compile(optimizer='adam', loss='mean_squared_error')
    checkpointer = ModelCheckpoint(filepath='best_model.hdf5',
                            monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce = ReduceLROnPlateau(monitor='val_loss', patience=7,
                            verbose=1,
                            factor=0.5,
                            min_lr=1e-6)
    history = new_model.fit(concatenate_trainX,np.array(trainY),
                        steps_per_epoch=len(trainY) // batch_size,
                        validation_data=(concatenate_valX,np.array(valY)),
                        validation_steps=len(valY) / batch_size,
                        batch_size=batch_size,
                        callbacks=[checkpointer, reduce],
                        epochs=epochs,verbose=0)
    new_model.save('multi_task'+str(fff)+ '.h5')    #task+str(fff)+ '.h5'
    shutil.move('multi_task'+str(fff)+ '.h5', target_folder)
    preds = new_model.predict(concatenate_testX)
    test_Y = np.asarray(test_Y)
    preds = np.asarray(preds)
    #print(preds.shape)
    #print(test_Y.shape)
    #print(preds)
    #print(test_Y)
    correlation_coefficients_pearsonr = []
    correlation_coefficients_mse = []
    for task in range(len(tasks)):
    # 获取当前任务的两个数组
        task_array1 =test_Y [:,task]
        task_array2 = preds[:,task]
        #print(task_array1.shape)
        #print(task_array2.shape)
    # 计算 Pearson 相关系数
        correlation, _ = pearsonr(task_array1, task_array2)
        per_mse = mean_squared_error(task_array1, task_array2)
        correlation_coefficients_pearsonr.append(correlation)
        correlation_coefficients_mse.append(per_mse)
    #print(correlation_coefficients)
    pearson_all_100test_pearsonr.append(correlation_coefficients_pearsonr)
    pearson_all_100test_mse.append(correlation_coefficients_mse)
    ###############################################################################################
    ##基因重要性评分
    ## 存储每个特征的重要性
    #feature_importances_pearson = []
    #feature_importances_mse = []
    tf.keras.backend.clear_session()
    print(fff)
end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time  # 计算时间差，即代码执行时间
print(f"共计用时{elapsed_time}")
print(pearson_all_100test_pearsonr)
print(pearson_all_100test_mse)


# In[ ]:


#保存文件
import csv
csv_file_path ='C:/Users/ASUS/Desktop/multi_task_pearson.csv' 
flat_list = [[value.item() if isinstance(value, np.ndarray) else value for value in sublist] for sublist in pearson_all_100test_pearsonr]
# 以写入模式打开 CSV 文件
with open(csv_file_path, mode='w', newline='') as file:
    # 创建 CSV writer 对象 
    writer = csv.writer(file)
    
    # 写入列表数据
    writer.writerows(flat_list)

