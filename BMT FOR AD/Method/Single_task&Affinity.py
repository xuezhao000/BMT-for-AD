#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
#import albumentations  #数据增强
import pandas as pd
from collections import namedtuple
from tensorflow.keras.initializers import glorot_uniform
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense,Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
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
from scipy.stats import spearmanr


# In[3]:


#下列超参数均需根据实际情况进行修改
tasks = ['CDRSB_bl','ADAS13_bl','MMSE_bl','FAQ_bl','MOCA_bl','FDG_bl','AV45_bl']
#target_size = (218,178)
epochs = 30
batch_size =12
INIT_LR=0.001
optimizer = Adam(learning_rate=INIT_LR)
#input_shape = 13574
kernel_initializer='glorot_uniform'


# In[4]:


import warnings
warnings.filterwarnings("ignore")


# In[5]:


target_folder='/home/j304011/xz/type2_data_1/model_weight/not5_share5csv/'#储存任务模型参数文件夹
npy_floder = '/home/j304011/xz/type2_data_1/npy/not5_share5csv/'#储存任务间亲和度文件夹
#num_iterations =1
start_time = time.time()
pearson_all_100test = []
for fff in range(1,101):
    def load_phenotype(phenotype_excel_path):
        phenotype_excel_path = phenotype_excel_path
        phenotype_data = pd.read_csv(phenotype_excel_path)
        phe_ids = phenotype_data.iloc[:, 0].astype(str).tolist()  #ID
        labels = phenotype_data.iloc[:, 1:].to_numpy()   #数据
        #scaler = MinMaxScaler()
        #scaler = StandardScaler()
        #train_labels = scaler.fit_transform(labels)
        #indexed_data = {phe_ids[i]:train_labels[i] for i in range(len(phe_ids))}
        trainY_id,result_list1_id,trainY,result_list2 = train_test_split(phe_ids, labels, test_size=0.2)#,random_state = fff
        valY_id,testY_id,valY,test_Y = train_test_split(result_list1_id, result_list2, test_size=0.5)
        return trainY_id,valY_id,testY_id,trainY,valY,test_Y
    phenotype_excel_path=f'/home/j304011/xz/type2_data_1/simulation_data/not5_share5csv/y_not5_share5_{fff-1}.csv'#表型文件夹位置，如果唯一则不需要循环，此处为模拟数据写法，故存在循环
    trainY_id,valY_id,testY_id,trainY,valY,test_Y = load_phenotype(phenotype_excel_path)
    trainY = trainY.tolist()
    valY = valY.tolist()
    test_Y = test_Y.tolist()
    concatenate_X = []
    concatenate_trainX = []
    concatenate_valX = []
    concatenate_testX = []
    for a in range(1,57):
        def load_dataset(file_path):
            # 加载基因数据
            file_path = file_path
            gene_data = pd.read_csv(file_path, sep=' ')
            gene_ids = gene_data.iloc[:, 1].astype(str).tolist() #ID
            gene_data = gene_data.iloc[:, 6:].to_numpy() 
            return gene_ids,gene_data
        
        file_path =f'/home/j304011/xz/type2_data_1/txt_file/gene_{a}.txt'#SNPs文件夹位置
        gene_ids,gene_data = load_dataset(file_path)
        indexed_data = {gene_ids[i]:gene_data[i] for i in range(len(gene_ids))}
        train_gene_data = []
        for i in trainY_id:
            data = indexed_data[i]
            train_gene_data.append(data)
        train_gene_data = np.array(train_gene_data)
        concatenate_trainX.append(train_gene_data)
        print(i)
        val_gene_data = []
        for i in valY_id:
            data = indexed_data[i]
            val_gene_data.append(data)
        val_gene_data = np.array(val_gene_data)
        concatenate_valX.append(val_gene_data)
        test_gene_data = []
        for i in testY_id:
            data = indexed_data[i]
            test_gene_data.append(data)
        test_gene_data = np.array(test_gene_data)
        concatenate_testX.append(test_gene_data)
        input_shape = len(gene_data[0])
        def gene_model_Multi(input_shape):

            if input_shape > 32:
                nunit1 = min(16,math.ceil(input_shape/2))

            elif input_shape < 11:
                nunit1 = input_shape

            else:
                nunit1 = input_shape // 2  # 将nunit1范围控制在[5,25]

            inputs = Input(shape=input_shape)
            if input_shape > 100:
                x = Dropout(0.5)(inputs)
                x = Dense(units=nunit1,activation='sigmoid')(x)
            else:
                x = Dense(units=nunit1,activation='sigmoid')(inputs)

            outputs = Dense(7)(x)
            model = Model(inputs=inputs, outputs = outputs, name=f'gene_{a}_model2')
            return model
        model = gene_model_Multi(input_shape=input_shape)
        #model.summary()
        save_model(model, f'gene_{a}_model2.h5')
    models = [load_model(f'gene_{a}_model2.h5') for a in range(1,57)]
    outputs = [model.layers[-2].output for model in models]
    inputs = [model.input for model in models]
    merged = concatenate(outputs)
    x = Dropout(0.3)(merged)
    #x = Dense(512,activation='sigmoid')(x)
    #x = Dropout(0.2)(x)
    x = Dense(128,activation='sigmoid')(x)
    x = Dropout(0.2)(x)
    x = Dense(32,activation='sigmoid')(x)
    outputs = Dense(1)(x)
    new_model = Model(inputs=inputs, outputs=outputs)
    correlation_coefficients = []
    for task in tasks:
        new_model.compile(optimizer='adam', loss='mean_squared_error')
    #new_model.summary()
    #new_model.compile(optimizer='adam', loss='mean_squared_error')
        checkpointer = ModelCheckpoint(filepath='best_model1.hdf5',
                                monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        reduce = ReduceLROnPlateau(monitor='val_loss', patience=7,
                                verbose=1,
                                factor=0.5,
                                min_lr=1e-6)
        b = tasks.index(task)
        trainY_final = [data[b] for data in trainY]
        valY_final = [data[b] for data in valY]
        test_Y_final = [data[b] for data in test_Y]

        history = new_model.fit(concatenate_trainX,np.array(trainY_final),
                            steps_per_epoch=len(trainY_final) // batch_size,
                            validation_data=(concatenate_valX,np.array(valY_final)),
                            validation_steps=len(valY_final) / batch_size,
                            batch_size=batch_size,
                            callbacks=[checkpointer, reduce],
                            epochs=epochs,verbose=0)
        new_model.save(task+str(fff)+ '.h5')    #task+str(fff)+ '.h5'
        shutil.move(task+str(fff)+ '.h5', target_folder)
        preds = new_model.predict(concatenate_testX)
        test_Y_final = np.asarray(test_Y_final)
        preds = np.asarray(preds)
        test_Y_final = test_Y_final.flatten()
        preds = preds.flatten()
        correlation, _ = pearsonr(test_Y_final, preds)
        correlation_coefficients.append(correlation)
        tf.keras.backend.clear_session()
    pearson_all_100test.append(correlation_coefficients)
    #tf.keras.backend.clear_session()
    k = len(concatenate_trainX)
    def get_model_RDM(Layers):
        model_RDM = []
    # 加载模型
        for task in tasks:#此处的8应该修改为trained_model列表中元素的数量。
            model = load_model(target_folder+task+str(fff)+ '.h5')  # 这里是加载已训练好的resnet50模型的代码
            #sub_model = model.get_layer('gene_model_Multi')
            #layers = layers  # 以第4个block中的第6个卷积层为例
            filenames = []
            for aa in Layers:
                #layer = sub_model.layers[a]
                layer = model.layers[aa]
                #intermediate_layer_model = tf.keras.Model(inputs=sub_model.input,
                #                                  outputs=layer.output)
                intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                                  outputs=layer.output)            
        # 使用加载好的模型进行预测
                features = intermediate_layer_model.predict(concatenate_trainX)
        # 输出指定block之后的层的输出    
                output_filename = f"block_{Layers.index(aa)}_output1.npy"
            
                np.save(output_filename, features)
                filenames.append(output_filename)
            #print('qaq')
            img_list = []
            for bb in range(len(Layers)):#此处的4为列表layers的元素数量
                img_file = f"block_{bb}_output1.npy"  # 根据文件名规律构造文件名
                img = np.load(img_file)
                img_list.append(img)
            correlations = []
            for c in range(len(img_list)):
                corr_matrix = np.zeros((k,k))#此处的8对应最初的待处理图片的数量 # 初始化相关系数矩阵
                for l in range(k):
                    for m in range(k):
                        img1 = img_list[c][l].reshape(-1)
                        img2 = img_list[c][m].reshape(-1)
                        corr_coeff, _ = pearsonr(img1, img2)            
                        corr_matrix[l, m] = corr_coeff
                
                correlations.append(1-corr_matrix)
            model_RDM.append(correlations)
    # 输出相关系数矩阵列表
        return model_RDM
    Layers = [-5,-4,-3,-2]#此代码仅展示当有4个分支点时的计算，且此处指定其分支点位置如前，如需修改分支点个数和分支点位置则需进行修改
    model_RDM = get_model_RDM(Layers)
    def get_spearman(model_RDM):
        data = np.array(model_RDM)
        list1 = []
        for fir_layer in data:
            
            list2 = []
            for sec_layer in fir_layer:
                list3 = []
                for i in range(len(sec_layer)):
                    for j in range(i):
                        list3.append(sec_layer[i][j])
                list2.append(list3)
            list1.append(list2)
        RDM_LUT = np.array(list1)#RDM_LUT指的是RDM的下三角部分
        spear_matrix = []
        for i in range(len(RDM_LUT)):
            for j in range(len(RDM_LUT)):
            
                arr1 = RDM_LUT[i]
                arr2 = RDM_LUT[j]
                spear_corr = []
                for k in range(arr1.shape[0]):
                    r,_ = spearmanr(arr1[k],arr2[k])
                    spear_corr.append(r)
                #print(spear_corr)        
                spear_matrix.append(spear_corr)
        spear_matrix_arr = np.array(spear_matrix)
        spearman = spear_matrix_arr.reshape(len(tasks),len(tasks),len(Layers))
        return spearman
    spearman = get_spearman(model_RDM)
    np.save('not5_share5_'+str(fff)+'.npy',spearman)#储存的文件名需根据数据不同进行修改
    shutil.move('not5_share5_'+str(fff)+'.npy', npy_floder)
    print(fff)
end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time  # 计算时间差，即代码执行时间
print(f"共计用时{elapsed_time}")
print(pearson_all_100test)


# In[6]:


#保存文件
import csv
csv_file_path ='/home/j304011/xz/type2_data_1/result/not5_share5.csv'#此为保存单任务模型性能的文件夹集及文件名位置
flat_list = [[value.item() if isinstance(value, np.ndarray) else value for value in sublist] for sublist in pearson_all_100test]
# 以写入模式打开 CSV 文件
with open(csv_file_path, mode='w', newline='') as file:
    # 创建 CSV writer 对象 
    writer = csv.writer(file)
    
    # 写入列表数据
    writer.writerows(flat_list)


# In[ ]:




