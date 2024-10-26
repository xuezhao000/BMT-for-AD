#!/usr/bin/env python
# coding: utf-8

# In[31]:


#导入函数
import csv
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
#from sklearn.metrics import f1_score
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
from sklearn.metrics import mean_squared_error


# In[32]:


#超参数需要调整，但需和Single_task&Affinity.py文件一致
tasks = ['CDRSB_bl','ADAS13_bl','MMSE_bl','FAQ_bl','MOCA_bl','FDG_bl','AV45_bl']
#target_size = (218,178)
epochs =30
batch_size =12
INIT_LR=0.001
optimizer = Adam(learning_rate=INIT_LR)
#input_shape = 13574
kernel_initializer='glorot_uniform'
total_tasks = 7#此处为需要进行分组的任务总数！！！

# In[1]:


#此部分代码用于迭代计算每一层的多个任务可能的组合数
def generate_task_combinations(total_tasks, num_groups, combination, combinations):
    # 如果所有组都已分配完毕，则将当前组合添加到结果集合中
    if num_groups == 0:
        if sum(combination) == total_tasks:
            combinations.add(tuple(sorted(combination)))
        return    
    # 在当前组分配任务数量
    for i in range(1, total_tasks - num_groups + 2):
        # 将一个任务数量分配给当前组
        new_combination = combination + [i]
        # 递归处理剩余的组和任务数量
        generate_task_combinations(total_tasks, num_groups - 1, new_combination, combinations)
# 测试示例

num_groups_size = list(range(1,total_tasks+1))
#print(num_groups_size)#此行输出表示所有的元素可能聚为几组

combinations = set()
for num_groups in num_groups_size:
    generate_task_combinations(total_tasks, num_groups, [], combinations)
# 输出结果
    groups_list = [list(combination) for combination in combinations]
#print(groups_list)#此行输出表示分好的组中有几个任务


# In[2]:


#此部分将任务用小写字母代替，用来生成真实可能的组合情况，为确保之前计算的亲和度能够使用，
#字母的顺序应该与训练任务时总的任务组合tasks中的每个任务顺序相对应。
import copy
def partition_tasks(tasks, groups, current_partition, results):
    # 如果所有任务都已分配完毕，则将当前的分组情况添加到结果列表中
    if len(tasks) == 0:
        results.append(copy.deepcopy(current_partition))
        return    
    # 逐个将任务分配到不同的组中
    for i in range(len(groups)):
        if groups[i] > 0:
            # 将一个任务分配到当前组
            current_partition[i].append(tasks[0])
            groups[i] -= 1
            # 递归处理剩余的任务和组
            partition_tasks(tasks[1:], groups, current_partition, results)
            # 恢复状态，以便尝试其他分组方式
            current_partition[i].remove(tasks[0])
            groups[i] += 1
def remove_duplicates(lst):
        seen = set()
        result = []
        for sublist in lst:
            sublist.sort()  # 对子列表进行排序
            sublist_tuple = tuple(tuple(inner_list) for inner_list in sublist)
            if sublist_tuple not in seen:
                seen.add(sublist_tuple)
                result.append([list(inner_list) for inner_list in sublist])
        return result
# 创建一个空列表用于存储结果
results = []
# 测试示例
#tasks = ['CDRSB_bl','ADAS13_bl','MMSE_bl','FAQ_bl','MOCA_bl','FDG_bl','AV45_bl']#字母数量与任务总数相等, 'e', 'f','g'
for groups in groups_list:
    current_partition = [[] for _ in range(len(groups))]

# 调用函数进行分组，并将结果保存到结果列表中
    partition_tasks(tasks, groups, current_partition, results)    
    unique_list = remove_duplicates(results)
#print(len(unique_list))#共有多少种情况
# 打印去重后的列表
#for sublist in unique_list:
    #print(sublist) #列出所有情况 


# In[47]:


task_to_index = {}
index = 0
for group in ['CDRSB_bl','ADAS13_bl','MMSE_bl','FAQ_bl','MOCA_bl','FDG_bl','AV45_bl']: 
    task_to_index[group] = index
    index += 1

# 打印每个字符串与数字对应的映射关系
for task, index in task_to_index.items():
    print(f"{task}: {index}")


# In[49]:

group_list = []
import numpy as np 
for fff in range(1,101):
    data = np.load( f'/home/j304011/xz/type2_data_1/npy/not5_share5csv/not5_share5_'+str(fff)+'.npy')#此处为Single_task&Affinity.py文件储存的数据
#计算选取此层中，任务组与组间不相似度最高（即组间相似度最低的分组情况）
    def max_scores(task_group_list,affinity_scores):
        group_scores_list = []
        task_group_list = task_group_list
        for i in range(len(task_group_list)):
            task_group = task_group_list[i]
            if len(task_group)== 1:
                continue
            affinity_scores = affinity_scores
            score_sum = 0.0
            count = (len(task_group)*(len(task_group)-1))/2
    
            for i in range(len(task_group)):
                for j in range(i+1, len(task_group)):
                    task_group1 = task_group[i]  # 获取第一个组的任务名
                    task_group2 = task_group[j]  # 获取第二个组的任务名
            
            
                    corr_score = []
            # 遍历两个组内的所有任务组合，并计算亲和度分数的和
                    for task1 in task_group1:
                        for task2 in task_group2:
                            index1 = task_to_index[task1]  # 获取任务1的索引值
                            index2 = task_to_index[task2]  # 获取任务2的索引值             
                            corr_score.append(affinity_scores[index1][index2])  # 提取亲和度分数
                    score = max(corr_score)#先取两组间任务与任务最亲和的两个任务，保证其他任务在组间比这两个更不相似
                    score_sum+= score
            group_scores = score_sum / count
            group_scores_list.append(group_scores)
        #print(min(group_scores_list))#计算组间最小相似的分组情况
        min_index = group_scores_list.index(min(group_scores_list))
        #print(task_group_list[min_index])
        # 输出对应的task_group
        return task_group_list[min_index]
    task_group_list = unique_list
    affinity_scores = data[:,:,3]#此处选择4层列表的最后一层，即位于最靠近输出层的分支点
    #affinity_scores = data
    #print(i)
    max_scores_1 = max_scores(task_group_list,affinity_scores)  #输出计算的最小值和对应的分组情况
    group_list.append(max_scores_1)
    a = len(max_scores_1)
    #print(f"长度为{a}")
#print(group_list)


# In[46]:


output_pearson = '/home/j304011/xz/type2_data_1/result/multi_not5_share5.csv'#此文件为final_model的性能评价输出文件夹
start_time = time.time()
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
    phenotype_excel_path=f'/home/j304011/xz/type2_data_1/simulation_data/not5_share5csv/y_not5_share5_{fff-1}.csv'
    trainY_id,valY_id,testY_id,trainY,valY,test_Y = load_phenotype(phenotype_excel_path)
    #labels = pd.DataFrame(dataset.train_labels, columns=tasks)
    trainY = pd.DataFrame(trainY, columns=tasks)
    valY = pd.DataFrame(valY, columns=tasks)
    test_Y = pd.DataFrame(test_Y, columns=tasks)
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
        
        file_path =f'/home/j304011/xz/type2_data_1/txt_file/gene_{a}.txt'
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
    x1 = Dropout(0.2)(x)
    x1 = Dense(32,activation='sigmoid')(x1)
    line = group_list[fff-1]
# 去除列表中每个元素的空格，此处实现了任务分组的自动识别和对应模型框架的搭建，但现在仅实现了一个分支点的自动化，当需要搭建多个分支点的网络时，因为
# 要得到靠近输出层位置的最佳信息后才能继续上一个分支点最优的选择，最终情况较为复杂。因此需要一些手动调整，详见除Single_task&Affinity.py外的其余几个文件内容
    result_list = [[task.strip() for task in sublist] for sublist in line]
    print(result_list)
    flattened_list = [item for sublist in result_list for item in sublist]
    if len(result_list) == 2:
        trainY_new = trainY[result_list[0] + result_list[1]].to_numpy()
        valY_new = valY[result_list[0] + result_list[1]].to_numpy()
        test_Y_new = test_Y[result_list[0] + result_list[1]].to_numpy()
    elif len(result_list) == 3:
        trainY_new = trainY[result_list[0] + result_list[1]+ result_list[2]].to_numpy()
        valY_new = valY[result_list[0] + result_list[1]+ result_list[2]].to_numpy()
        test_Y_new = test_Y[result_list[0] + result_list[1]+ result_list[2]].to_numpy()
    elif len(result_list) == 4:
        trainY_new = trainY[result_list[0] + result_list[1]+ result_list[2]+ result_list[3]].to_numpy()
        valY_new = valY[result_list[0] + result_list[1]+ result_list[2]+ result_list[3]].to_numpy()
        test_Y_new = test_Y[result_list[0] + result_list[1]+ result_list[2]+ result_list[3]].to_numpy()
    elif len(result_list) == 5:
        trainY_new = trainY[result_list[0] + result_list[1]+ result_list[2]+ result_list[3]+ result_list[4]].to_numpy()
        valY_new = valY[result_list[0] + result_list[1]+ result_list[2]+ result_list[3]+ result_list[4]].to_numpy()
        test_Y_new = test_Y[result_list[0] + result_list[1]+ result_list[2]+ result_list[3]+ result_list[4]].to_numpy()
    elif len(result_list) == 6:
        trainY_new = trainY[result_list[0] + result_list[1]+ result_list[2]+ result_list[3]+ result_list[4]+ result_list[5]].to_numpy()
        valY_new = valY[result_list[0] + result_list[1]+ result_list[2]+ result_list[3]+ result_list[4]+ result_list[5]].to_numpy()
        test_Y_new = test_Y[result_list[0] + result_list[1]+ result_list[2]+ result_list[3]+ result_list[4]+ result_list[5]].to_numpy()
    elif len(result_list) == 7:
        trainY_new = trainY[result_list[0] + result_list[1]+ result_list[2]+ result_list[3]+ result_list[4]+ result_list[5]+ result_list[6]].to_numpy()
        valY_new = valY[result_list[0] + result_list[1]+ result_list[2]+ result_list[3]+ result_list[4]+ result_list[5]+ result_list[6]].to_numpy()
        test_Y_new = test_Y[result_list[0] + result_list[1]+ result_list[2]+ result_list[3]+ result_list[4]+ result_list[5]+ result_list[6]].to_numpy()
    #print(trainY_new.shape)
    #print(type(trainY_new))
    x_layers = []
    for i in range(len(result_list)):
        x_layers.append(Dense(32, activation='sigmoid', name=f'x2_{i+1}')(x1))

    outputs_list = []
    index = 1
    
    for i, feature_group in enumerate(result_list):
        group_size = len(feature_group)
        for _ in range(group_size):
            output = Dense(1, name=f'output_{index}')(x_layers[i])  # 使用对应的 x2 层
            outputs_list.append(output)
            index += 1
            
    concatenated_outputs = concatenate(outputs_list, name='concatenated_outputs')
    print(concatenated_outputs.shape)
    new_model = Model(inputs=inputs, outputs=concatenated_outputs)
    new_model.compile(optimizer='adam', loss='mean_squared_error')
    checkpointer = ModelCheckpoint(filepath='best_model1.hdf5',
                            monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce = ReduceLROnPlateau(monitor='val_loss', patience=7,
                            verbose=1,
                            factor=0.5,
                            min_lr=1e-6)
    history = new_model.fit(concatenate_trainX,trainY_new,
                        steps_per_epoch=len(trainY_new) // batch_size,
                        validation_data=(concatenate_valX,valY_new),
                        validation_steps=len(valY_new) / batch_size,
                        batch_size=batch_size,
                        callbacks=[checkpointer, reduce],
                        epochs=epochs,verbose=0)
    new_model.save('multi_task'+str(fff)+ '.h5')    #task+str(fff)+ '.h5'
    #shutil.move('multi_task'+str(fff)+ '.h5', target_folder)
    preds = new_model.predict(concatenate_testX)
    test_Y = test_Y_new
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
    with open(output_pearson, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(flattened_list)
        writer.writerow(correlation_coefficients_pearsonr)
        
    #with open(output_mse, mode='a', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerow(flattened_list)
    #    writer.writerow(correlation_coefficients_mse)

    tf.keras.backend.clear_session()
    print(fff)
end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time  # 计算时间差，即代码执行时间
print(f"共计用时{elapsed_time}")

# In[ ]:




