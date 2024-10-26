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
import albumentations  #数据增强
import pandas as pd
from collections import namedtuple
from tensorflow.keras.initializers import glorot_uniform
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense,Dropout
from tensorflow.keras.models import Model


# In[2]:


#超参数
tasks = ['5_o_Clock_Shadow', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Goatee', 'Mustache', 'No_Beard', 'Rosy_Cheeks', 'Wearing_Hat']
target_size = (218,178)
epochs = 15
batch_size = 8
INIT_LR=0.01
optimizer = Adam(learning_rate=INIT_LR)
input_shape = (218,178,3)


# In[3]:


from collections import namedtuple
#训练集
def load_dataset(target_size):
    train_image_dir = 'C:/Users/ASUS/Desktop/data/mydata2/train'
    train_attr_path = 'C:/Users/ASUS/Desktop/data/mydata2/a.csv'
    train_df = pd.read_csv(train_attr_path)

    train_images = []
    for filename in os.listdir(train_image_dir):
        img_path = os.path.join(train_image_dir, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size)
        img_arr = tf.keras.preprocessing.image.img_to_array(img)
        train_images.append(img_arr)

    train_images = tf.stack(train_images)
    train_images = tf.image.convert_image_dtype(train_images, tf.float32)

    train_labels = train_df.iloc[:, 1:].astype('int32').to_numpy()

    tasks = train_df.columns[1:]  # 获取任务列表

    train_img_list = []
    train_labels_list = []

    for i in range(len(train_labels)):
        labels = train_labels[i]
        img = train_images[i]

        train_labels_list.append(labels)
        train_img_list.append(img)

    train_labels_array = np.array(train_labels_list)
    train_img_array = np.array(train_img_list)

    Dataset = namedtuple('Dataset', ['train_img', 'train_labels'])
    return Dataset(train_img_array, train_labels_array)
dataset = load_dataset(target_size=(218,178))  #请务必运行此句


# In[4]:


#定义搭建模型的层
def conv_bn_relu(x, filters, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def identity_block(x, filters):
    shortcut = x
    x = conv_bn_relu(x, filters=filters, kernel_size=(1, 1))
    x = conv_bn_relu(x, filters=filters, kernel_size=(3, 3))
    x = Conv2D(filters=4 * filters, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv_block(x, filters, strides=(2, 2)):
    shortcut = x
    x = conv_bn_relu(x, filters=filters, kernel_size=(1, 1), strides=strides)
    x = conv_bn_relu(x, filters=filters, kernel_size=(3, 3))
    x = Conv2D(filters=4 * filters, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    shortcut = Conv2D(filters=4 * filters, kernel_size=(1, 1), strides=strides)(shortcut)
    shortcut = BatchNormalization()(shortcut)
    x = tf.keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


# In[13]:


#请注意，此网络是在基础网络上添加了我们之前得到的各个分支点间的分组情况得到的，所以此网络是
#唯一的，当在之前的代码中得到的分支情况改变时，需要重新编写此模型分支点处的代码
def final_resnet50(input_shape):
    #第一个block处不做分支点，故省略第一个，直接从第二个开始
    def block_1(input_shape):
        inputs = Input(shape=input_shape)
        x = conv_bn_relu(inputs, filters=64, kernel_size=(7, 7), strides=(2, 2))
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = conv_block(x, filters=64, strides=(1, 1))
        x = identity_block(x, filters=64)
        out1 = identity_block(x, filters=64)
        return out1,inputs
    out1,inputs = block_1(input_shape=(218,178,3))
    out1 = out1
    inputs = inputs
    # 第一次分组任务
    group11_tasks = ['task1', 'task2', 'task3','task4', 'task5', 'task7', 'task8','task9']
    group12_tasks = [ 'task6']
    group11_branch = out1
    for _ in range(2):  # 共享层的数量可以调整
        group11_branch = Dense(units=64, activation='relu')(group11_branch)
    group11_output = Dense(units=len(group11_tasks), activation=None, name='group11')(group11_branch)
    group12_branch = out1
    for _ in range(2):  # 共享层的数量可以调整
        group12_branch = Dense(units=64, activation='relu')(group12_branch)
    group12_output = Dense(units=len(group12_tasks), activation=None, name='group12')(group12_branch)
    
    def block_2(input):
        #input = Input(input_shape=(28,28,1))
        x = conv_block(input, filters=128)
        x = identity_block(x, filters=128)
        x = identity_block(x, filters=128)
        out2 = identity_block(x, filters=128)
        return out2  
    # 第二次分组任务
    out2_1 = block_2(input=group11_output)
    out2_2 = block_2(input=group12_output)
    group21_tasks = ['task1', 'task2', 'task3','task4', 'task5', 'task7', 'task8','task9']
    group22_tasks = [ 'task6']
    group21_branch = out2_1
    for _ in range(2):  # 共享层的数量可以调整
        group21_branch = Dense(units=64, activation='relu')(group21_branch)
    group21_output = Dense(units=len(group21_tasks), activation=None, name='group21')(group21_branch)
    group22_branch = out2_2
    for _ in range(2):  # 共享层的数量可以调整
        group22_branch = Dense(units=64, activation='relu')(group22_branch)
    group22_output = Dense(units=len(group22_tasks), activation=None, name='group22')(group22_branch)
    
    def block_3(input):    
        x = conv_block(input, filters=256)
        x = identity_block(x, filters=256)
        x = identity_block(x, filters=256)
        x = identity_block(x, filters=256)
        x = identity_block(x, filters=256)
        out3 = identity_block(x, filters=256)
        return out3
    out3_1 = block_3(input=group21_output)
    out3_2 = block_3(input=group22_output)
    # 第三次分组任务
    group31_tasks = ['task1', 'task2', 'task3','task4', 'task5', 'task7', 'task8','task9']
    group32_tasks = [ 'task6']
    group31_branch = out3_1
    for _ in range(2):  # 共享层的数量可以调整
        group31_branch = Dense(units=64, activation='relu')(group31_branch)
    group31_output = Dense(units=len(group31_tasks), activation=None, name='group31')(group31_branch)
    group32_branch = out3_2
    for _ in range(2):  # 共享层的数量可以调整
        group32_branch = Dense(units=64, activation='relu')(group32_branch)
    group32_output = Dense(units=len(group32_tasks), activation=None, name='group32')(group32_branch)
    
    def block_4(input):    
        x = conv_block(input, filters=512)
        x = identity_block(x, filters=512)
        out4 = identity_block(x, filters=512)
        return out4
    out4_1 = block_4(input=group31_output)
    out4_2 = block_4(input=group32_output)

    # 第四次分组任务
    group41_tasks = ['task1']
    group42_tasks = ['task2','task3', 'task5', 'task7','task8','task9']
    group43_tasks = ['task4']
    group44_tasks = [ 'task6']
 
    group41_branch = out4_1
    for _ in range(2):  # 共享层的数量可以调整
        group41_branch = Dense(units=64, activation='relu')(group41_branch)
    group41_output = Dense(units=len(group41_tasks), activation=None, name='group41')(group41_branch)

    group42_branch = out4_1
    for _ in range(2):  # 共享层的数量可以调整
        group42_branch = Dense(units=64, activation='relu')(group42_branch)
    group42_output = Dense(units=len(group42_tasks), activation=None, name='group42')(group42_branch)

    group43_branch = out4_1
    for _ in range(2):  # 共享层的数量可以调整
        group43_branch = Dense(units=64, activation='relu')(group43_branch)
    group43_output = Dense(units=len(group43_tasks), activation=None, name='group43')(group43_branch)

    group44_branch = out4_2
    for _ in range(2):  # 共享层的数量可以调整
        group44_branch = Dense(units=64, activation='relu')(group44_branch)
    group44_output = Dense(units=len(group44_tasks), activation=None, name='group44')(group44_branch)

    def block_5(input):    
        x = GlobalAveragePooling2D()(input)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.5)(x)   # 在Flatten层后面添加Dropout层
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions = Dense(2, activation='softmax')(x)
        return predictions
    task1_pre = block_5(group41_output)
    task2_pre = block_5(group42_output)
    task3_pre = block_5(group42_output)
    task4_pre = block_5(group43_output)
    task5_pre = block_5(group42_output)
    task6_pre = block_5(group44_output)
    task7_pre = block_5(group42_output)
    task8_pre = block_5(group42_output)
    task9_pre = block_5(group42_output)
    model = Model(inputs=inputs, outputs=[task1_pre,task2_pre,task3_pre,task4_pre,task5_pre,task6_pre,task7_pre,task8_pre,task9_pre], name='final_resnet50')
    return model
model = final_resnet50(input_shape=input_shape)


# In[10]:


def generator(file_pathList, labels, batch_size, train_action=False):
    L = len(file_pathList)
    while True:
        input_labels = []
        input_samples = []
        for _ in range(batch_size):
            temp = np.random.randint(0, L)
            X = file_pathList[temp]
            Y = labels[temp]
            Y_categorical = [tf.keras.utils.to_categorical(y, 2) for y in Y]  # Convert labels to one-hot encoding
            input_samples.append(X)
            input_labels.append(Y_categorical)  # Append list of one-hot encoded labels for all tasks

        batch_x = np.asarray(input_samples)
        batch_y = [np.asarray(task_labels) for task_labels in zip(*input_labels)]  # Convert list of task labels to a list of arrays

        yield (batch_x, batch_y)


# In[15]:


import time

checkpointer = ModelCheckpoint(filepath='best_model.hdf5',
                        monitor='val_loss', verbose=1, save_best_only=True, mode='min')#此处得到的val_acc是每个任务的，
                                                                                        #故选择val_loss，可根据需求调整
#该回调函数 ModelCheckpoint 将在每个epoch后保存模型到filepath
reduce = ReduceLROnPlateau(monitor='val_loss', patience=10,
                    verbose=1,
                    factor=0.5,
                    min_lr=1e-6)
train_img_X = [dataset.train_img[i] for i in range(len(dataset.train_img))]
train_labels_X = [dataset.train_labels[i] for i in range(len(dataset.train_img))]
trainX, valX, trainY, valY = train_test_split(train_img_X, train_labels_X, test_size=0.3, random_state=42)
model.compile(optimizer='adam',  loss='categorical_crossentropy', metrics=['accuracy'])
start_time = time.time()
history = model.fit(
    generator(trainX, trainY, batch_size, train_action=True),
    steps_per_epoch=len(trainX) // batch_size,  # Use integer division
    validation_data=generator(valX, valY, batch_size, train_action=False),
    validation_steps=len(valX) // batch_size,  # Use integer division
    batch_size=batch_size,
    callbacks=[checkpointer, reduce],
    epochs=epochs,
    verbose=1
)
model.save('face_final_model')
end_time = time.time()
total_time = end_time - start_time
print(f'Total training time: {total_time}')


# In[38]:


#将上述运行过程中的loss，acc，val_loss,val_acc等内容提取出来以字典形式储存
task_losses = {}  # 用于存储每个任务的损失值
task_val_losses = {}
task_accuracies = {}  # 用于存储每个任务的准确率
task_val_accuracies = {}
# 从输出中提取任务的损失值和准确率
for key in history.history:
    if key.startswith('dense_')and key.endswith('_loss'):
        task_name = key.split('_loss')[0]  # 提取任务名
        modified_task_name =  task_name + '_loss'
        task_losses[modified_task_name] = history.history[key]
    elif key.startswith('dense_')and key.endswith('_accuracy'):
        task_name = key.split('_accuracy')[0]  # 提取任务名
        modified_task_name =  task_name + '_acc'
        task_accuracies[modified_task_name] = history.history[key]
    elif key.startswith('val_dense')and key.endswith('_loss'):
        task_name = key.split('_loss')[0]  # 提取任务名
        modified_task_name =  task_name + '_loss'
        task_val_losses[modified_task_name] = history.history[key]
    elif key.startswith('val_dense')and key.endswith('_accuracy'):
        task_name = key.split('_accuracy')[0]  # 提取任务名
        modified_task_name =  task_name + '_acc'
        task_val_accuracies[modified_task_name] = history.history[key]
        
print(task_losses)
print(task_accuracies)
print(task_val_losses)
print(task_val_accuracies)


# In[40]:


#以下一段代码用来处理需要作图的数据，此处选择了val_acc的值随epochs的变化而变化的图，可调整数据来源做其他的图
#将层的名字用任务代替，方便后续作图
data =task_val_accuracies.copy()#此处为数据来源，更换可以得到其他的图
new_names =  ['5_o_Clock_Shadow', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Goatee', 'Mustache', 'No_Beard', 'Rosy_Cheeks', 'Wearing_Hat']
new_data = {}
for i, (old_key, values) in enumerate(data.items()):
    new_key = new_names[i]  # 使用新名称列表中的名称
    new_data[new_key] = values
print(new_data)
#根据得到的每个任务的
data = new_data.copy()
sample_count = len(data[new_names[0]])
sum_by_position = {i: 0 for i in range(sample_count)}
for task_key, task_values in data.items():
    for i, value in enumerate(task_values):
        sum_by_position[i] += value
average_by_position = {i: total_value / len(data) for i, total_value in sum_by_position.items()}
data['average_acc'] = list(average_by_position.values())
#print(data)
#开始导入函数进行作图
import matplotlib.pyplot as plt# 绘制损失值随epochs的变化曲线
plt.figure(figsize=(20, 10))#横纵坐标格数
for task_name, loss_values in new_data.items():
    plt.plot([i+1 for i in range(epochs)],loss_values, label=task_name)
    #i+1是因为在range()函数取值时会从0开始而不取最大值，为了与实际进行对应，epochs是从1开始，所以加1
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.title('Evaluation ACC after grouping')
plt.plot([i+1 for i in range(epochs)], data['average_acc'], label='average_acc', color='red', linestyle='--', linewidth=2)
#添加了一条平均值的线，并设置了相应的特征
plt.legend(loc='lower right')
plt.ylim(0, 1)
plt.grid(True)
last_epoch = 10
last_epoch_avg_acc = data['average_acc'][last_epoch-1]#加1是因为上面减1以后数据又对不上了，调整过后就符合了
plt.annotate(f'MAX (Avg ACC): {last_epoch_avg_acc:.2f}', 
             xy=(last_epoch, last_epoch_avg_acc), 
             xytext=(-75, -80), 
             textcoords='offset points', 
             color='red', 
             fontsize=14,
             arrowprops=dict(facecolor='red', arrowstyle='wedge,tail_width=0.7', alpha=0.5))
plt.show()


# In[42]:





# In[66]:





# In[19]:


#测试test集准确率
#加载测试集数据
from collections import namedtuple
def load_dataset_test(target_size):
    train_image_dir = 'C:/Users/ASUS/Desktop/data/mydata/test'
    train_attr_path = 'C:/Users/ASUS/Desktop/data/mydata/list_attr_test_1300_1600.csv'
    train_df = pd.read_csv(train_attr_path)
    train_images = []
    for filename in os.listdir(train_image_dir):
        img_path = os.path.join(train_image_dir, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size)
        img_arr = tf.keras.preprocessing.image.img_to_array(img)
        train_images.append(img_arr)
    train_images = tf.stack(train_images)
    train_images = tf.image.convert_image_dtype(train_images, tf.float32)
    train_labels = train_df.iloc[:, 1:].astype('int32').to_numpy()
    tasks = train_df.columns[1:]  # 获取任务列表
    train_img_list = []
    train_labels_list = []
    
    for i in range(len(train_labels)):
        labels = train_labels[i]
        img = train_images[i]
        train_labels_list.append(labels)
        train_img_list.append(img)
    train_labels_array = np.array(train_labels_list)
    train_img_array = np.array(train_img_list)
    Dataset = namedtuple('Dataset', ['test_img', 'test_labels'])
    return Dataset(train_img_array, train_labels_array)
dataset_test = load_dataset_test(target_size=(218,178))  #请务必运行此句


# In[20]:


# 加载之前训练好的模型
from keras.models import load_model
import numpy as np
model = load_model('face_final_model')


# In[30]:


# 准备测试数据集
test_data = dataset_test.test_img
test_labels = dataset_test.test_labels
Y_categorical = [tf.keras.utils.to_categorical(y, 2) for y in test_labels]  # num_classes为你的分类数目
batch_y = [np.asarray(task_labels) for task_labels in zip(*Y_categorical)]
batch_y = np.array(batch_y)
print(batch_y.shape)
# 进行预测
predictions = model.predict(test_data)
import numpy as np
threshold = 0.5  # 设置阈值，根据具体情况调整
rounded_predictions = []
for example_predictions in predictions:
    rounded_example_predictions = []
    for task_predictions in example_predictions:
        rounded_task_predictions = [1 if p >= threshold else 0 for p in task_predictions]
        rounded_example_predictions.append(rounded_task_predictions)
    rounded_predictions.append(rounded_example_predictions)

rounded_predictions = np.array(rounded_predictions)
print(rounded_predictions.shape)

import numpy as np
accuracies = []
for task in range(len(tasks)):
    correct_predictions = 0
    total_samples = len(rounded_predictions[task])
    for sample_idx in range(total_samples):
        prediction = np.argmax(rounded_predictions[task][sample_idx])
        label = np.argmax(batch_y[task][sample_idx])
        if prediction == label:
            correct_predictions += 1
    accuracy = correct_predictions / total_samples
    accuracies.append(accuracy)
for task, accuracy in enumerate(accuracies):
    print(f"Task {task+1} 准确率: {accuracy:.2f}")


# In[31]:





# In[ ]:




