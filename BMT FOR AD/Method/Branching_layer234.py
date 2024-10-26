#!/usr/bin/env python
# coding: utf-8

# In[23]:


#层2到层4的处理思想类似，但都与上一层输出的分组结果有关，故取第二层加以说明
def generate_task_combinations(total_tasks, num_groups, combination, combinations):
    if num_groups == 0:
        if sum(combination) == total_tasks:
            combinations.add(tuple(sorted(combination)))
        return    
    for i in range(1, total_tasks - num_groups + 2):
        new_combination = combination + [i]
        generate_task_combinations(total_tasks, num_groups - 1, new_combination, combinations)

# 测试示例
total_tasks = 4#在此处加上一层生成的分组中，每个组当成一个元素，上层的结果中得到4组，故此处值为4
num_groups_size = list(range(1,total_tasks+1))
print(num_groups_size)
combinations = set()
for num_groups in num_groups_size:
    generate_task_combinations(total_tasks, num_groups, [], combinations)
# 输出结果
    groups_list = [list(combination) for combination in combinations]
print(groups_list)
print(len(groups_list))


# In[24]:


#与层1同理，列出所有可能的分组情况
import copy
tasks =['A', 'B', 'C', 'D']#用大写字母代替4个分组

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
            #sublist_tuple = tuple(sublist)
            if sublist_tuple not in seen:
                seen.add(sublist_tuple)
                result.append([list(inner_list) for inner_list in sublist])
        return result
# 创建一个空列表用于存储结果
results = []
# 测试示例
for groups in groups_list:
    current_partition = [[] for _ in range(len(groups))]
# 调用函数进行分组，并将结果保存到结果列表中
    partition_tasks(tasks, groups, current_partition, results)    
    unique_list = remove_duplicates(results)
print(len(unique_list))
# 打印去重后的列表
for sublist in unique_list:
    print(sublist)


# In[25]:





# In[26]:


#此处将大写字母代替的分组与其内部包含的任务进行对应，并用任务填充所有可能的分组情况。
A = ['a']
B = ['b', 'c', 'e', 'g', 'h', 'i']
C = ['d']
D = ['f']
unique_list_1 = []#此处请在训练不同层的时候重新为列表命名，防止交叉。也有可能不用，请注意或思考
for i in range(len(unique_list)):
    original_list = unique_list[i]
# 定义字母和变量的映射关系字典
    letter_to_variable = {
        'A': A,
        'B': B,
        'C': C,
        'D': D
    }
# 新的填充后的列表
    filled_list = []
# 遍历给定的列表，根据字母查找对应的变量，并填充新列表
    for sublist in original_list:
        filled_sublist = []
        for letter in sublist:
            filled_sublist.extend(letter_to_variable[letter])
        filled_list.append(filled_sublist)

    print(filled_list)
    unique_list_1.append(filled_list)


# In[27]:


#进行任务与字母的对应
task_to_index = {}
for i, group in enumerate(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h','i']):
    for task in group:
        task_to_index[task] = i
print(task_to_index)


# In[29]:


#与层1同理得到最优的分组
import numpy as np 
data = np.load('face_spearman.npy')
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
                score = max(corr_score)
                score_sum+= score
        group_scores = score_sum / count
        group_scores_list.append(group_scores)
    print(min(group_scores_list))
    min_index = group_scores_list.index(min(group_scores_list))
    print(task_group_list[min_index])
    # 输出对应的task_group
    return task_group_list[min_index]
task_group_list = unique_list_1
affinity_scores = data[:,:,2]
max_scores_1 = max_scores(task_group_list,affinity_scores)
#得到的结果用于第三层的输入，原理与整个第二层相似，多个地方需要参考上一次分组得到的分组组数和分组情况。
#如果得到的结果两组则不继续聚合，因为我们的目的是选择组与组间最不相似的情况，也就是希望组间亲和度较小的任务尽量不聚到一组。


# In[30]:





# In[ ]:




