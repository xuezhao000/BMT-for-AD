#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
total_tasks = 7#此处为需要进行分组的任务总数
num_groups_size = list(range(1,total_tasks+1))
print(num_groups_size)#此行输出表示所有的元素可能聚为几组

combinations = set()
for num_groups in num_groups_size:
    generate_task_combinations(total_tasks, num_groups, [], combinations)
# 输出结果
    groups_list = [list(combination) for combination in combinations]
print(groups_list)#此行输出表示分好的组中有几个任务


# In[ ]:


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
tasks = ['a', 'b', 'c', 'd', 'e', 'f', 'g']#字母数量与任务总数相等
for groups in groups_list:
    current_partition = [[] for _ in range(len(groups))]

# 调用函数进行分组，并将结果保存到结果列表中
    partition_tasks(tasks, groups, current_partition, results)    
    unique_list = remove_duplicates(results)
print(len(unique_list))#共有多少种情况
# 打印去重后的列表
for sublist in unique_list:
    print(sublist) #列出所有情况 


# In[ ]:


#将任务与字母的顺序一一对应
task_to_index = {}
for i, group in enumerate(['a', 'b', 'c', 'd', 'e', 'f', 'g']):
    for task in group:
        task_to_index[task] = i
print(task_to_index)


# In[ ]:


#加载要使用的数据集，此处注意因为聚类操作是从底到顶的，所以第一次聚类选用的值是之前得到的最后一组值
import numpy as np 
data = np.load('face_spearman.npy')
print(data[:,:,3])#共有4个分支点，最后一层在列表的角标中取3


# In[ ]:


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
    print(min(group_scores_list))#计算组间最小相似的分组情况
    min_index = group_scores_list.index(min(group_scores_list))
    print(task_group_list[min_index])
    # 输出对应的task_group
    return task_group_list[min_index]
task_group_list = unique_list
affinity_scores = data[:,:,3]
max_scores_1 = max_scores(task_group_list,affinity_scores)  #输出计算的最小值和对应的分组情况

