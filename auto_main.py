from expansion import expand
from result_sorting import sort_result
from result_displaying import display_result
from sis_ana import sis
from coefficients_fitting import fit
from result_plotting import plot_result
import pandas as pd
import torch
import os
from timer import Timer

# 配置运行设备：优先使用GPU，无GPU则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取当前代码运行的工作目录
path = os.getcwd()
# path = "yourfilepath"

# 程序启动提示
print("-------------------程序开始运行---------------------")

# 读取输入数据与目标变量
data = pd.read_csv(os.path.join(path, "data.csv"))
focus = pd.read_csv(os.path.join(path, "focus.csv"))

# 对原始数据进行特征空间扩展，参数为5元扩展
data_expanded = expand(data,5)

# 确定性独立筛选，筛选出Top10高相关性特征
data_sis = sis(data_expanded, focus.to_numpy(), 10)
# data_sis = data_expanded

# 初始化存储超参数搜索结果的列表
results2=[]

# 遍历所有超参数组合，进行网格搜索
for num_epochs in [10,20]:
    for batch_size in [4,8,16]:
        for lr  in[1e-2,1e-3,1e-4]:
            # 使用当前超参数进行模型系数拟合
            r2, coef, loss = fit(data_sis, focus.to_numpy(), device,
                                        num_epochs=num_epochs,
                                        batch_size=batch_size,
                                        lr=lr)

            # 对拟合结果进行排序，获取最优特征
            results = sort_result(data_sis.to_numpy(), 
                                data_sis.columns.to_numpy(),
                                r2, coef, loss, 1)
            
            # 将超参数与对应最优结果存入列表
            results2.append([num_epochs,batch_size,lr,results[0]])
            # 打印当前参数组训练完成提示
            print(num_epochs,batch_size,"结束")
            
# 根据最优特征的R2值对结果列表进行降序排序
results2_sorted = sorted(results2, key=lambda x: x[3].get_r2(), reverse=True)

# 初始化日志字符串与表头
log = ''
str_show = [['num_epochs','batch_size','lr', 'r2', 'loss', 'score', 'model']]

# 遍历排序后的结果，格式化输出内容
for result in results2_sorted:
    str_show.append([result[0],result[1],result[2],
                     result[3].get_r2(), 
                     result[3].get_loss(), 
                     result[3].get_score(), 
                     result[3].get_full_name()])
    
# 按对齐格式拼接日志文本
for line in str_show:
    log += '\n{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{}\n'.format(*line)

# 将完整日志写入文件
log_path = os.path.join(path, 'log.txt')
with open(log_path, 'w', encoding='utf-8') as file:
    file.write(log)