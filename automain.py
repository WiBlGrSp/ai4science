from expansion import expand
from result_sorting import sort_result
from result_displaying import display_result
from sis_ana import sis
from coefficients_fitting import fit
from coefficients_fitting import autofit
from result_plotting import plot_result
import pandas as pd
import torch
import os
from timer import Timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 代码运行目录
path = os.getcwd()
# path = "yourfilepath"

print("-------------------程序开始运行---------------------")

# 读取数据
data = pd.read_csv(os.path.join(path, "data.csv"))
focus = pd.read_csv(os.path.join(path, "focus.csv"))

# 初始输入的扩充
data_expanded = expand(data)

# 确定性独立筛选
# data_sis = sis(data_expanded, focus.to_numpy(), 10)
data_sis = data_expanded

#遍历可能的超参数组合
results2=[]
for num_epochs in [10,30]:
    for batch_size in [8,16]:
        for lr  in[0.1]:
            #系数拟合
            r2, coef, loss = autofit(data_sis, focus.to_numpy(), device,
                                        num_epochs=num_epochs,
                                        batch_size=batch_size,
                                        lr=lr,use_net=True)

            # 结果整理
            results = sort_result(data_sis.to_numpy(), 
                                data_sis.columns.to_numpy(),
                                r2, coef, loss, 1)
            results2.append([num_epochs,batch_size,lr,results[0]])
            print(num_epochs,batch_size,"结束")
            
#对results2按R2降序排列
results2_sorted = sorted(results2, key=lambda x: x[3].get_r2(), reverse=True)

#输出日志
log = ''
str_show = [['num_epochs','batch_size','lr', 'r2', 'loss', 'score', 'model']]
for result in results2_sorted:
    str_show.append([result[0],result[1],result[2],
                     result[3].get_r2(), 
                     result[3].get_loss(), 
                     result[3].get_score(), 
                     result[3].get_full_name()])
    
for line in str_show:
    log += '\n{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{}\n'.format(*line)
# 输出log
log_path = os.path.join(path, 'log')
with open(log_path, 'a', encoding='utf-8') as file:
    file.write(log)
    
