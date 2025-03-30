import os
import re
import pandas as pd

# 获取当前目录下的 mydata 子目录的路径
mydata_dir = os.path.join(os.getcwd(), 'mydata')
# 获取 mydata 子目录中的所有文件
if os.path.exists(mydata_dir):
    file_names = os.listdir(mydata_dir)  # 获取 mydata 子目录中的所有文件名
    print(file_names)  # 打印文件名列表
else:
    print(f"目录 {mydata_dir} 不存在！")

# 使用正则表达式匹配所有符合 pattern 的文件名
pattern = r'^data\d+\.csv$'  # 匹配 data 后跟一个或多个数字，再加 .csv 扩展名
matching_files = [file for file in file_names if re.match(pattern, file)]


# 存储所有读取的数据
all_data = []
file_path = "./Data.csv"

if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    df = pd.read_csv(file_path, header=None)
else:
    df = pd.DataFrame()  # 创建空DataFrame

all_data.append(df)
# 读取并合并所有的 CSV 文件
for file_name in matching_files:
    if os.path.exists(f'./labeled/{file_name}'):
        df = pd.read_csv(f'./labeled/{file_name}', header=None)  # 读取每个 CSV 文件
        row = df.values.flatten()  # 将列转为一维数组（行）
        all_data.append(pd.DataFrame([row]))
        print(f'{file_name}已添加')
    else:
        print(f"文件 {file_name} 不存在！")

# 合并数据（按行合并）
merged_data = pd.concat(all_data, axis=0, ignore_index=True)

merged_data.to_csv('./Data.csv', header=False, index=False)
Test = merged_data.sample(frac=0.1, random_state=42)
Train = merged_data.drop(Test.index)

# 提取最后一列数据作为 Y_train
Y_train = Train.iloc[:, -2:]
# 提取前面所有的列作为 X_train
X_train = Train.iloc[:, :-2]
Y_test = Test.iloc[:, -2:]
X_test = Test.iloc[:, :-2]


# 保存 X_train、Y_train、X_test 和 Y_test 到新的 CSV 文件
X_train.to_csv('./X_train.csv', header=False, index=False)
Y_train.to_csv('./Y_train.csv', header=False, index=False)
X_test.to_csv('./X_test.csv', header=False, index=False)
Y_test.to_csv('./Y_test.csv', header=False, index=False)
print("X_train、Y_train、X_test 和 Y_test 已保存。")

import datetime

# 获取当前时间
now = datetime.datetime.now()

# 提取月、日、时、分
month = now.strftime("%m")  # 月
day = now.strftime("%d")  # 日
hour = now.strftime("%H")  # 时
minute = now.strftime("%M")  # 分

# 构建文件名后缀
file_suffix = f"{month}_{day}_{hour}_{minute}"

# 假设你要保存的文件名
filename = f"rawData{file_suffix}"
# 文件夹路径
folder_path = f'./{filename}'

# 创建文件夹（如果文件夹已存在，os.makedirs() 不会报错）
os.makedirs(folder_path, exist_ok=True)  # exist_ok=True 表示如果文件夹已存在不报错
print(f"文件夹 {folder_path} 已创建")

import shutil
for file_name in matching_files:
    # 源文件路径
    source_file = f'./labeled/{file_name}'
    if os.path.exists(source_file):
        # 移动文件
        shutil.move(source_file, folder_path)
        print(f"文件 {source_file} 已移动到 {folder_path}")
    else:
        print(f"文件 {file_name} 不存在！")
