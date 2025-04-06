import os
import re
import pandas as pd
import shutil
import datetime

# 文件夹路径
mydata_dir = os.path.join(os.getcwd(), 'labeled')

# 获取文件列表
if os.path.exists(mydata_dir):
    file_names = os.listdir(mydata_dir)
    print(file_names)
else:
    print(f"目录 {mydata_dir} 不存在！")
    file_names = []

# 匹配 dataN.csv 文件，并按数字 N 排序
pattern = r'^data(\d+)\.csv$'
matching_files = []
for file in file_names:
    match = re.match(pattern, file)
    if match:
        index = int(match.group(1))
        matching_files.append((index, file))

matching_files.sort()
matching_files = [file for _, file in matching_files]

# 清洗函数：提取第一个一位非负整数（0~9）
def clean_cell(cell):
    match = re.search(r'[0-9]', str(cell))  # 匹配第一个一位数字字符
    return match.group(0) if match else None

# 合并所有 CSV 文件的数据
all_data = []

# 如果已有 Data.csv，则先读取
if os.path.exists("./Data.csv") and os.path.getsize("./Data.csv") > 0:
    df_existing = pd.read_csv("./Data.csv", header=None)
    all_data.append(df_existing)
else:
    print("未发现或未加载旧 Data.csv")

# 清洗并读取每个文件
for file_name in matching_files:
    file_path = os.path.join(mydata_dir, file_name)
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, header=None)
            df_cleaned = df.applymap(clean_cell)  # 清洗数据
            df_cleaned = df_cleaned.apply(pd.to_numeric, errors='coerce')  # 转换为数值
            row = df_cleaned.values.flatten()
            all_data.append(pd.DataFrame([row]))
            print(f"{file_name} 已清洗并添加")
        except Exception as e:
            print(f"{file_name} 处理出错: {e}")
    else:
        print(f"{file_name} 不存在")

# 合并并保存总数据
merged_data = pd.concat(all_data, axis=0, ignore_index=True)
merged_data.to_csv('./Data.csv', header=False, index=False)

# 分割训练和测试数据（10%为测试）
Test = merged_data.sample(frac=0.1, random_state=42)
Train = merged_data.drop(Test.index)

# 拆分数据
Y_type_train = Train.iloc[:, -2:-1]
Y_dir_train = Train.iloc[:, -1:]
X_train = Train.iloc[:, :-2]

Y_type_test = Test.iloc[:, -2:-1]
Y_dir_test = Test.iloc[:, -1:]
X_test = Test.iloc[:, :-2]

# 保存数据集
X_train.to_csv('./X_train.csv', header=False, index=False)
Y_type_train.to_csv('./type/Y_train.csv', header=False, index=False)
Y_dir_train.to_csv('./direction/Y_train.csv', header=False, index=False)

X_test.to_csv('./X_test.csv', header=False, index=False)
Y_type_test.to_csv('./type/Y_test.csv', header=False, index=False)
Y_dir_test.to_csv('./direction/Y_test.csv', header=False, index=False)
print("训练集和测试集已保存")

# 创建时间戳文件夹并移动原始 labeled 文件
now = datetime.datetime.now()
suffix = now.strftime("%m_%d_%H_%M")
archive_folder = f'./rawData{suffix}'
os.makedirs(archive_folder, exist_ok=True)

for file_name in matching_files:
    source_path = os.path.join(mydata_dir, file_name)
    if os.path.exists(source_path):
        shutil.move(source_path, archive_folder)
        print(f"{file_name} 已移动到 {archive_folder}")
    else:
        print(f"{file_name} 未找到")
