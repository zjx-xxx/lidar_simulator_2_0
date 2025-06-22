import os
import re
import pandas as pd
import shutil
import datetime

# 读取单列362行数据，转置成1行362列，同时清洗最后一列标签
def clean_and_transpose_single_column(df):
    # 转置，变成1行多列
    row = df.T.values.flatten()

    # 最后一列数据取首个数字，若无数字则置0
    last_val = str(row[-1])
    match = re.search(r'[0-9]', last_val)
    if match:
        last_label = int(match.group(0))
    else:
        last_label = 0
    row[-1] = last_label

    # 其他列尝试转换为数字，不是数字的转为NaN
    cleaned_row = []
    for i, val in enumerate(row):
        if i == len(row) - 1:  # 最后一列已经处理
            cleaned_row.append(row[-1])
        else:
            try:
                # 保留数字，如果是字符串数字，转成int或float
                num = float(val)
                # 如果是整数形式，转int
                if num.is_integer():
                    num = int(num)
                cleaned_row.append(num)
            except:
                cleaned_row.append(float('nan'))
    return pd.DataFrame([cleaned_row])

# 数据文件夹路径
mydata_dir = os.path.join(os.getcwd(), 'labeled/classification')

# 获取文件列表
if os.path.exists(mydata_dir):
    file_names = os.listdir(mydata_dir)
else:
    print(f"目录 {mydata_dir} 不存在！")
    file_names = []

# 只匹配 dataN.csv 文件并排序
pattern = r'^data(\d+)\.csv$'
matching_files = []
for file in file_names:
    match = re.match(pattern, file)
    if match:
        idx = int(match.group(1))
        matching_files.append((idx, file))

matching_files.sort()
matching_files = [f for _, f in matching_files]

all_data = []

# 读取已有Data.csv（如果存在且非空）
try:
    if os.path.exists("./Data.csv") and os.path.getsize("./Data.csv") > 0:
        df_existing = pd.read_csv("./Data.csv", header=None)
        all_data.append(df_existing)
        print("已加载旧 Data.csv")
    else:
        print("未发现或未加载旧 Data.csv（文件不存在或为空）")
except pd.errors.EmptyDataError:
    print("Data.csv 文件存在但为空，跳过加载")

# 处理每个文件
for file_name in matching_files:
    file_path = os.path.join(mydata_dir, file_name)
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, header=None)
            if df.shape[1] != 1 or df.shape[0] != 362:
                print(f"{file_name} 不是单列362行数据，跳过")
                continue
            df_cleaned = clean_and_transpose_single_column(df)
            all_data.append(df_cleaned)
            print(f"{file_name} 已处理并添加")
        except Exception as e:
            print(f"{file_name} 处理出错: {e}")
    else:
        print(f"{file_name} 不存在")

# 合并所有数据并保存
if all_data:
    merged_data = pd.concat(all_data, axis=0, ignore_index=True)
    merged_data.to_csv('./Data.csv', header=False, index=False)
else:
    print("没有数据合并，未生成 Data.csv")
    merged_data = pd.DataFrame()

# 如果有数据，划分训练测试集并保存
if not merged_data.empty:
    Test = merged_data.sample(frac=0.1, random_state=42)
    Train = merged_data.drop(Test.index)

    Y_type_train = Train.iloc[:, -1:]
    Y_dir_train = Train.iloc[:, -2:-1]
    X_train = Train.iloc[:, :-2]

    Y_type_test = Test.iloc[:, -1:]
    Y_dir_test = Test.iloc[:, -2:-1]
    X_test = Test.iloc[:, :-2]

    os.makedirs('./type', exist_ok=True)
    os.makedirs('./direction', exist_ok=True)

    X_train.to_csv('./X_train.csv', header=False, index=False)
    Y_type_train.to_csv('./type/Y_train.csv', header=False, index=False)
    Y_dir_train.to_csv('./direction/Y_train.csv', header=False, index=False)

    X_test.to_csv('./X_test.csv', header=False, index=False)
    Y_type_test.to_csv('./type/Y_test.csv', header=False, index=False)
    Y_dir_test.to_csv('./direction/Y_test.csv', header=False, index=False)
    print("训练集和测试集已保存")
else:
    print("无数据，跳过训练测试集划分和保存")

# 归档原始 labeled 文件
now = datetime.datetime.now()
suffix = now.strftime("%m_%d_%H_%M")
archive_folder = f'./labeledData{suffix}'
os.makedirs(archive_folder, exist_ok=True)

for file_name in matching_files:
    source_path = os.path.join(mydata_dir, file_name)
    if os.path.exists(source_path):
        shutil.move(source_path, archive_folder)
        print(f"{file_name} 已移动到 {archive_folder}")
    else:
        print(f"{file_name} 未找到")
