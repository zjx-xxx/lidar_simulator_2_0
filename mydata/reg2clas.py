import os
import pandas as pd
import re

# 定义路径
input_dir = "./labeled/regression"
output_dir = "./labeled/classification"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录下所有 CSV 文件
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            # 读取 CSV 文件
            df = pd.read_csv(input_path, header=None)

            # 只保留前 361 行
            df = df.iloc[:361]

            # 清理非数字字符，只保留数字（可包含小数点和负号）
            df[0] = df[0].astype(str).apply(lambda x: re.sub(r"[^\d\.\-]", "", x))

            # 再转回数值型（如果某个值为空则会变为 NaN）
            df[0] = pd.to_numeric(df[0], errors='coerce')

            # 删除空值行
            df = df.dropna()

            # 保存到目标目录
            df.to_csv(output_path, index=False, header=False)
            print(f"已处理并保存: {filename}")
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
