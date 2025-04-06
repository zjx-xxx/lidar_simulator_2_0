import os
import pandas as pd

def read_csv_as_series(file_path):
    # 读取为一列Series
    data = pd.read_csv(file_path, header=None).squeeze()
    return data

def main():
    folder = 'raw'
    seen = set()
    deleted_count = 0  # 删除计数器

    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder, filename)
            try:
                data = read_csv_as_series(file_path)

                if data.shape[0] != 360:
                    print(f"跳过 {filename}，不是360行")
                    continue

                data_tuple = tuple(data)

                if data_tuple in seen:
                    print(f"删除重复文件：{filename}")
                    os.remove(file_path)
                    deleted_count += 1
                else:
                    seen.add(data_tuple)
                    data.to_csv(file_path, index=False, header=False)
            except Exception as e:
                print(f"处理 {filename} 时出错：{e}")

    print(f"\n共删除了 {deleted_count} 个重复文件。")

if __name__ == '__main__':
    main()
