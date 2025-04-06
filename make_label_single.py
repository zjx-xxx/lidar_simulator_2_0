import numpy as np
import tkinter as tk
import pandas as pd
import os
import re

def draw_lidar_from_csv(csv_files):
    index = 0  # 当前处理的文件索引
    total_files = len(csv_files)
    if total_files == 0:
        print("没有找到 CSV 文件。")
        return

    root = tk.Tk()
    root.title("Lidar Point Cloud Labeling")

    label_entry = tk.Entry(root)
    label_entry.pack()
    label_entry.focus_force()

    save_button = tk.Button(root, text="保存标签")
    save_button.pack()

    canvas_size = 500
    lidar_canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
    lidar_canvas.pack()

    center_x, center_y = canvas_size // 2, canvas_size // 2
    radius = 220
    lidar_canvas.create_oval(
        center_x - radius, center_y - radius, center_x + radius, center_y + radius, outline="black"
    )

    def update_canvas(data):
        lidar_canvas.delete("points")  # 只删除点云数据，避免整个画布刷新
        for angle in range(360):
            distance = data[angle]
            if distance > 0:
                radian = np.deg2rad(angle - 90)
                end_x = center_x + 1.1 * distance * np.cos(radian)
                end_y = center_y + 1.1 * distance * np.sin(radian)
                lidar_canvas.create_oval(
                    end_x - 1, end_y - 1, end_x + 1, end_y + 1, fill="red", outline="red", tags="points"
                )

    def load_next_file():
        nonlocal index
        if index >= total_files:
            print("所有文件标注完成。")
            root.destroy()
            return

        csv_file = csv_files[index]
        label_entry.delete(0, tk.END)
        label_entry.focus_force()
        try:
            data = pd.read_csv(f'./mydata/raw/{csv_file}', header=None).values.flatten()
            if len(data) != 360:
                raise ValueError("CSV 文件数据不完整，缺少激光雷达距离值。")
            update_canvas(data)
        except Exception as e:
            print(f"无法读取 {csv_file}: {e}")
            index += 1
            load_next_file()
            return

    def save_label(event=None):
        nonlocal index
        label_value = label_entry.get().strip()
        if not label_value:
            return  # 需要填写标签

        csv_file = csv_files[index]
        if label_value == "9":
            os.remove(f'./mydata/raw/{csv_file}')
            print(f"文件 {csv_file} 已删除，数据被舍弃。")
        else:
            try:
                data = pd.read_csv(f'./mydata/raw/{csv_file}', header=None).values.flatten()
                data_with_labels = np.append(data, [label_value])
                df = pd.DataFrame(data_with_labels)
                df.to_csv(f'./mydata/labeled/classification/{csv_file}', index=False, header=False)
                os.remove(f'./mydata/raw/{csv_file}')
                print(f"标签 {label_value} 已保存到 {csv_file}")
            except Exception as e:
                print(f"保存 {csv_file} 时出错: {e}")

        index += 1
        load_next_file()

    save_button.config(command=save_label)
    label_entry.bind("<Return>", save_label)  # 按 Enter 直接保存

    load_next_file()
    root.mainloop()

# 获取所有文件并开始循环
csv_files = []
raw_data_dir = os.path.join(os.getcwd(), 'mydata/raw')
labeled_data_dir = os.path.join(os.getcwd(), 'mydata/labeled/classification')

if os.path.exists(raw_data_dir):
    file_names = os.listdir(raw_data_dir)
    print(file_names)
else:
    print(f"目录 {raw_data_dir} 不存在！")
    file_names = []

pattern = r'^data(\d+)\.csv$'

# 提取匹配的文件名，并按数字部分排序
matching_files = []
for file in file_names:
    match = re.match(pattern, file)
    if match:
        file_index = int(match.group(1))
        matching_files.append((file_index, file))

# 按照数字 index 排序
matching_files.sort()

# 只取排序后的文件名
csv_files = [file for _, file in matching_files]
for file_name in matching_files:
    if os.path.exists(f'./mydata/raw/{file_name}'):
        csv_files.append(file_name)

draw_lidar_from_csv(csv_files)