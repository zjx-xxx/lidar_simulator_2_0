import os
import sys
import math
import random
import time
import serial
import datetime
import serial.tools.list_ports
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk, messagebox, filedialog

from sympy import false

from predict import  predict
import torch
from model import NeuralNetwork
from model_reg import RegressionNetwork
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def map_exp(x, factor):
    return 0.5/(math.exp(- x * factor * x * factor) + 0.5)


#PID控制器
class PID:
    def __init__(self, Kp, Ki, Kd, output_limit=None, integral_limit=None, map=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.map = map

    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time  # 计算时间间隔
        if dt == 0:  # 避免除零错误
            dt = 1e-6

        self.integral += error * dt  # 积分项累积
        if self.integral_limit:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)  # 限制积分项

        derivative = (error - self.prev_error) / dt  # 计算微分项

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative  # PID 公式

        if self.output_limit:
            output = np.clip(output, -self.output_limit, self.output_limit)  # 限制输出，防止过大

        # if self.map:
        #     output = map_exp(output, 70)

        self.prev_error = error
        self.last_time = current_time

        return output

class Simulator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulator")
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        middle_frame = tk.Frame(main_frame)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.simulation_canvas = tk.Canvas(left_frame, bg="lightgray", width=800, height=800)
        self.simulation_canvas.pack(fill=tk.BOTH, expand=True)

        # self.canvas_root = self.simulation_canvas.winfo_toplevel()

        self.lidar_canvas = tk.Canvas(middle_frame, bg="lightgray", width=500, height=500)
        self.lidar_canvas.pack(side=tk.TOP)

        log_frame = tk.Frame(middle_frame)
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            log_frame, state="disabled", width=50, height=10, font=("等线", 12, "bold")
        )
        self.log_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        log_text_frame = tk.Frame(log_frame)
        log_text_frame.pack(side=tk.TOP, fill=tk.X)
        log_text_frame.grid_columnconfigure(0, weight=1)
        log_text_frame.grid_columnconfigure(1, weight=1)

        self.completed_polyline_segments = []  # 已完成的折线段（绿色）
        self.follow_step_counter = 0
        self.current_target_point = None

        self.timer_label = tk.Label(
            log_text_frame, text=f"用时: {0:>5d}", font=("等线", 12, "bold")
        )
        self.timer_label.grid(row=0, column=0, pady=5)

        self.collision_count_label = tk.Label(
            log_text_frame, text=f"碰撞: {0:>5d}", font=("等线", 12, "bold")
        )
        self.collision_count_label.grid(row=0, column=1, pady=5)

        self.car_control_velocity_label = tk.Label(
            log_text_frame, text=f"速度: {0:>5d}", font=("等线", 12, "bold")
        )
        self.car_control_velocity_label.grid(row=1, column=0, pady=5)

        self.road_label = tk.Label(
            log_text_frame, text=f"路径分类预测: 无", font=("等线", 12, "bold")
        )
        self.road_label.grid(row=2, column=0, pady=5)

        self.car_control_steering_label = tk.Label(
            log_text_frame, text=f"实际转角: {0:>5d}", font=("等线", 12, "bold")
        )
        self.car_control_steering_label.grid(row=3, column=0, pady=5)

        self.speed_input_label = tk.Label(
            log_text_frame, text="自动速度:", font=("等线", 12, "bold")
        )
        self.speed_input_label.grid(row=1, column=1, pady=5)
        self.speed_input = tk.Entry(log_text_frame, font=("等线", 12, "bold"), width=6)
        self.speed_input.insert(0, "50")
        self.speed_input.grid(row=1, column=2, pady=5)

        self.predicted_angle_label = tk.Label(
            log_text_frame, text="预测舵机角度: 无", font=("等线", 12, "bold")
        )
        self.predicted_angle_label.grid(row=3, column=1, pady=5)

        logo_frame = tk.Frame(right_frame)
        logo_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(right_frame)
        button_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, anchor=tk.N)

        serial_frame = tk.Frame(right_frame)
        serial_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        button_width = 20
        button_height = 2

        self.recording = False
        try:
            with open('index.txt', 'r') as f:
                self.dataindex = int(f.read())
        except FileNotFoundError:
            self.dataindex = 0
        self.data_dir = "./mydata/raw"

        load_default_map_button = tk.Button(
            button_frame,
            text="加载默认地图",
            command=self.load_default_map,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        load_default_map_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        load_map_button = tk.Button(
            button_frame,
            text="加载地图",
            command=self.load_map,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        load_map_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        toggle_map_edit_button = tk.Button(
            button_frame,
            text="开启/关闭地图编辑",
            command=self.toggle_map_edit,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        toggle_map_edit_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        save_map_button = tk.Button(
            button_frame,
            text="保存地图",
            command=self.save_map,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        save_map_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        set_car_pose_button = tk.Button(
            button_frame,
            text="设置小车位置/朝向",
            command=self.set_car_pose,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        set_car_pose_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        toggle_keyboard_control_button = tk.Button(
            button_frame,
            text="开启/关闭键盘控制",
            command=self.toggle_keyboard_control,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )

        # toggle_mouse_control_button = tk.Button(
        #     button_frame,
        #     text="开启/关闭鼠标跟随",
        #     command=self.toggle_mouse_control,
        #     width=button_width,
        #     height=button_height,
        #     font=("等线", 12, "bold")
        # )
        # toggle_mouse_control_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        toggle_keyboard_control_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        toggle_clear_data_button = tk.Button(
            button_frame,
            text="清除数据",
            command=self.toggle_clear_data,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        toggle_clear_data_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)


        toggle_auto_control_button = tk.Button(
            button_frame,
            text="开启/关闭自动控制",
            command=self.toggle_auto_control,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        toggle_auto_control_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        clear_log_button = tk.Button(
            button_frame,
            text="清除日志",
            command=self.clear_log,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        clear_log_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        exit_button = tk.Button(
            button_frame,
            text="退出程序",
            command=self.quit,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        exit_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        undo_polyline_button = tk.Button(
            button_frame,
            text="撤销上一个折线点",
            command=self.undo_last_polyline_point,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        undo_polyline_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        self.start_dataindex = None
        self.end_dataindex = None

        # delete_mouse_data_button = tk.Button(
        #     button_frame,
        #     text="撤销上次鼠标记录",
        #     command=self.delete_last_mouse_data,
        #     width=button_width,
        #     height=button_height,
        #     font=("等线", 12, "bold")
        # )
        # delete_mouse_data_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        # --- 在 start_polyline_following() 中添加记录起始索引 ---
        self.polyline_record_start = self.dataindex

        # --- 在 follow_polyline() 完成时，添加记录结束索引 ---
        self.polyline_record_end = self.dataindex

        # --- 添加按钮 ---
        delete_polyline_data_button = tk.Button(
            button_frame,
            text="撤销折线循迹记录",
            command=self.delete_last_polyline_data,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        delete_polyline_data_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        logo1_image = Image.open(resource_path("data/logo1.png"))
        logo1_image = ImageTk.PhotoImage(logo1_image)
        logo1_label = tk.Label(logo_frame, image=logo1_image)
        logo1_label.image = logo1_image
        logo1_label.pack()

        logo2_image = Image.open(resource_path("data/logo2.png"))
        logo2_image = ImageTk.PhotoImage(logo2_image)
        logo2_label = tk.Label(logo_frame, image=logo2_image)
        logo2_label.image = logo2_image
        logo2_label.pack()

        self.grid = np.zeros((40, 40), dtype=np.uint8)

        # 新增扰动参数

        # 转向扰动
        self.steering_perturbation_enabled = True  # 是否启用转向扰动
        self.steering_perturbation_prob = 0.1  # 扰动发生概率 (20%)
        self.steering_perturbation_counter = 0  # 扰动计数器
        self.steering_perturbation_interval = 10  # 扰动最小间隔(更新次数)
        self.steering_perturbation_range = 8  # 转向扰动范围(度)

        # 位移扰动
        self.position_perturbation_enabled = True  # 是否启用位移扰动
        self.position_perturbation_prob = 0.06  # 扰动发生概率 (10%)
        self.position_perturbation_counter = 0  # 扰动计数器
        self.position_perturbation_interval = 10  # 扰动最小间隔(更新次数)
        self.position_perturbation_range = 10  # 位移扰动范围(像素)

        self.on_mouse_control = False
        self.in_map_edit = False
        self.in_keyboard_control = False
        self.in_auto_control = False
        self.in_timing = False
        self.in_set_car_pose = False
        self.data = np.zeros((1, 360))
        self.model = NeuralNetwork()
        self.model.load_state_dict(torch.load('./model/model', weights_only=True))

        self.key_state = {'w': False, 's': False, 'a': False, 'd': False}
        self.key_press_time = {}

        self.car = None
        self.car_pose = None
        self.car_control_value = np.zeros((2, 1), dtype=np.float32)
        self.road = -1
        self.car_size = 30
        self.car_image = Image.open(resource_path("data/car.png")).resize((self.car_size, self.car_size))
        self.car_photo = ImageTk.PhotoImage(self.car_image)

        self.velocity_command = ""
        self.steering_command = ""
        self.command_state = -1
        self.deltatheta = 0

        self.map_type = 0
        self.simulation_time_interval = 0.02
        self.lidar_scan_time_interval = 0.2
        self.time_interval = 0.2
        self.lidar_result = []

        self.collision_count = 0
        self.collided_obstacles = set()

        self.timestamp = None
        self.load_default_map()
        self.update()
        self.lidar_scan()

        # 初始化两个模型
        self.model_cls = NeuralNetwork()
        self.model_cls.load_state_dict(torch.load('./model/model', weights_only=True))
        self.model_cls.eval()

        self.model_reg = RegressionNetwork()
        self.model_reg.load_state_dict(torch.load('./model/model_regression.pth'))
        self.model_reg.eval()

        self.predicted_angle = 0.0

        # 自动控制状态标志初始化
        self.in_auto_control = False
        self.polyline_following = False
        self.polyline_points = []
        self.polyline_index = 0
        self.drawing_polyline = False

        polyline_button = tk.Button(
            button_frame,
            text="绘制折线路径",
            command=self.toggle_polyline_drawing,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        polyline_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        follow_polyline_button = tk.Button(
            button_frame,
            text="开启折线循迹",
            command=self.toggle_polyline_following,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        follow_polyline_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        self.polyline_record_start = None
        self.polyline_record_end = None
        self.polyline_segments = []  # 保存多个折线段

        self.passed_points = []  # 已跟随的点（绿色）
        self.current_follow_point = None  # 当前跟随的前视点（红色）

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def draw_map(self):
        self.simulation_canvas.delete("all")
        grid_size = 20
        for row in range(40):
            for col in range(40):
                x1 = col * grid_size
                y1 = row * grid_size
                x2 = x1 + grid_size
                y2 = y1 + grid_size
                if self.grid[row][col] == 1:  # 障碍
                    self.simulation_canvas.create_rectangle(
                        x1, y1, x2, y2, fill="orange", outline="gray"
                    )
                elif self.grid[row][col] == 2:  # 终点
                    self.simulation_canvas.create_rectangle(
                        x1, y1, x2, y2, fill="red", outline="gray"
                    )
                else:  # 空地
                    self.simulation_canvas.create_rectangle(
                        x1, y1, x2, y2, fill="gray", outline="gray"
                    )

    def update(self):
        start_time = time.time()
        self.update_velocity()
        self.update_car()
        end_time = time.time()
        delay_time = self.simulation_time_interval - (end_time - start_time)
        self.after(int(max(delay_time, 0) * 1000), self.update)

    def update_mouse_control(self):
        if self.on_mouse_control:
            start_time = time.time()
            self.update_velocity()
            self.update_car()
            end_time = time.time()
            delay_time = self.simulation_time_interval - (end_time - start_time)
            self.after(5, self.follow_mouse)  # 没有括号！
        else:
            self.close_mouse_control()

    def delete_last_polyline_data(self):
        if self.polyline_record_start is None or self.polyline_record_end is None:
            self.log("没有找到可以删除的折线循迹记录段")
            return
        deleted = 0
        for i in range(self.polyline_record_start, self.polyline_record_end):
            file_path = f"./mydata/raw/data{i}.csv"
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted += 1
        self.log(
            f"已删除 {deleted} 个文件（data{self.polyline_record_start} 到 data{self.polyline_record_end - 1}）")
        self.polyline_record_start = None
        self.polyline_record_end = None

    def update_car(self):
        if self.car is not None:
            grid_size = 20
            next_car_pose = self.car_pose.copy()
            next_car_pose[2, 0] += self.car_control_value[0, 0] / (0.75 * self.car_size) * np.tan(
                np.deg2rad(self.car_control_value[1, 0])
            ) * self.simulation_time_interval
            next_car_pose[0, 0] += self.car_control_value[0, 0] * np.cos(
                next_car_pose[2, 0]) * self.simulation_time_interval
            next_car_pose[1, 0] += self.car_control_value[0, 0] * np.sin(
                next_car_pose[2, 0]) * self.simulation_time_interval
            car_x = next_car_pose[0, 0]
            car_y = next_car_pose[1, 0]
            car_radius = self.car_size / 2
            new_collided_obstacles = set()
            for row in range(40):
                for col in range(40):
                    if self.grid[row][col] == 1:  # 障碍物存在
                        # 计算障碍物的边界
                        obstacle_left = col * grid_size
                        obstacle_right = (col + 1) * grid_size
                        obstacle_top = row * grid_size
                        obstacle_bottom = (row + 1) * grid_size
                        # 找到小车中心与障碍物最近点
                        nearest_x = max(obstacle_left, min(car_x, obstacle_right))
                        nearest_y = max(obstacle_top, min(car_y, obstacle_bottom))
                        # 计算小车中心到障碍物边界的距离
                        dx = car_x - nearest_x
                        dy = car_y - nearest_y
                        # 如果距离小于小车半径，则发生碰撞
                        if (dx * dx + dy * dy) < (car_radius * car_radius):
                            new_collided_obstacles.add((row, col))
                            # 限制小车位置，使其与障碍物相切
                            distance = (dx ** 2 + dy ** 2) ** 0.5
                            if distance > 0:  # 避免除以零
                                overlap = car_radius - distance
                                car_x += (dx / distance) * overlap
                                car_y += (dy / distance) * overlap
            # 如果有新碰撞，更新碰撞次数
            if new_collided_obstacles and not self.collided_obstacles:
                self.collision_count += 1
            if not (self.collided_obstacles and new_collided_obstacles):
                self.car_pose[0, 0] = car_x
                self.car_pose[1, 0] = car_y
                self.car_pose[2, 0] = next_car_pose[2, 0]
            self.collided_obstacles = new_collided_obstacles
            self.collision_count_label.config(text=f"碰撞: {self.collision_count:>5d}")
            if self.in_auto_control:
                if self.check_collision(self.car_pose[0, 0], self.car_pose[1, 0], "destination"):
                    self.log("到达终点")
                    self.close_auto_control()
                self.timer_label.config(text=f"用时: {int(time.time() - self.timestamp):>5d}")
            self.draw_car()

    def load_default_map(self):
        if self.in_auto_control:
            self.log("请先关闭自动控制")
            return
        if self.in_keyboard_control:
            self.log("请先关闭键盘控制")
            return
        if self.in_map_edit:
            self.close_map_edit()
        self.grid = np.load(resource_path("data/map_default.npy"))
        self.draw_map()
        self.log("加载默认地图")
        if self.car is not None:
            self.simulation_canvas.delete(self.car)
            self.car = None

    def load_map(self):
        if self.in_auto_control:
            self.log("请先关闭自动控制")
            return
        if self.in_keyboard_control:
            self.log("请先关闭键盘控制")
            return
        if self.in_map_edit:
            self.close_map_edit()
        file_path = filedialog.askopenfilename(
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        if file_path:
            self.grid = np.load(file_path)
            self.draw_map()
            self.log(f"从{file_path}成功加载地图")
            if self.car is not None:
                self.simulation_canvas.delete(self.car)
                self.car = None

    def toggle_map_edit(self):
        if self.in_map_edit:
            self.close_map_edit()
        else:
            self.open_map_edit()

    def open_map_edit(self):
        if self.in_keyboard_control:
            self.log("请先关闭键盘控制")
            return
        if self.in_auto_control:
            self.log("请先关闭自动控制")
            return
        self.in_map_edit = True
        self.simulation_canvas.bind("<Button-1>", self.handle_map_edit_click)
        self.simulation_canvas.bind("<ButtonPress-3>", self.handle_set_destination_click)
        self.simulation_canvas.bind("<B1-Motion>", self.handle_map_edit_drag)
        self.simulation_canvas.bind("<B3-Motion>", self.handle_set_destination_drag)
        self.log("开启地图编辑")

    def close_map_edit(self):
        self.in_map_edit = False
        self.simulation_canvas.unbind("<Button-1>")
        self.simulation_canvas.unbind("<ButtonPress-3>")
        self.simulation_canvas.unbind("<B1-Motion>")
        self.simulation_canvas.unbind("<B3-Motion>")
        self.log("关闭地图编辑")

    def handle_map_edit_drag(self, event):
        grid_size = 20
        row = event.y // grid_size
        col = event.x // grid_size
        if 0 <= row < 40 and 0 <= col < 40:
            if self.map_type == 1 and self.grid[row][col] != 2:
                self.grid[row][col] = 1
                if self.car is not None:
                    if self.check_collision(self.car_pose[0, 0], self.car_pose[1, 0]):
                        self.grid[row][col] = 0
            elif self.map_type == 0 and self.grid[row][col] != 2:
                self.grid[row][col] = 0
            self.draw_single_block(row, col)

    def handle_map_edit_click(self, event):
        grid_size = 20  # 计算每个格子的像素大小
        row = event.y // grid_size
        col = event.x // grid_size
        if 0 <= row < 40 and 0 <= col < 40:
            if self.grid[row][col] == 0:
                self.grid[row][col] = 1
                self.map_type = 1
                if self.car is not None:
                    if self.check_collision(self.car_pose[0, 0], self.car_pose[1, 0]):
                        self.grid[row][col] = 0
            elif self.grid[row][col] == 1:
                self.grid[row][col] = 0
                self.map_type = 0
            else:
                self.map_type = 2
            self.draw_single_block(row, col)

    def handle_set_destination_click(self, event):
        grid_size = 20
        row = event.y // grid_size
        col = event.x // grid_size
        if 0 <= row < 40 and 0 <= col < 40:
            if self.grid[row][col] == 0:
                self.grid[row][col] = 2
                self.map_type = 2
            elif self.grid[row][col] == 1:
                self.map_type = 1
            else:
                self.grid[row][col] = 0
                self.map_type = 0
            self.draw_single_block(row, col)

    def handle_set_destination_drag(self, event):
        grid_size = 20
        row = event.y // grid_size
        col = event.x // grid_size
        if 0 <= row < 40 and 0 <= col < 40:
            if self.map_type == 2 and self.grid[row][col] != 1:
                self.grid[row][col] = 2
            elif self.map_type == 0 and self.grid[row][col] != 1:
                self.grid[row][col] = 0
            self.draw_single_block(row, col)

    def draw_single_block(self, row, col):
        grid_size = 20
        x1 = col * grid_size
        y1 = row * grid_size
        x2 = x1 + grid_size
        y2 = y1 + grid_size
        if self.grid[row][col] == 1:
            color = "orange"
        elif self.grid[row][col] == 2:
            color = "red"
        else:
            color = "gray"
        self.simulation_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

    def save_map(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
        )
        if file_path:
            np.save(file_path, self.grid)
            self.log(f"地图保存到: {file_path}")

    def set_car_pose(self):
        if self.in_auto_control:
            self.log("请先关闭自动控制")
            return
        if self.in_keyboard_control:
            self.log("请先关闭键盘控制")
            return
        if self.in_map_edit:
            self.log("请先关闭地图编辑")
            return
        if self.on_mouse_control:
            self.log("请先关闭鼠标跟随")
            return
        if self.car is not None:
            self.simulation_canvas.delete(self.car)
            self.car = None
        self.log("设置小车位置/朝向")
        self.simulation_canvas.bind("<Button-1>", self.handle_set_car_pose_click)

    def check_collision(self, car_x, car_y, collision_type="obstacle"):
        grid_size = 20
        car_radius = self.car_size / 2
        map_type = 1 if collision_type == "obstacle" else 2
        for row in range(40):
            for col in range(40):
                if self.grid[row][col] == map_type:  # 障碍物存在
                    nearest_x = max(col * grid_size, min(car_x, (col + 1) * grid_size))
                    nearest_y = max(row * grid_size, min(car_y, (row + 1) * grid_size))
                    # 计算小车中心到障碍物边界的距离
                    dx = car_x - nearest_x
                    dy = car_y - nearest_y
                    # 如果距离小于小车半径，则发生碰撞
                    if (dx * dx + dy * dy) < (car_radius * car_radius):
                        return True
        return False

    def handle_set_car_pose_click(self, event):
        if not self.in_set_car_pose:
            if self.check_collision(event.x, event.y):
                self.log("小车位置与障碍物冲突，请重新设置")
                return
            self.car_pose = np.array([event.x, event.y, 0], dtype=np.float32).reshape(3, 1)
            self.in_set_car_pose = True
        else:
            self.car_pose[2, 0] = np.arctan2(event.y - self.car_pose[1, 0], event.x - self.car_pose[0, 0])
            self.in_set_car_pose = False
            self.draw_car()
            self.simulation_canvas.unbind("<Button-1>")

    def draw_car(self):
        if self.car is not None:
            self.simulation_canvas.delete(self.car)

        car_image = self.car_image.rotate(np.rad2deg(-self.car_pose[2, 0] - np.pi / 2), expand=True)
        self.car_photo = ImageTk.PhotoImage(car_image)
        self.car = self.simulation_canvas.create_image(
            self.car_pose[0, 0],
            self.car_pose[1, 0],
            image=self.car_photo,
            anchor="center"
        )

    def toggle_keyboard_control(self):
        if self.in_keyboard_control:
            self.close_keyboard_control()
        else:
            self.open_keyboard_control()

    def toggle_mouse_control(self):
        if self.on_mouse_control:
            self.close_mouse_control()
        else:
            self.open_mouse_control()

    def open_keyboard_control(self):
        if self.in_auto_control:
            self.log("请先关闭自动控制")
            return
        if self.car is None:
            self.log("请先设置小车位置/朝向")
            return
        if self.in_map_edit:
            self.close_map_edit()
        if self.on_mouse_control:
            self.log("请先关闭鼠标跟随")
            return
        self.in_keyboard_control = True
        self.collision_count = 0
        self.bind("<KeyPress>", self.on_key_press)
        self.bind("<KeyRelease>", self.on_key_release)
        self.log("开启键盘控制")

    def close_keyboard_control(self):
        self.in_keyboard_control = False
        self.simulation_canvas.unbind("<KeyPress>")
        self.simulation_canvas.unbind("<KeyRelease>")
        self.log("关闭键盘控制")

    def open_mouse_control(self):
        if self.in_auto_control:
            self.log("请先关闭自动控制")
            return
        if self.car is None:
            self.log("请先设置小车位置/朝向")
            return
        if self.in_map_edit:
            self.close_map_edit()
        if self.in_keyboard_control:
            self.log("请先关闭键盘控制")
            return
        # self.on_mouse_control = True
        self.start_dataindex = self.dataindex  #记录起始索引
        self.log("打开鼠标跟随")
        # 在 __init__ 里初始化 PID 控制器
        self.steering_pid = PID(Kp=3.0, Ki=0.2, Kd=0.6, output_limit=30, map=1)
        self.speed_pid = PID(Kp=3.0, Ki=0.15, Kd=0.8, output_limit=10)
        self.simulation_canvas.bind("<Button-1>", self.mouse_control)

    def delete_last_mouse_data(self):
        if self.start_dataindex is None or self.end_dataindex is None:
            self.log("没有找到可以删除的鼠标记录段")
            return
        deleted = 0
        for i in range(self.start_dataindex, self.end_dataindex):
            file_path = f"./mydata/raw/data{i}.csv"
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted += 1
        self.log(f"已删除 {deleted} 个文件（data{self.start_dataindex} 到 data{self.end_dataindex - 1}）")
        self.start_dataindex = None
        self.end_dataindex = None


    def close_mouse_control(self):
        self.on_mouse_control = False
        self.simulation_canvas.unbind("<Button-1>")
        #这里需要补充关闭记录
        self.recording = False
        self.end_dataindex = self.dataindex  # 记录结束索引
        self.log("关闭鼠标跟随，数据段索引 {} 到 {}".format(self.start_dataindex, self.end_dataindex))

    def on_key_press(self, event):
        key = event.keysym.lower()
        if key in self.key_state and not self.key_state[key]:
            self.key_state[key] = True
            self.key_press_time[key] = time.time()

    def on_key_release(self, event):
        key = event.keysym.lower()
        if key in self.key_state:
            self.key_state[key] = False

    def update_velocity(self):
        # 在计算控制量之前添加转向扰动
        if (self.steering_perturbation_enabled and
                self.recording and
                self.steering_perturbation_counter >= self.steering_perturbation_interval and
                random.random() < self.steering_perturbation_prob):

            # 生成随机扰动角度
            steering_perturb = random.uniform(-self.steering_perturbation_range, self.steering_perturbation_range)

            # 应用扰动到当前转向角度
            self.car_control_value[1, 0] += steering_perturb

            # 日志记录
            self.log(f"施加转向扰动: {steering_perturb:.2f}°")

            # 重置计数器
            self.steering_perturbation_counter = 0
        else:
            self.steering_perturbation_counter += 1

        # 在计算控制量之前添加位移扰动
        if (self.position_perturbation_enabled and
                self.recording and
                self.position_perturbation_counter >= self.position_perturbation_interval and
                random.random() < self.position_perturbation_prob):

            # 生成随机位移扰动
            angle = random.uniform(0, 2 * math.pi)  # 随机方向
            distance = random.uniform(3, self.position_perturbation_range)

            # 应用位移扰动
            self.car_pose[0, 0] += distance * math.cos(angle)
            self.car_pose[1, 0] += distance * math.sin(angle)

            # 日志记录
            self.log(f"施加位移扰动: 方向={math.degrees(angle):.1f}°, 距离={distance:.2f}px")

            # 重置计数器
            self.position_perturbation_counter = 0
        else:
            self.position_perturbation_counter += 1


        if self.in_keyboard_control:
            for key, pressed in self.key_state.items():
                if pressed:
                    press_duration = time.time() - self.key_press_time.get(key, time.time())
                    self.key_press_time[key] = time.time()
                    if key == 'w':
                        self.car_control_value[0, 0] += press_duration * 200
                    elif key == 's':
                        self.car_control_value[0, 0] -= press_duration * 200
                    elif key == 'a':
                        self.car_control_value[1, 0] -= press_duration * 30
                    elif key == 'd':
                        self.car_control_value[1, 0] += press_duration * 30

        second_column_as_row = [row[1] for row in self.lidar_result]

        if len(second_column_as_row) == 360:
            X_cls = [second_column_as_row]
            self.road = predict(self.model_cls, X_cls)

        if self.in_auto_control and len(second_column_as_row) == 360:
            input_reg = torch.tensor([second_column_as_row + [self.deltatheta]], dtype=torch.float32)
            with torch.no_grad():
                pred_angle = self.model_reg(input_reg).item()
                pred_angle = np.clip(pred_angle, -30, 30)
                self.car_control_value[1, 0] = pred_angle
                self.predicted_angle = pred_angle
                self.predicted_angle_label.config(text=f"预测舵机角度: {self.predicted_angle:.2f} °")

            try:
                speed = float(self.speed_input.get())
            except ValueError:
                speed = 50.0
            self.car_control_value[0, 0] = np.clip(speed, -200, 200)

        if not self.in_auto_control:
            if self.in_keyboard_control:
                velocity_decrease_flag = True
                steering_decrease_flag = True
                for key, pressed in self.key_state.items():
                    if pressed:
                        if key in ['w', 's']:
                            velocity_decrease_flag = False
                        if key in ['a', 'd']:
                            steering_decrease_flag = False
                if velocity_decrease_flag:
                    self.car_control_value[0, 0] *= 0.95
                if steering_decrease_flag:
                    self.car_control_value[1, 0] *= 0.85
            else:
                self.car_control_value[0, 0] *= 0.95
                self.car_control_value[1, 0] *= 0.85

        self.car_control_value[0, 0] = np.maximum(np.minimum(self.car_control_value[0, 0], 200), -200)
        self.car_control_value[1, 0] = np.maximum(np.minimum(self.car_control_value[1, 0], 30), -30)

        self.car_control_velocity_label.config(text=f"速度: {int(self.car_control_value[0, 0]):>5d}")
        self.road_label.config(text=f"路径分类预测: {self.road}")
        self.car_control_steering_label.config(text=f"实际转角: {int(self.car_control_value[1, 0]):>5d}")

    def lidar_scan(self):
        start_time = time.time()
        if self.car is not None:
            max_distance = 200
            x1, y1 = self.car_pose[0, 0], self.car_pose[1, 0]
            theta = self.car_pose[2, 0]
            self.deltatheta = self.deltatheta + theta
            self.lidar_result = []
            for angle_increment in range(360):
                angle = theta + np.deg2rad(angle_increment)
                x2 = x1 + max_distance * np.cos(angle)
                y2 = y1 + max_distance * np.sin(angle)
                distance = self.lidar_detect(x1, y1, x2, y2)
                self.lidar_result.append((angle_increment, distance))
            self.draw_lidar()
        end_time = time.time()
        delay_time = self.lidar_scan_time_interval - (end_time - start_time)
        self.after(int(max(delay_time, 0) * 1000), self.lidar_scan)
        # self.lidar_result.append(self.car_control_value[1,0])

        df = pd.DataFrame(self.lidar_result, columns=['degree', 'distance'])
        if(len(df) != 0 and self.recording):
            second_column = df['distance'].transpose()
            df_t = pd.Series(int(self.car_control_value[1, 0]))
            second_column = pd.concat([second_column,df_t])
            file_path = f'./mydata/raw/data{self.dataindex}.csv'
            while os.path.exists(file_path):
                self.dataindex += 1
                file_path = f'./mydata/raw/data{self.dataindex}.csv'
            second_column.to_csv(file_path, index=False, header=False)
            self.dataindex = self.dataindex + 1
            with open('index.txt', 'w') as f:
                f.write(str(self.dataindex))

    # def model_control(self):
    #     start_time = time.time()
    #     if self.in_auto_control:
    #         i = 0
    #         for _, distance in self.lidar_result:
    #             self.data[0, i] = distance
    #             i = i + 1
    #         speed, angle = car_control(self.data)
    #         # speed, angle = predict(self.model, self.data)
    #         self.car_control_value[0, 0] = speed
    #         self.car_control_value[1, 0] = angle
    #     end_time = time.time()
    #     delay_time = self.time_interval - (end_time - start_time)
    #     self.after(int(max(delay_time, 0) * 1000), self.model_control)

    def close_mouse_control_handle(self, event):
        self.close_mouse_control()

    def mouse_click(self, event):
        self.simulation_canvas.bind("<Button-1>", self.close_mouse_control_handle)
        self.follow_mouse()

    def follow_mouse(self):
        if self.on_mouse_control:
            x_mouse, y_mouse = self.winfo_pointerx(), self.winfo_pointery()
            widget_x = self.winfo_rootx()
            widget_y = self.winfo_rooty()
            x_mouse -= widget_x
            y_mouse -= widget_y
            # self.log(f"{x_mouse}, {y_mouse}")
            #这里需要补充开始记录
            self.recording = True
            x_car, y_car = self.car_pose[0, 0], self.car_pose[1, 0]

            theta_car = self.car_pose[2, 0]
            dx = x_mouse - x_car
            dy = y_mouse - y_car

            target_theta = np.arctan2(dy, dx)  # 期望航向角
            # self.log(f"{dx}, {dy}, {target_theta}")
            # 计算方向误差（当前角度 - 目标角度）
            heading_error = (target_theta - theta_car + np.pi) % (2 * np.pi) - np.pi
            heading_error = np.rad2deg(heading_error)


            # 计算舵机控制量
            steering_control = self.steering_pid.compute(heading_error)
            # 计算速度控制量（根据距离调整速度）
            distance_to_target = np.hypot(dx, dy)
            speed_control = self.speed_pid.compute(distance_to_target)

            # self.log(f"{speed_control}")

            speed_control = np.clip(speed_control, -10, 10)
            speed_control_mapped = np.interp(speed_control, [-10, 10], [-80, 80])
            self.car_control_value[0,0] = speed_control_mapped
            self.car_control_value[1,0] = steering_control
            self.after(5, self.update_mouse_control)  # 固定 5ms 更新 没有括号！
        else:
            self.close_mouse_control()
            return




    def mouse_control(self, event):
        if not self.on_mouse_control:
            self.on_mouse_control = True
            self.mouse_click(event)
        else:
            self.on_mouse_control = False

    def toggle_polyline_drawing(self):
        self.polyline_points = []
        self.polyline_index = 0
        if self.drawing_polyline:
            self.simulation_canvas.unbind("<Button-1>")
            self.drawing_polyline = False
            self.log("关闭折线路径绘制")
        else:
            self.polyline_points = []
            self.polyline_index = 0
            self.simulation_canvas.bind("<Button-1>", self.add_polyline_point)
            self.drawing_polyline = True
            self.log("点击画布绘制折线路径")

    def add_polyline_point(self, event):
        self.polyline_points.append((event.x, event.y))
        if len(self.polyline_points) > 1:
            self.simulation_canvas.create_line(
                *self.polyline_points[-2], *self.polyline_points[-1], fill="blue", width=3
            )

    def toggle_polyline_following(self):
        if self.polyline_following == False:
            self.start_polyline_following()
        else:
            self.close_polyline_following()

    def start_polyline_following(self):
        self.polyline_following = True
        self.polyline_record_start = self.dataindex
        if self.car is None:
            self.log("请先设置小车位置")
            return
        if not self.polyline_points:
            self.log("请先绘制折线路径")
            return
        self.log("开始折线循迹")
        self.polyline_index = 0
        self.steering_pid = PID(Kp=6.0, Ki=0.1, Kd=0.3, output_limit=30)
        self.speed_pid = PID(Kp=3.0, Ki=0.15, Kd=0.8, output_limit=10)
        self.recording = True
        self.followed_path_points = []  # 重置走过的点
        self.follow_step_counter = 0
        self.follow_polyline()

    def close_polyline_following(self):
        self.polyline_following = False
        self.recording = False
        self.log("关闭折线循迹")
        self.car_control_value[:, 0] = 0
        self.polyline_record_end = self.dataindex
        self.log(f"折线循迹结束，记录数据索引：{self.polyline_record_start} 到 {self.polyline_record_end}")

    def follow_polyline(self):
        if self.polyline_index >= len(self.polyline_points):
            self.close_polyline_following()
            return

        car_x, car_y = self.car_pose[0, 0], self.car_pose[1, 0]
        theta_car = self.car_pose[2, 0]

        # 每隔几帧记录小车位置作为轨迹点
        self.follow_step_counter += 1
        if self.follow_step_counter % 3 == 0:
            self.followed_path_points.append((car_x, car_y))

        target_x, target_y = self.polyline_points[self.polyline_index]
        dx = target_x - car_x
        dy = target_y - car_y
        distance_to_target = np.hypot(dx, dy)
        self.current_target_point = (target_x, target_y)

        speed_control = self.speed_pid.compute(distance_to_target)
        speed_control = np.clip(speed_control, -10, 10)

        base_lookahead = 10.0
        variable_component = 0.5 * abs(speed_control)
        lookahead_distance = np.clip(base_lookahead + variable_component, 10, 25)

        if distance_to_target > 1e-5:
            unit_dx = dx / distance_to_target
            unit_dy = dy / distance_to_target
            actual_lookahead = min(lookahead_distance, distance_to_target)
            lookahead_x = car_x + unit_dx * actual_lookahead
            lookahead_y = car_y + unit_dy * actual_lookahead
        else:
            lookahead_x, lookahead_y = target_x, target_y

        self.current_follow_point = (lookahead_x, lookahead_y)

        dx_la = lookahead_x - car_x
        dy_la = lookahead_y - car_y
        target_theta = np.arctan2(dy_la, dx_la)
        heading_error = (target_theta - theta_car + np.pi) % (2 * np.pi) - np.pi
        heading_error = np.rad2deg(heading_error)

        steering_control = self.steering_pid.compute(heading_error)
        speed_mapped = np.interp(speed_control, [-10, 10], [-80, 80])

        self.car_control_value[0, 0] = speed_mapped
        self.car_control_value[1, 0] = steering_control

        while distance_to_target < 5 and self.polyline_index < len(self.polyline_points) - 1:
            self.polyline_index += 1
            target_x, target_y = self.polyline_points[self.polyline_index]
            dx = target_x - car_x
            dy = target_y - car_y
            distance_to_target = np.hypot(dx, dy)

        self.draw_follow_points()
        self.after(50, self.follow_polyline)

    def draw_follow_points(self):
        self.simulation_canvas.delete("follow_point")

        # 1. 原始折线：蓝色
        for i in range(1, len(self.polyline_points)):
            x1, y1 = self.polyline_points[i - 1]
            x2, y2 = self.polyline_points[i]
            self.simulation_canvas.create_line(
                x1, y1, x2, y2, fill='blue', width=2, tags="follow_point"
            )

        # 2. 实际走过的轨迹线段：绿色
        for i in range(1, len(self.followed_path_points)):
            x1, y1 = self.followed_path_points[i - 1]
            x2, y2 = self.followed_path_points[i]
            self.simulation_canvas.create_line(
                x1, y1, x2, y2, fill='green', width=2, tags="follow_point"
            )

        # 3. 当前前视点：红色圆点
        if self.current_follow_point:
            x, y = self.current_follow_point
            self.simulation_canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5, fill='red', outline='white', width=2, tags="follow_point")

        # 4. 当前路径点（目标点）：黄色圆点
        if self.current_target_point:
            x, y = self.current_target_point
            self.simulation_canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4, fill='yellow', outline='black', width=1, tags="follow_point"
            )

    def lidar_detect(self, x1, y1, x2, y2):
        grid_size = 20
        x1_grid, y1_grid = int(x1 // grid_size), int(y1 // grid_size)
        x2_grid, y2_grid = int(x2 // grid_size), int(y2 // grid_size)
        dx = x2 - x1
        dy = y2 - y1

        x_crossings = []
        y_crossings = []
        min_distance = 200
        if dx != 0:
            x_steps = range(min(x1_grid, x2_grid) + 1, max(x1_grid, x2_grid) + 1)
            x_crossings = [x * grid_size for x in x_steps]
        if dy != 0:
            y_steps = range(min(y1_grid, y2_grid) + 1, max(y1_grid, y2_grid) + 1)
            y_crossings = [y * grid_size for y in y_steps]
        for xc in x_crossings:
            t = (xc - x1) / dx
            yc = y1 + t * dy
            row = math.floor(yc / grid_size) if dy >= 0 else math.ceil(yc / grid_size) - 1
            col = math.floor(xc / grid_size) if dx >= 0 else math.ceil(xc / grid_size) - 1
            if 0 <= row < 40 and 0 <= col < 40 and self.grid[row][col] == 1:
                distance = np.sqrt((xc - x1) ** 2 + (yc - y1) ** 2)
                min_distance = min(distance, min_distance)
        for yc in y_crossings:
            t = (yc - y1) / dy
            xc = x1 + t * dx
            row = math.floor(yc / grid_size) if dy >= 0 else math.ceil(yc / grid_size) - 1
            col = math.floor(xc / grid_size) if dx >= 0 else math.ceil(xc / grid_size) - 1
            if 0 <= row < 40 and 0 <= col < 40 and self.grid[row][col] == 1:
                distance = np.sqrt((xc - x1) ** 2 + (yc - y1) ** 2)
                min_distance = min(distance, min_distance)
        return int(min_distance)

    def draw_lidar(self):
        self.lidar_canvas.delete("all")
        center_x, center_y = 250, 250  # 假设雷达画布的中心
        radius = 220  # 圆的半径，缩小到220

        self.lidar_canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            outline="black",
        )

        for i in range(12):
            angle = i * 30
            end_x = center_x + radius * np.cos(np.deg2rad(angle))
            end_y = center_y + radius * np.sin(np.deg2rad(angle))
            self.lidar_canvas.create_line(
                center_x, center_y, end_x, end_y, fill="black"
            )

        for angle, distance in self.lidar_result:
            if distance == 200:
                continue
            angle = np.deg2rad(angle - 90)
            end_x = center_x + 1.1 * distance * np.cos(angle)
            end_y = center_y + 1.1 * distance * np.sin(angle)
            self.lidar_canvas.create_oval(
                end_x - 1, end_y - 1, end_x + 1, end_y + 1, fill="red", outline="red"
            )

        arrow_length = 20
        end_arrow_x = center_x
        end_arrow_y = center_y - arrow_length
        self.lidar_canvas.create_line(
            center_x,
            center_y,
            end_arrow_x,
            end_arrow_y,
            arrow=tk.LAST,
            fill="blue",
            width=2,
        )

        for i in range(12):
            angle = i * 30
            distance = next((d for a, d in self.lidar_result if a == angle), None)
            if distance is not None:
                text_x = center_x + (radius + 20) * np.cos(np.deg2rad(angle - 90))
                text_y = center_y + (radius + 20) * np.sin(np.deg2rad(angle - 90))
                self.lidar_canvas.create_text(
                    text_x, text_y, text=f"{distance}", fill="blue"
                )

    def toggle_auto_control(self):
        if self.in_auto_control:
            self.close_auto_control()
        else:
            self.open_auto_control()

    def open_auto_control(self):
        if self.in_keyboard_control:
            self.log("请先关闭键盘控制")
            return
        if self.on_mouse_control:
            self.log("请先关闭鼠标控制")
            return
        if self.car is None:
            self.log("请先设置小车位置/朝向")
            return
        if self.in_map_edit:
            self.close_map_edit()
        self.in_auto_control = True
        self.timestamp = time.time()
        self.collision_count = 0
        self.log("开启自动控制（使用模型预测）")

    def toggle_clear_data(self):
        if self.on_mouse_control:
            self.log("请关闭鼠标跟随")
            return
        if not os.listdir(self.data_dir):
            popup = tk.Toplevel(self)  # 创建一个新的窗口
            popup.title("清除数据")
            popup.geometry("200x100")

            label = tk.Label(popup, text="文件夹已经为空")
            label.pack(pady=10)

            close_button = tk.Button(popup, text="关闭", command=popup.destroy)
            close_button.pack()
        else:
            popup = tk.Toplevel(self)  # 创建一个新的窗口
            popup.title("清除数据")
            popup.geometry("200x100")

            label = tk.Label(popup, text="您确定要删除文件夹中的所有数据吗？")
            label.pack(pady=10)

            enter_button = tk.Button(popup, text="确认", command=self.clear_data)
            enter_button.pack()

    def clear_data(self):
        directory = self.data_dir
        for filename in os.listdir(directory):
            if filename != "raw.md":
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):  # 仅删除文件，不删除子目录
                    os.remove(file_path)
        self.log("已经删除所有数据")
        return

    def close_auto_control(self):
        self.in_auto_control = False
        self.log("关闭自动控制")

    def clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    def undo_last_polyline_point(self):
        if not self.polyline_points:
            self.log("没有可以撤销的折线点")
            return
        self.polyline_points.pop()
        self.redraw_polyline()
        self.log("已撤销最近一个折线点")

    # --- 重绘路径 ---
    def redraw_polyline(self):
        self.draw_map()
        for i in range(1, len(self.polyline_points)):
            x1, y1 = self.polyline_points[i - 1]
            x2, y2 = self.polyline_points[i]
            self.simulation_canvas.create_line(x1, y1, x2, y2, fill="blue", width=3)
        if self.car:
            self.draw_car()


if __name__ == "__main__":
    Simulator().mainloop()
