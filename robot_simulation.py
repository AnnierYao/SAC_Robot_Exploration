from scipy import spatial  # 用于KD树
from skimage import io  # 用于图像读取
import numpy as np  # 主要的数值计算库
import numpy.ma as ma  # 处理掩码数组
import time  # 时间相关函数
import sys  # 与系统交互
from scipy import ndimage  # 用于图像处理
import matplotlib.pyplot as plt  # 用于绘图
from collections import deque
from scipy.spatial import distance

sys.path.append(sys.path[0] + '/..')  # 添加上级目录到模块搜索路径

# 从自定义模块中导入函数
from build.inverse_sensor_model import *  # 逆传感器模型
from build.astar import *  # A*算法
from random import shuffle  # 随机排序列表
import os  # 与操作系统交互

class Robot:
    def __init__(self, index_map, train, plot):
        """
        Initialize the Robot class.

        Args:
            index_map (int): The index of the map to be used.
            train (bool): Flag indicating whether the robot is in training mode.
            plot (bool): Flag indicating whether to enable plotting.

        Attributes:
            mode (bool): Flag indicating the training mode.
            plot (bool): Flag indicating whether plotting is enabled.
            map_dir (str): Directory path of the maps.
            map_list (list): List of map files.
            map_number (int): Number of maps.
            li_map (int): Index of the current map.
            global_map (ndarray): Processed map with 255 representing free space and 1 representing obstacles.
            robot_position (ndarray): Position of the robot.
            op_map (ndarray): Operation map with 127 representing unexplored areas.
            map_size (tuple): Size of the map.
            finish_percent (float): Completion percentage for exploration.
            resolution (int): Resolution of the map.
            sensor_range (int): Range of the robot's sensors.
            old_position (ndarray): Previous position of the robot.
            old_op_map (ndarray): Previous operation map.
            action_space (ndarray): Action space for the robot.
            t (ndarray): Map points.
            free_tree (KDTree): KDTree of free points.
            robot_size (int): Size of the robot.
            local_size (int): Size of the local map.
            xPoint (ndarray): X-coordinates for plotting the robot's position.
            yPoint (ndarray): Y-coordinates for plotting the robot's position.
            x2frontier (ndarray): X-coordinates for plotting the frontier.
            y2frontier (ndarray): Y-coordinates for plotting the frontier.
        """
        # Code implementation goes here
        # 初始化机器人
        self.mode = train  # 训练模式标志
        self.plot = plot  # 绘图标志
        if self.mode:
            self.map_dir = './DungeonMaps/train'  # 训练地图目录
        else:
            self.map_dir = './DungeonMaps/test'  # 测试地图目录
        self.map_list = sorted(os.listdir(self.map_dir))  # 地图文件列表
        self.map_number = np.size(self.map_list)  # 地图数量
        # if self.mode:
        #     shuffle(self.map_list)  # 如果是训练模式，随机排序地图列表
        self.li_map = index_map  # 地图索引
        self.map_step = 0  # 地图步数
        # 返回处理过的地图和机器人位置，其中255表示free space，1表示obstacle
        self.global_map, self.robot_position = self.map_setup(self.map_dir + '/' + self.map_list[self.li_map])  # 初始化地图和机器人位置
        # 创建一个与global_map相同大小的数组，数组中的元素全为127，表示unexplored area
        self.op_map = np.ones(self.global_map.shape) * 127  # 初始化操作地图
        self.latent_map = np.ones(self.global_map.shape) * 127  # 初始化隐藏地图
        self.map_size = np.shape(self.global_map)  # 地图尺寸
        self.finish_percent = 0.985  # 探索完成百分比
        self.resolution = 1 
        self.sensor_range = 80  # 传感器范围
        self.old_position = np.zeros([2])  # 上一个位置
        self.old_op_map = np.empty([0])  # 上一个操作地图
        current_dir = os.path.dirname(os.path.realpath(__file__))  # 当前文件所在目录
        self.action_space = np.genfromtxt(current_dir + '/scripts/action_points.csv', delimiter=",")  # 行动空间
        self.t = self.map_points(self.global_map)  # 地图点
        self.free_tree = spatial.KDTree(self.free_points(self.global_map).tolist())  # 自由点的KD树
        self.robot_size = 6  # 机器人大小
        self.local_size = 40  # 局部地图尺寸
        if self.plot:
            self.xPoint = np.array([self.robot_position[0]])  # 绘图用的机器人所在位置的x坐标
            self.yPoint = np.array([self.robot_position[1]])  # 绘图用的机器人所在位置的y坐标
            self.x2frontier = np.empty([0])  # 绘图用的边界x坐标
            self.y2frontier = np.empty([0])  # 绘图用的边界y坐标

    def begin(self):
        """
        Begins the exploration process.

        Returns:
            map_local (numpy.ndarray): The local map generated based on the robot's position and sensor readings.
        """
        self.op_map = self.inverse_sensor(self.robot_position, self.sensor_range, self.op_map, self.global_map)  # 更新操作地图
        step_map = self.robot_model(self.robot_position, self.robot_size, self.t, self.op_map)  # 获取机器人模型
        map_local = self.local_map(self.robot_position, step_map, self.map_size, self.sensor_range + self.local_size)  # 获取局部地图
        if self.plot:
            self.plot_env()  # 绘制环境
        self.latent_map = self.op_map.copy()  # 更新隐藏地图
        # return map_local, self.latent_map  # 返回局部地图
        return map_local

    def step(self, action_index):
        # 执行步骤
        terminal = False  # 终止标志
        complete = False  # 完成标志
        new_location = False  # 新位置标志
        finish_all_map = False  # 所有地图标志
        self.old_position = self.robot_position.copy()  # 保存上一个位置
        self.old_op_map = self.op_map.copy()  # 保存上一个操作地图

        # 执行动作
        self.take_action(action_index, self.robot_position)
        self.map_step += 1

        # 碰撞检测
        collision_points, collision_index = self.collision_check(self.old_position, self.robot_position, self.map_size,
                                                                 self.global_map)

        if collision_index:
            # 如果发生碰撞
            self.robot_position = self.nearest_free(self.free_tree, collision_points)  # 找到最近的自由点
            self.op_map = self.inverse_sensor(self.robot_position, self.sensor_range, self.op_map, self.global_map)  # 更新操作地图
            step_map = self.robot_model(self.robot_position, self.robot_size, self.t, self.op_map)  # 获取机器人模型
        else:
            # 如果没有发生碰撞
            self.op_map = self.inverse_sensor(self.robot_position, self.sensor_range, self.op_map, self.global_map)  # 更新操作地图
            step_map = self.robot_model(self.robot_position, self.robot_size, self.t, self.op_map)  # 获取机器人模型
 
        map_local = self.local_map(self.robot_position, step_map, self.map_size, self.sensor_range + self.local_size)  # 获取局部地图
        boundaries = self.detect_map_boundaries(self.op_map)
        num_sequences, sequence_lengths, boundary_list, center_list = self.count_sequences(boundaries, self.robot_position)
        self.latent_map, _ = self.cal_latent_s(self.latent_map, sequence_lengths, boundary_list, center_list)
        # reward = self.get_extrinsic_reward(self.old_op_map, self.op_map, collision_index) + self.get_intrinsic_reward(self.latent_map, self.robot_position)  # 获取奖励
        reward = self.get_extrinsic_reward(self.old_op_map, self.op_map, collision_index)

        if reward <= 0.02 and not collision_index:
            # 如果奖励小于等于0.02且没有发生碰撞
            reward = -0.8  # 设置奖励为-0.8
            new_location = True  # 新位置标志为True
            terminal = True  # 终止标志为True

        if collision_index:
            # 如果发生碰撞
            if not self.mode:
                # 如果是测试模式
                new_location = False  # 新位置标志为False
                terminal = False  # 终止标志为False
            else:
                # 如果是训练模式
                new_location = True  # 新位置标志为True
                terminal = True  # 终止标志为True
            if self.plot and self.mode:
                # 如果是训练模式且绘图标志为True
                self.xPoint = ma.append(self.xPoint, self.robot_position[0])  # 添加机器人x坐标到绘图列表
                self.yPoint = ma.append(self.yPoint, self.robot_position[1])  # 添加机器人y坐标到绘图列表
                self.plot_env()  # 绘制环境
            self.robot_position = self.old_position.copy()  # 恢复机器人位置
            self.op_map = self.old_op_map.copy()  # 恢复操作地图
            if self.plot and self.mode:
                # 如果是训练模式且绘图标志为True
                self.xPoint[self.xPoint.size-1] = ma.masked  # 将绘图列表的最后一个x坐标标记为掩码
                self.yPoint[self.yPoint.size-1] = ma.masked  # 将绘图列表的最后一个y坐标标记为掩码
        else:
            # 如果没有发生碰撞
            if self.plot:
                # 如果绘图标志为True
                self.xPoint = ma.append(self.xPoint, self.robot_position[0])  # 添加机器人x坐标到绘图列表
                self.yPoint = ma.append(self.yPoint, self.robot_position[1])  # 添加机器人y坐标到绘图列表
                self.plot_env()  # 绘制环境

        if np.size(np.where(self.op_map == 255))/np.size(np.where(self.global_map == 255)) > self.finish_percent:
            # 如果探索完成百分比大于规定的完成百分比
            self.li_map += 1  # 地图索引加1
            if self.li_map == self.map_number:
                # 如果地图索引等于地图数量
                self.li_map = 0  # 地图索引归零
                finish_all_map = True  # 所有地图标志为True
            self.__init__(self.li_map, self.mode, self.plot)  # 重新初始化
            complete = True  # 完成标志为True
            new_location = False  # 新位置标志为False
            terminal = True  # 终止标志为True

        # return map_local, self.latent_map, reward, terminal, complete, new_location, collision_index, finish_all_map, self.li_map, self.map_step
        return map_local, reward, terminal, complete, new_location, collision_index, finish_all_map

    def rescuer(self):
        # 救援模式
        complete = False  # 完成标志
        all_map = False  # 所有地图标志
        self.map_step += 1
        pre_position = self.robot_position.copy()  # 保存机器人位置
        #到当前操作地图的第一个free的边界
        # self.robot_position = self.frontier(self.op_map, self.map_size, self.t)
        boundaries = self.detect_map_boundaries(self.op_map)
        num_sequences, sequence_lengths, boundary_list, center_list = self.count_sequences(boundaries, self.robot_position)
        if len(boundary_list) == 0:
            self.robot_position = pre_position.copy()  # 设置机器人位置为之前的位置
        else:
            _, self.robot_position = self.cal_latent_s(self.latent_map, sequence_lengths, boundary_list, center_list)
        self.op_map = self.inverse_sensor(self.robot_position, self.sensor_range, self.op_map, self.global_map)  # 更新操作地图
        step_map = self.robot_model(self.robot_position, self.robot_size, self.t, self.op_map)  # 获取机器人模型
        map_local = self.local_map(self.robot_position, step_map, self.map_size, self.sensor_range + self.local_size)  # 获取局部地图
        boundaries = self.detect_map_boundaries(self.op_map)
        num_sequences, sequence_lengths, boundary_list, center_list = self.count_sequences(boundaries, self.robot_position)
        self.latent_map, _ = self.cal_latent_s(self.latent_map, sequence_lengths, boundary_list, center_list)

        if self.plot:
            # 如果绘图标志为True
            path = self.astar_path(self.op_map, pre_position.tolist(), self.robot_position.tolist())  # 获取A*路径
            self.x2frontier = ma.append(self.x2frontier, ma.masked)  # 添加掩码到边界x坐标列表
            self.y2frontier = ma.append(self.y2frontier, ma.masked)  # 添加掩码到边界y坐标列表
            self.x2frontier = ma.append(self.x2frontier, path[1, :])  # 添加路径x坐标到边界x坐标列表
            self.y2frontier = ma.append(self.y2frontier, path[0, :])  # 添加路径y坐标到边界y坐标列表
            self.xPoint = ma.append(self.xPoint, ma.masked)  # 添加掩码到绘图x坐标列表
            self.yPoint = ma.append(self.yPoint, ma.masked)  # 添加掩码到绘图y坐标列表
            self.xPoint = ma.append(self.xPoint, self.robot_position[0])  # 添加机器人x坐标到绘图x坐标列表
            self.yPoint = ma.append(self.yPoint, self.robot_position[1])  # 添加机器人y坐标到绘图y坐标列表
            self.plot_env()  # 绘制环境

        if np.size(np.where(self.op_map == 255))/np.size(np.where(self.global_map == 255)) > self.finish_percent:
            # 如果探索完成百分比大于规定的完成百分比
            self.li_map += 1  # 地图索引加1
            if self.li_map == self.map_number:
                # 如果地图索引等于地图数量
                self.li_map = 0  # 地图索引归零
                all_map = True  # 所有地图标志为True
            self.__init__(self.li_map, self.mode, self.plot)  # 重新初始化
            complete = True  # 完成标志为True
            new_location = False  # 新位置标志为False
            terminal = True  # 终止标志为True
        #return map_local, self.latent_map, complete, all_map, self.li_map, self.map_step
        return map_local, complete, all_map

    def take_action(self, action_index, robot_position):
        # 执行动作
        # move_action = self.action_space[action_index, :]  # 获取动作
        move_action = action_index
        robot_position[0] = np.round(robot_position[0] + move_action[0])  # 更新机器人x坐标
        robot_position[1] = np.round(robot_position[1] + move_action[1])  # 更新机器人y坐标

    def map_setup(self, location):
        # 地图设置

        # 当将图像读取为灰度图像时，输出的数据将是一个二维数组，表示灰度图像的像素值。这些像素值通常在 0 到 1 之间，表示图像的亮度，其中 0 表示黑色，1 表示白色。
        global_map = (io.imread(location, 1) * 255).astype(int)  # 读取地图并转换为整数类型
        # 地图共含有array([127, 194, 208])三个数值，其中208表示机器人的位置，194表示障碍物，127表示未知区域
        # 机器人大小为16*16，因此为208的像素点共有256个
        robot_location = np.nonzero(global_map == 208)  # 找到机器人的位置
        # 取机器人位置的中值表示机器人
        robot_location = np.array([np.array(robot_location)[1, 127], np.array(robot_location)[0, 127]])  # 转换机器人位置格式
        # 255表示free space，1表示obstacle
        global_map = (global_map > 150)  # 阈值处理地图
        global_map = global_map * 254 + 1  # 更新地图
        return global_map, robot_location  # 返回地图和机器人位置

    def map_points(self, map_glo):
        # 地图点
        map_x = map_glo.shape[1]  # 地图x方向大小
        map_y = map_glo.shape[0]  # 地图y方向大小
        x = np.linspace(0, map_x - 1, map_x)  # 在x方向上均匀生成点
        y = np.linspace(0, map_y - 1, map_y)  # 在y方向上均匀生成点
        t1, t2 = np.meshgrid(x, y)  # 生成网格
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T  # 将网格展平成点
        return points  # 返回地图点

    # self.local_size = 40  # 局部地图尺寸
    def local_map(self, robot_location, map_glo, map_size, local_size):
        # 获取以robot_location为中心的局部地图，大小为80*80
        minX = robot_location[0] - local_size
        maxX = robot_location[0] + local_size
        minY = robot_location[1] - local_size
        maxY = robot_location[1] + local_size

        # 如果超出边界则向边界内平移
        if minX < 0:
            maxX = abs(minX) + maxX
            minX = 0
        if maxX > map_size[1]:
            minX = minX - (maxX - map_size[1])
            maxX = map_size[1]
        if minY < 0:
            maxY = abs(minY) + maxY
            minY = 0
        if maxY > map_size[0]:
            minY = minY - (maxY - map_size[0])
            maxY = map_size[0]

        map_loc = map_glo[int(minY):int(maxY)][:, int(minX):int(maxX)]
        return map_loc

    # 得到当前op_map中所有的free points
    def free_points(self, op_map):
        index = np.where(op_map == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def get_extrinsic_reward(self, old_op_map, op_map, coll_index):
        if not coll_index:
            reward = float(np.size(np.where(op_map == 255)) - np.size(np.where(old_op_map == 255))) / 14000
            if reward > 1:
                reward = 1
        else:
            reward = -1
        return reward
    
    def get_intrinsic_reward(self, latent_map, robot_position):
        latent_map[latent_map == 255] = 127
        reward = latent_map[int(robot_position[1]), int(robot_position[0])]/255 * 0.1
        return reward

    def nearest_free(self, tree, point):
        pts = np.atleast_2d(point)
        index = tuple(tree.query(pts)[1])
        nearest = tree.data[index]
        return nearest

    # step_map = self.robot_model(self.robot_position, self.robot_size, self.t, self.op_map)
    # self.robot_size = 6  # 机器人大小
    # self.local_size = 40  # 局部地图尺寸
    # points为栅格地图的所有点
    def robot_model(self, position, robot_size, points, map_glo):
        map_copy = map_glo.copy()
        robot_points = self.range_search(position, robot_size, points)
        for i in range(0, robot_points.shape[0]):
            rob_loc = np.int32(robot_points[i, :])
            rob_loc = np.flipud(rob_loc)
            map_copy[tuple(rob_loc)] = 76
        map_with_robot = map_copy
        return map_with_robot

    # range_search函数用于搜索机器人周围的点
    # position: 机器人的位置
    # robot_size: 机器人的大小
    # points: 地图上的所有点
    def range_search(self, position, robot_size, points):
        # 计算机器人周围的点的距离
        nvar = position.shape[0]
        r2 = robot_size ** 2
        s = 0
        for d in range(0, nvar):
            s += (points[:, d] - position[d]) ** 2
        # 获取距离在机器人大小范围内的点
        idx = np.nonzero(s <= r2)
        idx = np.asarray(idx).ravel()
        inrange_points = points[idx, :]
        return inrange_points

    def collision_check(self, start_point, end_point, map_size, map_glo):
        x0, y0 = start_point.round()
        x1, y1 = end_point.round()
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        coll_points = np.ones((1, 2), np.uint8) * -1

        while 0 <= x < map_size[1] and 0 <= y < map_size[0]:
            k = map_glo.item(y, x)
            if k == 1:
                coll_points.itemset((0, 0), x)
                coll_points.itemset((0, 1), y)
                break

            if x == end_point[0] and y == end_point[1]:
                break

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        if np.sum(coll_points) == -2:
            coll_index = False
        else:
            coll_index = True

        return coll_points, coll_index

    def inverse_sensor(self, robot_position, sensor_range, op_map, map_glo):
        op_map = inverse_sensor_model(int(robot_position[0]), int(robot_position[1]), int(sensor_range), op_map, map_glo)
        return op_map

 
    def detect_map_boundaries(self, op_map):
        boundaries = []
        visited = set()

        def bfs(x, y):
            queue = deque([(x, y)])
            visited.add((x, y))
            while queue:
                cx, cy = queue.popleft()
                neighbors = [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]
                for nx, ny in neighbors:
                    if 0 <= nx < op_map.shape[0] and 0 <= ny < op_map.shape[1]:
                        if op_map[nx, ny] == 255:
                            boundaries.append((cx, cy))
                        elif op_map[nx, ny] != 1 and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))

        for i in range(op_map.shape[0]):
            for j in range(op_map.shape[1]):
                if op_map[i, j] == 127 and (i, j) not in visited:
                    bfs(i, j)
        boundaries = list(set(boundaries))
        return boundaries
    
    def dfs(self, x, y, boundaries, visited):
        visited.add((x, y))
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        length = 1
        boundary = [(x, y)]  # 初始将当前点加入边界列表
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) in boundaries and (nx, ny) not in visited:
                sub_length, sub_boundary = self.dfs(nx, ny, boundaries, visited)
                length += sub_length
                boundary.extend(sub_boundary)
        return length, boundary

    def get_center(self, coords):
        # 按照 x 坐标进行排序
        sorted_coords = sorted(coords, key=lambda x: x[0])

        # 计算坐标点数量
        num_coords = len(sorted_coords)

        # 如果坐标点数量为奇数，则返回中间的坐标点
        if num_coords % 2 == 1:
            middle_index = num_coords // 2
            return sorted_coords[middle_index]

        # 如果坐标点数量为偶数，则返回中间两个坐标点之一
        else:
            middle_index = num_coords // 2
            return sorted_coords[middle_index - 1]  # 返回中间两个坐标点之一

    '''高斯距离(指数距离,或者标准化欧式距离)，需要先计算欧式距离'''
    def exponent_distance(self, p, q, y):
        '''
        :param p: list
        :param q: list
        :param y: 缩放因子
        :return:
        '''
        d = distance.euclidean(p, q)
        d1 = np.exp(-y * d)
        return d1

    def count_sequences(self, coordinates, robot_location):
        visited = set()
        sequences = []
        boundary_list = []
        center_list = []  # 存储中心点坐标列表
        for x, y in coordinates:
            if (x, y) not in visited:
                length, boundary = self.dfs(x, y, coordinates, visited)
                sequences.append(length)
                boundary_list.append(boundary)
                center_list.append(self.get_center(boundary))
        dist = []
        for index in center_list:
            dist.append(self.exponent_distance(index, robot_location, 0.005))
        sequences = [x * y for x, y in zip(sequences, dist)]
        return len(sequences), sequences, boundary_list, center_list
    

    def cal_latent_s(self, latent_map, sequences, boundary_list, center_list):

        sorted_index = np.argsort(sequences)
        new_location = np.zeros(2, dtype=int)

        if len(sorted_index) == 0:
            return np.ones(self.global_map.shape) * 127, np.array([0, 0])

        # 计算最小值和最大值
        min_a = np.min(sorted_index)
        max_a = np.max(sorted_index)

        # 定义目标范围
        target_min = 128
        target_max = 254
        if len(sorted_index) == 1:
            normalized_a = [target_max]
        else:
            # 对数组 a 进行线性变换，将其归一化到指定范围内
            normalized_a = np.around(((sorted_index - min_a) / (max_a - min_a)) * (target_max - target_min) + target_min)
        latent_s = latent_map.copy()
        for k, sub_index in enumerate(sorted_index):
            for i,j in boundary_list[sub_index]:
                latent_s[i, j] = normalized_a[sub_index]
        # np.savetxt('latent_s.csv', latent_s, delimiter=',', fmt='%d')
        new_location[0], new_location[1] = int(center_list[sorted_index[-1]][1]), int(center_list[sorted_index[-1]][0])
        return latent_s,new_location

    def unique_rows(self, a):
        a = np.ascontiguousarray(a)  # 将输入数组转换为连续的数组，这是为了确保数据在内存中是连续存储的，这对于后续的视图操作是必要的。
        unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))  # 将数组a的每一行视为一个单独的元素，并找出其中的唯一元素。这实际上是在找出唯一的行。
        result = unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))  # 将unique_a的视图转换回原来的数据类型，并重新塑形为原始数组的形状。
        result = result[~np.isnan(result).any(axis=1)]  # 删除包含NaN值的行。
        return result 

    def astar_path(self, weights, start, goal, allow_diagonal=True):
        temp_start = [start[1], start[0]]
        temp_goal = [goal[1], goal[0]]
        temp_weight = (weights < 150) * 254 + 1
        # For the heuristic to be valid, each move must cost at least 1.
        if temp_weight.min(axis=None) < 1.:
            raise ValueError("Minimum cost to move must be 1, but got %f" % (
                temp_weight.min(axis=None)))
        # Ensure start is within bounds.
        if (temp_start[0] < 0 or temp_start[0] >= temp_weight.shape[0] or
                temp_start[1] < 0 or temp_start[1] >= temp_weight.shape[1]):
            raise ValueError("Start lies outside grid.")
        # Ensure goal is within bounds.
        if (temp_goal[0] < 0 or temp_goal[0] >= temp_weight.shape[0] or
                temp_goal[1] < 0 or temp_goal[1] >= temp_weight.shape[1]):
            print(goal)
            raise ValueError("Goal of lies outside grid.")

        height, width = temp_weight.shape
        start_idx = np.ravel_multi_index(temp_start, (height, width))
        goal_idx = np.ravel_multi_index(temp_goal, (height, width))

        path = astar(
            temp_weight.flatten(), height, width, start_idx, goal_idx, allow_diagonal,
        )
        return path

    def plot_env(self):
        plt.cla()
        plt.imshow(self.op_map, cmap='gray')
        plt.axis((0, self.map_size[1], self.map_size[0], 0))
        plt.plot(self.xPoint, self.yPoint, 'b', linewidth=2)
        plt.plot(self.x2frontier, self.y2frontier, 'r', linewidth=2)
        plt.plot(self.robot_position[0], self.robot_position[1], 'mo', markersize=8)
        plt.plot(self.xPoint[0], self.yPoint[0], 'co', markersize=8)
        plt.pause(0.05)
