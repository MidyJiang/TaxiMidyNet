# from datetime import datetime
import datetime
import time

import h5py
import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.metrics import r2_score


# 正确的struct0！考虑时区传入
def struct0(stamp, offset):
    timezone = datetime.timezone(datetime.timedelta(hours=offset))
    dt = datetime.datetime.fromtimestamp(stamp, timezone)
    timestr = dt.strftime('%Y-%m-%d %H:%M:%S')
    return timestr


class MinMaxNormalization(object):  # 定义类：归一化到[-1,1]区间
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):  # 获取数组中的min和max
        self._min = X.min()  # 定义最小值
        self._max = X.max()  # 定义最大值
        print("min:", self._min, "max:", self._max)

    def transform(self, X):  # 转换，输入x，输出归一化后在[-1,1]上的x(须已fit()获得了min和max)
        X = 1. * (X - self._min) / (self._max - self._min)  # 归一化
        X = X * 2. - 1.  # 将归一化后的x的范围转换为[-1,1]
        return X  # 归一化后的x处于[0,1]，x*2表示将数据扩大2倍，将值域变成[0,2]，再减去1，表示将数据的中心点平移到-1，从而得到[-1,1]区间的数据。

    def fit_transform(self, X):  # 输入x，输出归一化后在[-1,1]的x
        self.fit(X)  # 获取min和max
        return self.transform(X)  # 返回归一化到[-1,1]区间的数据

    def inverse_transform(self, X):  # 输入归一化到[-1,1]区间的数据，返回原始数据
        X = (X + 1.) / 2.  # 逆仿射
        X = 1. * X * (self._max - self._min) + self._min  # 逆归一化
        return X  # 返回原始数据


def mse(y_true, y_pred):  # 求MSE的函数
    return K.mean(K.square(y_pred - y_true))  # MSE=(∑(预测值-真实值)²)/样本数量


def rmse(y_true, y_pred):  # 求RMSE的函数
    return mse(y_true, y_pred) ** 0.5  # RMSE=√(MSE)


def mae(y_true, y_pred):  # 求MAE的函数
    return K.mean(K.abs(y_pred - y_true))  # MAE= ∑(|预测值-真实值|)/样本数量


def compute(y_true, y_pred):  # 输入真实y和预测y，返回y的MSE,y的MAE，y的MAPE（三元组）
    print('utils 62,in compute, y_true&pred.shape,', type(y_true), y_true.shape, y_true.flatten().shape)
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    y_mse = np.mean(np.square(y_true - y_pred))  # 求y的MSE
    # y_rmse = y_mse ** 0.5  # 求y的RMSE
    y_mae = np.mean(np.abs(y_true - y_pred))  # 求y的MAE：绝对值之和的均值
    # idx = (y_true > 1e-6).nonzero()  # 找出y_true中值大于0.000001的索引
    # y_mape = np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx]))  # 求MAPE，即(真实值-预测值)/真实值的绝对值之和的均值
    # 新方法mape_adjust:修正后的MAPE
    A, B = y_true, y_pred
    mape_adjust = np.mean(np.abs(
        (np.where(A == 0, np.ones_like(A), A) - np.where(B == 0, np.ones_like(B), B)) / np.where(A == 0,
                                                                                                 np.ones_like(A), A)))

    return y_mse, y_mae, mape_adjust, r2


def remove_incomplete_days(data, timestamps, T=48):  # 删除数据缺失的日期。传入数据集，时间戳，每天slot数目T。返回完整天的数据和时间戳
    # 删除一天中stamp少于48的日期
    days = []  # 存放完整天的时间戳available days: some day only contain some seqs
    days_incomplete = []  # 存放不完整天的时间戳
    i = 0
    while i < len(timestamps):  # 遍历时间戳。若当前时间戳以1结尾，且i+T-1<len(timestamps)，即当天48个时间戳都存在，则将其存入days；
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:  # 若不满足条件，则将其存入days_incomplete
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)  # 输出不完整天时间戳
    days = set(days)  # 将days转换为集合，去重
    idx = []
    for i, t in enumerate(timestamps):  # 从时间戳中获取完整天的时间戳，存入idx
        if t[:8] in days:
            idx.append(i)

    data = data[idx]  # 根据idx，从data和timestamps中筛选出完整天的数据
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps  # 返回筛选后的数据和时间戳


def load_stdata(fname):  # 从fname.h5加载数据
    # print('fname:', fname)
    f = h5py.File(fname, 'r')
    data = f['data'][()]
    timestamps = f['slot'][()]
    f.close()
    return data, timestamps


def string2timestamp(strings, T=48 * 3):  # 日期字符串转时间戳pd.Timestamp类型，输入、输出均为列表list
    """
    输入strings: list, eg. ['2017080912','2017080913']
    输出return: list, eg. [Timestamp('2017-08-09 05:30:00'), Timestamp('2017-08-09 06:00:00')]
    """
    start, end = 1372636800, 1404172800
    slots_table = dict(zip(np.arange(len(np.arange((end - start) / 600 + 1) * 600 + start)),  # 时隙600s=10min
                           np.arange((end - start) / 600 + 1) * 600 + start))  # 序号：stamp

    timestamps = []
    for t in strings:
        timestamps.append(pd.to_datetime(struct0(t, 1)))
    return timestamps  # Timestamp类型的时间点


class STMatrix(object):  # 用于实现时空序列数据的格式化,存储和管理

    def __init__(self, data, timestamps, T=144, CheckComplete=True):
        """
        data：时间序列数据，类型为numpy array格式
        timestamps：时间序列，类型为字符串格式
        T：代表时间序列的时间窗口，默认值为48
        CheckComplete：默认值为True，指示是否检查时间序列数据的完整性
        """
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)  # 在类的定义开头，先使用断言语句检查传入的参数是否符合要求，从而保证类的正确使用
        self.data = data
        # 这里注意：输入的data矩阵0维度=inflow=D，1维度=outflow=O!
        self.data_1 = data[:, 0, :, :]  # inflow矩阵,D
        self.data_2 = data[:, 1, :, :]  # outflow矩阵,O
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)  # timestamp类型的列表
        if CheckComplete:
            self.check_complete()  # 检查数据中每天的完整性

        self.make_index()  # 生成index

    def make_index(self):
        """
        make_index()方法是用来创建一个字典，该字典用于将时间戳映射到其相应的索引。
        该方法会使用pd_timestamps列表中的时间戳，并将其放入字典中。
        一旦字典创建完成，就可以使用时间戳查找其对应的索引。
        """
        self.get_index = dict()  # 创建这个字典get_index:dict
        for i, ts in enumerate(self.pd_timestamps):  # i是索引，ts是每个timestamps
            self.get_index[ts] = i  # 字典结构：{timestamps时间点:索引号}

    def check_complete(self):  # 检查时间戳是否完整
        """
        此函数的作用是检查时间戳是否完整，即检查是否有时间戳缺失。
        通过计算时间间隔，比较两个时间戳的差值，若不相等，则表示有缺失的时间戳。
        """
        missing_timestamps = []  # 初始化缺失时间戳列表
        offset = pd.DateOffset(minutes=24 * 60 // self.T)  # 计算时间间隔
        pd_timestamps = self.pd_timestamps  # 获取时间戳
        i = 1  # 初始化计数器
        while i < len(pd_timestamps):  # 遍历时间戳
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:  # 比较两个时间戳是否相等
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))  # 若不相等，则添加到缺失时间戳列表
            i += 1  # 计数器自增
        for v in missing_timestamps:  # 遍历缺失时间戳列表
            print("175 in utils, missing date:", v)  # 打印缺失时间戳
        assert len(missing_timestamps) == 0  # 断言缺失时间戳数量为零

    def get_matrix(self, timestamp):  # 用timestamp从字典get_index中获取一个三维矩阵:时间固定，[O/D,行，列]
        return self.data[self.get_index[timestamp]]  # 通过get_index函数来获取时间戳的索引，然后使用该索引从data中获取三维矩阵

    def get_matrix_1(self, timestamp):  # in_flow  #  根据给定的时间戳timestamp，从inflow矩阵data_1中获取二维矩阵并恢复成三维
        ori_matrix = self.data_1[self.get_index[timestamp]]  # 获取指定时间戳的inflow二维矩阵
        new_matrix = ori_matrix[np.newaxis, :]  # 在指定时间戳的inflow二维矩阵前增加一个维度，新维度长度为1，转成三维矩阵
        # print("new_matrix shape:", new_matrix.shape)  # 打印新矩阵的形状#(1,32,32)
        return new_matrix  # 返回新获得的三维矩阵

    def get_matrix_2(self, timestamp):  # out_flow#同理，获得给定时间戳的outflow的三维矩阵
        ori_matrix = self.data_2[self.get_index[timestamp]]
        new_matrix = ori_matrix[np.newaxis, :]
        # print("new_matrix shape:",new_matrix.shape) #(1, 32, 32)
        return new_matrix

    def save(self, fname):
        pass

    def check_it(self, depends):
        """
        检查depends中的所有时间是否都在时间列表中。如果有至少一个时间不在事件列表中，返回一个布尔值False。均在则True。
        :param depends: 一串时间点的序列
        :return: 一个布尔值，表示时间是否都在时间列表中
        """
        for d in depends:  # 遍历depends中的元素
            if d not in self.get_index.keys():  # 判断元素d是否在{时间:索引号}字典get_index的key中(即时间)
                return False  # 但凡有一个d不是时间列表中的时间点，返回False，函数结束
        return True  # 如果全部都在，返回True

    def create_dataset_3D(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        """
        创建三重依赖关系的数据集，并设置参数
        :param len_closeness:
        :param len_trend:
        :param TrendInterval:
        :param len_period:周期性
        :param PeriodInterval:
        :return:
        """
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)  # 按照T的大小，确定每个时间单位的间隔
        XC = []  # 存放距离特征的list
        XP = []  # 存放周期特征的list
        XT = []  # 存放趋势特征的list
        Y = []  # 存放标签特征的list
        timestamps_Y = []  # 存放标签特征的时间戳list
        depends = [range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend + 1)]]

        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)  # 获取第一个建模的数据
        while i < len(self.pd_timestamps):  # 不断获取建模的数据
            Flag = True  # 设置建模标识
            for depend in depends:  # 对每一个相关性参数进行判断
                if Flag is False:  # 若建模标识为假，跳出当前循环
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])  # 检查参数是否获取成功，是否可以建模

            if Flag is False:  # 若建模标识为假，跳出当前循环
                i += 1
                continue

            # closeness
            c_1_depends = list(depends[0])  # in_flow,获取closeness中in_flow的参数
            c_1_depends.sort(reverse=True)  # 参数倒序排序
            # print('----- c_1_depends:',c_1_depends)

            c_2_depends = list(depends[0])  # out_flow#获取closeness中out_flow的参数
            c_2_depends.sort(reverse=True)  # 参数倒序排序
            # print('----- c_2_depends:',c_2_depends)

            x_c_1 = [self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame) for j in
                     c_1_depends]  # [(1,32,32),(1,32,32),(1,32,32)] in_flow#获取in_flow的矩阵
            x_c_2 = [self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame) for j in
                     c_2_depends]  # [(1,32,32),(1,32,32),(1,32,32)] out_flow#获取out_flow的矩阵

            x_c_1_all = np.vstack(x_c_1)  # x_c_1_all.shape  (3, 32, 32)#拼接in_flow的矩阵
            x_c_2_all = np.vstack(x_c_2)  # x_c_1_all.shape  (3, 32, 32)#拼接out_flow的矩阵

            x_c_1_new = x_c_1_all[np.newaxis, :]  # (1, 3, 32, 32)#改变in_flow矩阵的维度
            x_c_2_new = x_c_2_all[np.newaxis, :]  # (1, 3, 32, 32)#改变out_flow矩阵的维度

            x_c = np.vstack([x_c_1_new, x_c_2_new])  # (2, 3, 32, 32)#拼接in_flow和out_flow的矩阵

            # period
            p_depends = list(depends[1])  # 获取period的参数
            if (len(p_depends) > 0):  # 若period的参数不为空
                p_depends.sort(reverse=True)  # 参数倒序排序
                # print('----- p_depends:',p_depends)

                x_p_1 = [self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame) for j in p_depends]  # 获取in_flow的矩阵
                x_p_2 = [self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame) for j in
                         p_depends]  # 获取out_flow的矩阵

                x_p_1_all = np.vstack(x_p_1)  # [(3,32,32),(3,32,32),...]#拼接in_flow的矩阵
                x_p_2_all = np.vstack(x_p_2)  # [(3,32,32),(3,32,32),...]#拼接out_flow的矩阵

                x_p_1_new = x_p_1_all[np.newaxis, :]  # (1, 3, 32, 32)#改变in_flow矩阵的维度
                x_p_2_new = x_p_2_all[np.newaxis, :]  # (1, 3, 32, 32)#改变out_flow矩阵的维度

                x_p = np.vstack([x_p_1_new, x_p_2_new])  # (2, 3, 32, 32)#拼接in_flow和out_flow的矩阵
            else:
                x_p = np.zeros((2, 0, 14, 30))

            # trend
            t_depends = list(depends[2])  # 获取trend的参数
            if (len(t_depends) > 0):  # 若trend的参数不为空
                t_depends.sort(reverse=True)  # 参数倒序排序

                x_t_1 = [self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame) for j in t_depends]  # 获取in_flow的矩阵
                x_t_2 = [self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame) for j in t_depends]  # 获取outflow的矩阵

                x_t_1_all = np.vstack(x_t_1)  # [(3,32,32),(3,32,32),...]#拼接in_flow的矩阵
                x_t_2_all = np.vstack(x_t_2)  # [(3,32,32),(3,32,32),...]#拼接outflow的矩阵

                x_t_1_new = x_t_1_all[np.newaxis, :]  # (1, 3, 32, 32))#改变in_flow矩阵的维度
                x_t_2_new = x_t_2_all[np.newaxis, :]  # (1, 3, 32, 32))#改变out_flow矩阵的维度

                x_t = np.vstack([x_t_1_new, x_t_2_new])  # (2, 3, 32, 32)

            y = self.get_matrix(self.pd_timestamps[i])

            # if len_closeness > 0:   XC.append(x_c)
            # if len_period > 0:      XP.append(x_p)
            # if len_trend > 0:       XT.append(x_t)
            XC.append(x_c)
            XP.append(x_p)
            XT.append(x_t)
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1

        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print("3D matrix - XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y


def timestamp2vec(timestamps):
    """
    输入一组时间戳序列，输出这组时间戳序列的独热编码与工作日标识矢量矩阵。
    :param timestamps: 时间戳序列
    :return: 形为(len(timestamps),8)的矩阵。第二维的8列中，前7列是星期数据的独热编码，第8列表示【是/否工作日】,weekday=1,weekend=0.
    """
    # tm_wday 表示提取一周的第几天，范围[0,6]，Monday=0, Sunday=6
    vec = [time.strptime(struct0(t, 1)[:10], '%Y-%m-%d').tm_wday for t in timestamps]  # 提取timestamps是星期几
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)
