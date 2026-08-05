import json
import os
import pickle
import sys
from copy import copy

from utils import *

# # 单独运行调试prep
# config_file = os.path.join(os.getcwd(), 'config', 'pt.conf')
# config = configparser.ConfigParser()
# config.read(config_file)
# training_config = config['Training']

# 2.batches传入超参数
configs = sys.argv[1]
training_config = json.loads(configs)
print('prep 11:training_config', training_config)
start, end = 1372636800, 1404172800
slot_num = int(24 * 3600 / 600)  # 一天被划分成这么多个slot(T=144)
dir = os.getcwd()


def load_holiday(timeslots, fname):
    """
    加载节假日特征函数，输入timeslots，fname
    :param timeslots:
    :param fname:
    :return:
    """
    f = open(os.path.join(fname, "PT_Holiday.txt"), 'r')  # 打开文件fname下的PT_Holiday.txt文件
    holidays = f.readlines()  # 读取文件的所有行
    holidays = set([h.strip() for h in holidays])  # 去掉换行符，将所有行放入一个集合中
    H = np.zeros(len(timeslots))  # 初始化一个大小为timeslots的0的数组
    for i, slot in enumerate(timeslots):  # 遍历timeslots
        if struct0(slot, 1)[:10] in holidays:  # 如果从timeslots中截取的前8个字符(yyyy-mm-dd)在holidays中
            H[i] = 1  # 将相应的位置.置为1
    # print(H.sum())  # 打印H的和
    # print(timeslots[H==1])  # 打印H中值为1的timeslots
    return H[:, None]  # 把H从一维行向量转成一维列向量，返回之


def load_meteorol(timeslots, fname):
    """
    加载气象数据
    :param timeslots:
    :param fname: 文件名
    :return:
    """
    f = h5py.File(os.path.join(fname, 'PT_METEOROLOGY.h5'), 'r')
    Timeslot = f['date'][()]
    WindSpeed = f['WindSpeed'][()]
    Weather = f['Weather'][()]
    Temperature = f['Temperature'][()]
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # WindSpeed
    WR = []  # Weather
    TE = []  # Temperature
    # print(46, timeslots, '\n\n')
    for slot in timeslots:
        # print(47, M)
        # print(60, slot, start)
        predicted_id = M[(slot - start) // 600]
        cur_id = predicted_id - 1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    print("shape: ", WS.shape, WR.shape, TE.shape)

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

    # print('meger shape:', merge_data.shape)
    return merge_data


def create_mask(city, city_dict):  # 定义函数create_mask()，参数是city和city_dict
    if city == 'Porto':  # 判断参数city的值，如果为NY
        shape = (14, 30)  # shape的值设置为(32, 32)
    sum_inflow = np.zeros(shape=shape)  # 创建sum_inflow数组，元素均为0，形状为shape
    sum_outflow = np.zeros(shape=shape)  # 创建sum_outflow数组，元素均为0，形状为shape
    for i in city_dict.keys():  # 遍历city_dict中的所有键
        if 'Inflow' in i:  # 如果键中含有Inflow
            sum_inflow += city_dict[i]  # 将city_dict[i]的值加到sum_inflow中
        elif 'Outflow' in i:  # 否则，如果键中含有Outflow
            sum_outflow += city_dict[i]  # 将city_dict[i]的值加到sum_outflow中
    sum_outflow = np.array([0 if x == 0 else 1 for x in sum_outflow.flatten()]).reshape(
        shape)  # 将sum_outflow扁平化，值为0的元素全部替换为1，再将该数组的形状设置为shape
    sum_inflow = np.array([0 if x == 0 else 1 for x in sum_inflow.flatten()]).reshape(
        shape)  # 将sum_inflow扁平化，值为0的元素全部替换为1，再将该数组的形状设置为shape
    return np.array([sum_outflow, sum_inflow])  # 返回sum_outflow和sum_inflow数组的列表


def create_dict(data, timestamps):  # 定义create_dict函数，传入data和timestamps参数，返回字典{时间O/D:值}
    ny_dict = {}  # 创建ny_dict字典
    for index in range(len(data)):  # 遍历data列表
        ny_dict[str(timestamps[index]) + '_Inflow'] = data[index][0].tolist()  # 将timestamps和data列表中的数据组合添加到ny_dict字典中
        ny_dict[str(timestamps[index]) + '_Outflow'] = data[index][1].tolist()  # 将timestamps和data列表中的数据组合添加到ny_dict字典中
    return ny_dict  # 返回ny_dict字典


def load_data_PT(T=slot_num, nb_flow=2, len_closeness=None, len_period=None, len_trend=None, len_test=None,
                 meta_data=True, holiday_data=True, meteorol_data=True):
    assert (len_closeness + len_period + len_trend > 0)
    dir = os.getcwd()
    # load data

    data_all = []
    timestamps_all = list()

    for month in [10]:  # 取2013-10一个月实现
        fname = os.path.join(dir, 'data', 'TaxiPorto', 'OctoberODmat{}.h5'.format(month))
        print("119 in prep,OD filename: ", fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have full slot_num(144) timestamps
        # data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        # create mask
        ny_dict = create_dict(data, timestamps)
        mask = create_mask('Porto', ny_dict)

        data_all.append(data)
        timestamps_all.append(timestamps)
        # print("\n")

    # minmax_scale
    data_train = np.vstack(copy(data_all))[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is
        # a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        # _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(
        #     len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
        # print("create dataset gsn")
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset_3D(len_closeness=len_closeness, len_period=len_period,
                                                                len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    meta_feature = []
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_Y)
        meta_feature.append(time_feature)
    if holiday_data:
        # load holiday
        holiday_feature = load_holiday(timestamps_Y, os.path.join(dir, 'data', 'TaxiPorto'))
        meta_feature.append(holiday_feature)
    if meteorol_data:
        # load meteorol data
        meteorol_feature = load_meteorol(timestamps_Y, os.path.join(dir, 'data', 'TaxiPorto'))
        meta_feature.append(meteorol_feature)

    meta_feature = np.hstack(meta_feature) if len(
        meta_feature) > 0 else np.asarray(meta_feature)
    metadata_dim = meta_feature.shape[1] if len(
        meta_feature.shape) > 1 else None
    # if metadata_dim < 1:
    #     metadata_dim = None
    if meta_data and holiday_data and meteorol_data:
        print('time feature:', time_feature.shape, 'holiday feature:', holiday_feature.shape,
              'meteorol feature: ', meteorol_feature.shape, 'mete feature: ', meta_feature.shape)

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
          "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]

    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:            X_test.append(X_)

    if metadata_dim is not None:
        meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)

    print('X train shape:')
    for _X in X_train:        print(_X.shape, end=',')
    # print()

    print('X test shape:')
    for _X in X_test:        print(_X.shape, end=',')
    # print()
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test, mask


T = slot_num  # number of time intervals in one day
nb_residual_unit = int(training_config['nb_residual_unit'])  # number of residual units,L
days_test = int(training_config['days_for_test'])
len_closeness = int(training_config['len_closeness'])  # length of closeness dependent sequence
len_period = int(training_config['len_period'])  # length of peroid dependent sequence
len_trend = int(training_config['len_trend'])  # length of trend dependent sequence
nb_flow = 2  # there are two types of flows: new-flow and end-flow
len_test = T * days_test  # test的slot数目
map_height, map_width = 14, 30  # grid size
nb_epoch = int(training_config['nb_epoch'])  # number of epoch at training stage
finetun_epoch = int(training_config['finetuning_epoch'])
consider_external_info = bool(int(training_config['consider_external_info']))
if consider_external_info:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, mask = \
        load_data_PT(T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend,
                     len_test=len_test, meta_data=consider_external_info, holiday_data=consider_external_info,
                     meteorol_data=consider_external_info)

    dir = os.getcwd()
    filename = os.path.join(dir, 'data', 'TaxiPorto', 'TaxiPT_c%d_p%d_t%d_ext' % (len_closeness, len_period, len_trend))
    print('244 in prep,filename:', filename)
    f = open(filename, 'wb')
    pickle.dump(X_train, f)
    pickle.dump(Y_train, f)
    pickle.dump(X_test, f)
    pickle.dump(Y_test, f)
    pickle.dump(mmn, f)
    pickle.dump(external_dim, f)
    pickle.dump(timestamp_train, f)
    pickle.dump(timestamp_test, f)
    pickle.dump(mask, f)
    f.close()

else:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test, mask = \
        load_data_PT(T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend,
                     len_test=len_test, meta_data=consider_external_info, holiday_data=consider_external_info,
                     meteorol_data=consider_external_info)

    dir = os.getcwd()
    filename = os.path.join(dir, 'data', 'TaxiPorto',
                            'TaxiPT_c%d_p%d_t%d_noext' % (len_closeness, len_period, len_trend))
    print(265, 'filename:', filename)
    f = open(filename, 'wb')
    pickle.dump(X_train, f)
    pickle.dump(Y_train, f)
    pickle.dump(X_test, f)
    pickle.dump(Y_test, f)
    pickle.dump(mmn, f)
    pickle.dump(external_dim, f)
    pickle.dump(timestamp_train, f)
    pickle.dump(timestamp_test, f)
    f.close()

print('279 in prepPorto.py, 已经跑完prep！目录保存于：', filename)
