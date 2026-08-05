def AstraNet_run(**training_config):
    import pickle
    import subprocess
    import sys, time, os
    os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
    import numpy as np
    import json
    # import visualkeras
    from Mail import sendmailqq
    from keras.optimizers import Adam
    from keras.utils import plot_model
    # from tensorflow.keras.utils import plot_model
    from sklearn.metrics import r2_score
    from AstraNet import AstraNet
    from utils import compute
    check_start_stamp = time.time()
    exp_count = training_config['exp_count']
    # K.clear_session()  # 重置GPU显存
    # prepPorto()  # prep数据预处理
    configs = json.dumps(training_config)
    subprocess.run(["python", "prepPorto.py", configs])  # 按照超参数，预处理一下，获得矩阵

    dir = os.getcwd()
    savedStdout = sys.stdout  # 保存默认输出流

    CUDA_VISIBLE_DEVICES = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定只是用第0块GPU

    lr = float(training_config['learning_rate'])
    batch_size = int(training_config['batch_size'])
    nb_residual_unit = int(training_config['nb_residual_unit'])  # number of residual units,L
    days_test = int(training_config['days_for_test'])
    nb_epoch = int(training_config['nb_epoch'])  # number of epoch at training stage
    finetune_epoch = int(training_config['finetuning_epoch'])
    consider_external_info = bool(int(training_config['consider_external_info']))

    len_closeness = int(training_config['len_closeness'])  # length of closeness dependent sequence
    len_period = int(training_config['len_period'])  # length of peroid dependent sequence
    len_trend = int(training_config['len_trend'])  # length of trend dependent sequence

    slot_num = int(24 * 3600 / 600)  # 一天被划分成这么多个slot

    def mape_adjust(A, B):
        A[A == 0] = 1e-10
        MAPE = np.mean(np.abs((A - B) / A))
        return MAPE

    cs = [True]  # 是否ConSider external
    for p in cs:
        consider_external_info = p
        filename = 'TaxiPT_c%d_p%d_t%d' % (len_closeness, len_period, len_trend)
        hyperparams_name = 'TaxiPT_ep%d_C%d_P%d_T%d_L%d_bs%d_lr%.1e' % (nb_epoch, len_closeness, len_period, len_trend, nb_residual_unit, batch_size, lr)
        T = slot_num  # number of time intervals in one day
        nb_flow = 2  # there are two types of flows: new-flow and end-flow
        days_test = days_test  # 预测时长4星期28天divide data into two subsets: Train & Test, of which the test set is the last 4 weeks
        len_test = T * days_test
        map_height, map_width = 14, 30  # grid size

        if consider_external_info:
            filename = filename + '_ext'
            hyperparams_name = hyperparams_name + '_ext'
        else:
            filename = filename + '_noext'
            hyperparams_name = hyperparams_name + '_noext'

        filename = os.path.join(dir, "data", 'TaxiPorto', filename)
        expdir = os.path.join(dir, "experiment", 'TaxiPT')
        fname_param = os.path.join(expdir, hyperparams_name + '/best.h5')

        # 检查预处理文件是否存在
        print('75 in main,checking prepfile exists:是否存在', os.path.exists(filename))
        assert os.path.exists(filename), "预处理文件不存在.prepfile does not exist."  # 判断 x 是否大于等于 0

        print('73 in train,filename:', filename)
        print('74 in train,fname_param:', fname_param)

        f = open(filename, 'rb')
        X_train = pickle.load(f)
        Y_train = pickle.load(f)
        # print(81, X_train)
        X_test = pickle.load(f)
        Y_test = pickle.load(f)
        mmn = pickle.load(f)
        external_dim = pickle.load(f)
        print(86, 'extenral dim=', external_dim)

        timestamp_train = pickle.load(f)
        timestamp_test = pickle.load(f)
        mask = pickle.load(f)
        print(88, 'all shapes', len(X_test), len(Y_test), )  # len(mmn),len(external_dim))

        Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
        Y_test = mmn.inverse_transform(Y_test)

        c_conf = (len_closeness, nb_flow, map_height, map_width) if len_closeness > 0 else None
        p_conf = (len_period, nb_flow, map_height, map_width) if len_period > 0 else None
        t_conf = (len_trend, nb_flow, map_height, map_width) if len_trend > 0 else None

        # 定义模型
        model = AstraNet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf, external_dim=external_dim,
                         nb_residual_unit=nb_residual_unit)

    # 绘制模型结构图(3D)
    # visualkeras.layered_view(model)
    print(106, "model summary replacing visualkeras...hidden.")

    # model.summary()
    def paint_history0():
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        print(127, 'in paint_history,paiting.')
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))

        # 图0，pre_loss连接fine_loss,pre_val_loss连接fine_val_loss
        train_color = 'tab:blue'
        val_color = 'tab:red'
        ax[0].plot(pre_history.history['loss'] + [None] * len(fine_history.history['loss']),
                   label='pre_loss', color=train_color)  # pre_loss
        ax[0].plot([None] * len(pre_history.history['loss']) + fine_history.history['loss'],
                   label='fine_loss', linestyle=":", color=train_color)  # fine_loss
        ax[0].plot([len(pre_history.history['loss']) - 1, len(pre_history.history['loss'])],
                   [pre_history.history['loss'][-1], fine_history.history['loss'][0]], color=train_color)  # 连接线joint
        ax[0].tick_params(axis='y', labelcolor=train_color)

        ax[0] = ax[0].twinx()
        ax[0].plot(pre_history.history['val_loss'] + [None] * len(fine_history.history['val_loss']),
                   label='pre_val_loss', color=val_color)
        ax[0].plot([None] * len(pre_history.history['val_loss']) + fine_history.history['val_loss'],
                   label='fine_val_loss', linestyle=":", color=val_color)
        ax[0].plot([len(pre_history.history['val_loss']) - 1, len(pre_history.history['val_loss'])],
                   [pre_history.history['val_loss'][-1], fine_history.history['val_loss'][0]],
                   color=val_color)  # 连接线joint
        ax[0].tick_params(axis='y', labelcolor=val_color)

        # 图0格式
        ax[0].set_title('Loss曲线：MAE')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss(MAE)')
        ax[0].grid()

        ax[0].set_title('Loss曲线：MAE')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss(MAE)')
        ax[0].grid()

        # 获取第一个子图的legend
        handles, labels = ax[0].get_legend_handles_labels()
        # 获取第二个子图的legend
        handles_add, labels_add = ax[0].get_legend_handles_labels()
        # 将第二个子图的legend添加到第一个子图的legend中
        handles += handles_add
        labels += labels_add
        # 绘制新的legend
        ax[0].legend(handles, labels)

        plt.tight_layout()

        # 图1，pre_mse连接fine_mse,pre_val_mse连接fine_val_mse
        ax[1].plot(pre_history.history['mse'] + [None] * len(fine_history.history['mse']), label='pre_mse',
                   color=train_color)  # pre_mse
        ax[1].plot([None] * len(pre_history.history['mse']) + fine_history.history['mse'], label='fine_mse',
                   linestyle=":", color=train_color)  # fine_mse
        ax[1].plot([len(pre_history.history['mse']) - 1, len(pre_history.history['mse'])],
                   [pre_history.history['mse'][-1], fine_history.history['mse'][0]], color=train_color)  # 连接线joint
        ax[1].tick_params(axis='y', labelcolor=train_color)

        ax[1] = ax[1].twinx()
        ax[1].plot(pre_history.history['val_mse'] + [None] * len(fine_history.history['val_mse']),
                   label='pre_val_mse',
                   color=val_color)  # pre_val_mse
        ax[1].plot([None] * len(pre_history.history['val_mse']) + fine_history.history['val_mse'],
                   label='fine_val_mse',
                   linestyle=":", color=val_color)
        ax[1].plot([len(pre_history.history['val_mse']) - 1, len(pre_history.history['val_mse'])],
                   [pre_history.history['val_mse'][-1], fine_history.history['val_mse'][0]],
                   color=val_color)  # 连接线joint
        ax[1].tick_params(axis='y', labelcolor=val_color)

        # 图1格式
        ax[1].set_title('Metric曲线：MSE')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Metric(MSE)')
        ax[1].grid()

        ax[1].set_title('Metric曲线：MSE')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Metric(MSE)')
        ax[1].grid()

        # 获取第一个子图的legend
        handles, labels = ax[1].get_legend_handles_labels()
        # 获取第二个子图的legend
        handles_add, labels_add = ax[1].get_legend_handles_labels()
        # 将第二个子图的legend添加到第一个子图的legend中
        handles += handles_add
        labels += labels_add
        # 绘制新的legend
        ax[1].legend(handles, labels)

        plt.tight_layout()
        plt.savefig(os.path.join(expdir, hyperparams_name + "/best.png"))
        plt.close()
        return

    def paint_history():
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        print(127, 'in paint_history,paiting.')
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))

        # 图0，pre_loss连接fine_loss,pre_val_loss连接fine_val_loss
        train_color = 'tab:blue'
        val_color = 'tab:red'
        ax[0].plot(pre_history.history['loss'] + [None] * len(fine_history.history['loss']),
                   label='pre_loss', color=train_color)  # pre_loss
        ax[0].plot([None] * len(pre_history.history['loss']) + fine_history.history['loss'],
                   label='fine_loss', linestyle=":", color=train_color)  # fine_loss
        ax[0].plot([len(pre_history.history['loss']) - 1, len(pre_history.history['loss'])],
                   [pre_history.history['loss'][-1], fine_history.history['loss'][0]], color=train_color)  # 连接线joint
        # ax[0].tick_params(axis='y', labelcolor=train_color)

        ax[0].plot(pre_history.history['val_loss'] + [None] * len(fine_history.history['val_loss']),
                   label='pre_val_loss', color=val_color)
        ax[0].plot([None] * len(pre_history.history['val_loss']) + fine_history.history['val_loss'],
                   label='fine_val_loss', linestyle=":", color=val_color)
        ax[0].plot([len(pre_history.history['val_loss']) - 1, len(pre_history.history['val_loss'])],
                   [pre_history.history['val_loss'][-1], fine_history.history['val_loss'][0]],
                   color=val_color)  # 连接线joint
        # ax[0].tick_params(axis='y', labelcolor=val_color)

        # 图0格式
        ax[0].set_title('Loss曲线：MAE')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss(MAE)')
        ax[0].grid()
        ax[0].legend()

        # # 获取第一个子图的legend
        # handles, labels = ax[0].get_legend_handles_labels()
        # # 获取第二个子图的legend
        # handles_add, labels_add = ax[0].get_legend_handles_labels()
        # # 将第二个子图的legend添加到第一个子图的legend中
        # handles += handles_add
        # labels += labels_add
        # # 绘制新的legend
        # ax[0].legend(handles, labels)

        plt.tight_layout()

        # 图1，pre_mse连接fine_mse,pre_val_mse连接fine_val_mse
        ax[1].plot(pre_history.history['mse'] + [None] * len(fine_history.history['mse']), label='pre_mse',
                   color=train_color)  # pre_mse
        ax[1].plot([None] * len(pre_history.history['mse']) + fine_history.history['mse'], label='fine_mse',
                   linestyle=":", color=train_color)  # fine_mse
        ax[1].plot([len(pre_history.history['mse']) - 1, len(pre_history.history['mse'])],
                   [pre_history.history['mse'][-1], fine_history.history['mse'][0]], color=train_color)  # 连接线joint
        # ax[1].tick_params(axis='y', labelcolor=train_color)

        ax[1].plot(pre_history.history['val_mse'] + [None] * len(fine_history.history['val_mse']),
                   label='pre_val_mse',
                   color=val_color)  # pre_val_mse
        ax[1].plot([None] * len(pre_history.history['val_mse']) + fine_history.history['val_mse'],
                   label='fine_val_mse',
                   linestyle=":", color=val_color)
        ax[1].plot([len(pre_history.history['val_mse']) - 1, len(pre_history.history['val_mse'])],
                   [pre_history.history['val_mse'][-1], fine_history.history['val_mse'][0]],
                   color=val_color)  # 连接线joint
        # ax[1].tick_params(axis='y', labelcolor=val_color)

        # 图1格式
        ax[1].set_title('Metric曲线：MSE')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Metric(MSE)')
        ax[1].grid(axis='both')

        # # 获取第一个子图的legend
        # handles, labels = ax[1].get_legend_handles_labels()
        # # 获取第二个子图的legend
        # handles_add, labels_add = ax[1].get_legend_handles_labels()
        # # 将第二个子图的legend添加到第一个子图的legend中
        # handles += handles_add
        # labels += labels_add
        # # 绘制新的legend
        # ax[1].legend(handles, labels)
        ax[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(expdir, hyperparams_name + "/best1.png"))
        plt.close()
        return

    # 模型结构图(2D)
    os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
    try:
        plot_model(model, to_file=os.path.join(expdir, 'PTmodel.png'), show_shapes=True,  # 显示每层的输出形状
                   show_dtype=False,  # 不显示数据类型
                   show_layer_names=True,  # 显示层名称
                   rankdir='TB',  # 从上到下排列（TB=Top to Bottom）
                   dpi=500, )
        print(295, "model structured. checking pics.")
    except:
        print("plot_model failed. skip plotting and go on...")
    from keras.callbacks import ModelCheckpoint

    if not os.path.exists(os.path.join(expdir, hyperparams_name)): os.makedirs(os.path.join(expdir, hyperparams_name))
    fname_param = os.path.join(expdir, hyperparams_name + '/best.h5')

    # 编译网络模型
    adam = Adam(learning_rate=lr)
    model.compile(loss='mae', optimizer=adam, metrics=['mse'])
    # model_checkpoint = ModelCheckpoint(fname_param, monitor='loss', verbose=1, save_best_only=True, mode='min')
    print(149, '保存于', fname_param)
    # for xxa, xxb, xxc in os.walk(""):        print(xxa, xxb, xxc)

    # 一次训练pre-training
    print('=' * 20)
    print("pre-training model...")
    start_time = time.time()
    print(309, len(X_train), len(Y_train))
    print()
    pre_history = model.fit(X_train, Y_train,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=[ModelCheckpoint(fname_param, monitor='loss', verbose=1,
                                                       save_best_only=True, mode='min')], verbose=2)  # pre-training
    end_time = time.time()
    print('cost %.f mins on training' % ((end_time - start_time) // 60))

    print(156, 'history:fine二次训练历史', pre_history.history.keys())
    print(180, 'fine的loss和metric列表', len(pre_history.history['loss']), len(pre_history.history['mse']))

    # paint_history(fig, ax, 0, 'pre')

    # pretrain性能评估，指标
    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    print(169, '一次验证:中段验证，从此加载：', fname_param)
    model.load_weights(fname_param)

    # pretrain模型在训练集上的性能评估
    pre_train_score = model.evaluate(X_train, Y_train, batch_size=int(Y_train.shape[0] // slot_num), verbose=0)
    print(196, 'pre train score', pre_train_score)
    print('训练集Loss(MAE)=%.6f, MSE=%.6f' % tuple(pre_train_score))

    # pretrain模型在测试集上的性能评估
    pre_test_score = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('测试集Loss(MAE)=%.6f, MSE=%.6f, R2=None%%' % tuple(pre_test_score))

    # 用pretrain对预测集实施预测，检验四大指标
    pre_Y_predict = model.predict(X_test, batch_size=Y_test.shape[0], verbose=0)
    pre_mse, pre_mae, pre_mape, pre_r2 = compute(Y_test, pre_Y_predict)
    print(177, 'pretrain四大指标,', hyperparams_name,
          ', mse:%.6f, mae:%.6f, mape:%.6f, R2:%.6f%%' % (pre_mse, pre_mae, pre_mape, pre_r2 * 100))

    #
    #
    # pretrain完成
    #
    #
    # 下面开始finetune
    #
    #
    # 二次训练fine tuning
    print('=' * 10)
    print("cont fine-tuning training model...")
    start_time = time.time()
    adam = Adam(learning_rate=0.1 * lr)
    k = r2_score([1, 2], [4, 5])
    model.compile(loss='mae', optimizer=adam, metrics=['mse'])  # finetune的指标设定
    model.load_weights(fname_param)
    # model_checkpoint = ModelCheckpoint(fname_param, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    fine_history = model.fit(X_train, Y_train,
                             epochs=finetune_epoch,  # fine tuning 的epoch
                             batch_size=batch_size, validation_split=.1,
                             callbacks=[ModelCheckpoint(fname_param, monitor='val_loss', verbose=1, save_best_only=True,
                                                        mode='min')], verbose=2)

    end_time = time.time()
    print('cost %.f mins on finetunining' % ((end_time - start_time) // 60))
    print(156, 'history:fine二次训练历史', fine_history.history.keys())
    print(180, 'fine的loss和metric列表', fine_history.history['loss'], fine_history.history['mse'])

    # 二次验证（验证finetune模型）
    print('=' * 10)
    print('cont evaluating ...')
    model.load_weights(fname_param)  # 从训练好的最佳模型中load weights

    # finetune模型在训练集上的性能评估
    fine_train_score = model.evaluate(X_train, Y_train, batch_size=int(Y_train.shape[0] // slot_num), verbose=0)
    print('finetune Train MAE: %.6f, MSE=%.6f, R2=None%%' % tuple(fine_train_score))

    # finetune模型在测试集上的性能评估
    fine_val_score = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('finetune Test- MAE: %.6f, MSE=%.6f, R2=None%%' % tuple(fine_val_score))

    # 画图
    paint_history()

    # 用finetune对预测集实施预测，检验四大指标
    fine_Y_predict = model.predict(X_test, batch_size=Y_test.shape[0], verbose=0)
    fine_mse, fine_mae, fine_mape, fine_r2 = compute(Y_test, fine_Y_predict)
    print(hyperparams_name + '. mse:%.6f, mae:%.6f, mape:%.6f, r2:%.6f%%' % (
        fine_mse, fine_mae, fine_mape, fine_r2 * 100))
    #
    #
    #
    #
    #
    #
    #
    #

    # return {'pretrain': (pre_train_mse, pre_train_mae, pre_train_mape,
    #                      pre_train_r2),
    #         "fine": []}  # pre_train_mse, pre_train_mae, pre_train_mape, pre_train_r2 # 返回一次检验结果

    # results = train()  # 应用train函数，完成全部训练
    # pre_train_mse, pre_train_mae, pre_train_mape, pre_train_r2 = results['pretrain']
    print('247 训练完成。两阶段训练全部指标如下：')
    print('pre模型on训练集', pre_train_score)
    print('pre模型on测试集', pre_test_score)
    print('pre模型四大指标', pre_mse, pre_mae, pre_mape, pre_r2)

    print('fine模型on训练集', fine_train_score)
    print('fine模型on测试集', fine_val_score)
    print('fine模型四大指标', fine_mse, fine_mae, fine_mape, fine_r2)

    #
    #
    #
    #
    #

    # for iday in range(days_test):  # 预测多少天，previously设定了1周7天。
    iday = 0  # 预测一天，试验
    pred = model.predict_on_batch(X_test)[iday * slot_num:(iday + 1) * slot_num]
    groundtruth = Y_test[iday * slot_num:(iday + 1) * slot_num]
    print('完整预测结果的尺寸', model.predict_on_batch(X_test).shape, iday, 'pred shape:', pred.shape,
          'gr_truth shape:', groundtruth.shape)
    if not os.path.exists(r'result'):          os.makedirs(r'result')
    if not os.path.exists(r'result/{}'.format(hyperparams_name)): os.mkdir('result/{}'.format(hyperparams_name))
    np.save('result/{}/pred_day'.format(hyperparams_name) + str(iday).zfill(2) + '.npy', pred)
    np.save('result/{}/groundtruth_day'.format(hyperparams_name) + str(iday).zfill(2) + '.npy', groundtruth)

    # file.close()  # 关闭output输出流文件
    sys.stdout = savedStdout  # 恢复默认输出流
    print(188, 'hyperparams:', hyperparams_name)
    print(189, 'fname:', fname_param)
    total_time_consumed = time.time() - check_start_stamp

    sendmailqq('782568799@qq.com', check_start_stamp, hyperparams_name=hyperparams_name,
               nb_epoch=nb_epoch, finetune_epoch=finetune_epoch, exp_count=exp_count,
               mse=fine_mse, mae=fine_mae, mape=fine_mape, r2=fine_r2,
               title='AstraNet[{}]{}:Epoch={}+{}_MSE={:.6f}_MAE={:.6f}_MAPE={:.6f}%_r2={:.6f}%'.format(
                   str(exp_count).zfill(5),
                   hyperparams_name, nb_epoch, finetune_epoch,
                   fine_mse, fine_mae, 100 * fine_mape, 100 * fine_r2),
               extra_content='{:.0f}h {:.0f}min {:.2f}s'.format(total_time_consumed // 3600,
                                                                (total_time_consumed // 60) % 60,
                                                                total_time_consumed % 60))

    print('program ends here.')
    """
    GMR:mse:10.487609,         mae:0.853193,          mape:0.101506
    STR:MSE= 3.353926420211792 MAE= 2.411409616470337 MAPE= 0.017955992370843887
    """
