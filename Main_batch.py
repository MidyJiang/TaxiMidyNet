# 2.6.6	《Main_batch.py》主函数批量训练代码
import traceback
from Mail import sendwarning
from AstraNet_train import *
import warnings

warnings.filterwarnings("ignore")
exp_count = 0


# learning_rate = 0.0002
# batch_size = 64
# nb_residual_unit = 4
# consider_external_info = 1
# nb_epoch = 200
# finetuning_epoch = 50
# len_closeness = 3
# len_period = 1
# len_trend = 1
# days_for_test = 7
def new():
    exp_count = 50
    try:
        for learning_rate in [1e-4]:
            for batch_size in [64]:
                for len_closeness in [1]:  # C
                    for len_period in [1]:  # P
                        for len_trend in [1]:  # T
                            for nb_residual_unit in [1]:  # L
                                for nb_epoch in [5]:
                                    try:
                                        days_for_test = 7
                                        consider_external_info = 0
                                        finetuning_epoch = int(nb_epoch * .2)
                                        # K.clear_session()  # 重置GPU显存
                                        exp_count += 1
                                        print('\n\n每个单独的循环:above all,exp_count=', exp_count,
                                              'TaxiPT_ep%d_C%d_P%d_T%d_L%d_bs%d_lr%.1e' % (
                                                  nb_epoch, len_closeness, len_period, len_trend,
                                                  nb_residual_unit, batch_size, learning_rate))
                                        if exp_count < -5:
                                            continue
                                        else:
                                            AstraNet_run(exp_count=exp_count,
                                                         learning_rate=learning_rate,
                                                         batch_size=batch_size,
                                                         nb_residual_unit=nb_residual_unit,
                                                         consider_external_info=consider_external_info,
                                                         nb_epoch=nb_epoch,
                                                         finetuning_epoch=finetuning_epoch,
                                                         len_closeness=len_closeness,
                                                         len_period=len_period,
                                                         len_trend=len_trend,
                                                         days_for_test=days_for_test)
                                    except Exception as E:
                                        traceback.print_exc()
                                        sendwarning("782568799@qq.com", E="【{}】{}".format(exp_count, E))
                                        continue
    except Exception as E:
        print('51 in topmain总main中报错跳出。', 'Error intercept Occurred.', E)
        traceback.print_exc()
        sendwarning("782568799@qq.com", E="【{}】{}".format(exp_count, E))


new()
