# 2.6.7	《Mail.py》发送邮件通知代码
import os
import time
from email.header import Header
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from smtplib import SMTP_SSL
def sendmailqq(receive_mail, check_start_stamp, **kwargs):
    print('before sendmailqq, check keys:', kwargs.keys())
    hyperparams_name = kwargs.get('hyperparams_name', None)
    nb_epoch = kwargs.get('nb_epoch', None)
    finetune_epoch = kwargs.get('finetune_epoch', None)
    exp_count = kwargs.get('exp_count', None)
    mse = kwargs.get('mse', None)
    mae = kwargs.get('mae', None)
    mape = kwargs.get('mape', None)
    r2 = kwargs.get('r2', None)
    title = kwargs.get('title', None)
    Aggregator = kwargs.get('Aggregator', None)
    extra_content = kwargs.get('extra_content', None)
    print('inside sendmail,', hyperparams_name, nb_epoch, finetune_epoch, mse, mae, mape, r2, title, extra_content)
    check_end_stamp = time.time()
    total_seconds = check_end_stamp - check_start_stamp
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60
    send_usr = '782568799@qq.com'  # 发件人
    send_pwd =   r"pwogffqqirxabfaf"
    receive = receive_mail  # '782568799@qq.com'  # 接收者
    content0 = '发送于{}.......【综合损失指标{}】<p>10月训练已完成{}</p>训练模型结果：Epoch={}+{}, MSE={:.6f}, MAE={:.6f}, MAPE={:.6f}, R2={:.6f}'.format(
        time.ctime(), Aggregator, hyperparams_name, nb_epoch, finetune_epoch, mse, mae, 100 * mape, 100 * r2)
    # content 内容设置

    # 在构建 html_img 之前添加
    image1_exists = os.path.exists(r'Experiment/TaxiPT/PTmodel.png')
    image3_exists = os.path.exists(os.path.join(r'experiment\taxiPT', hyperparams_name + '/best1.png'))

    # 动态构建 img 标签
    img1_tag = '<img src="cid:image1">' if image1_exists else ''
    img3_tag = '<img src="cid:image3">' if image3_exists else ''

    html_img = f'<p>{content0}<br>{img1_tag}{img3_tag}</br><br></br></p><p>{extra_content}</p>'
    # html_img = f'<p>{content0}<br><img src="cid:image1"><img src="cid:image3"></br><br></br></p><p>{extra_content}</p>'  # html格式添加图片
    email_server = 'smtp.qq.com'
    msg = MIMEMultipart()  # 构建主体
    msg['Subject'] = Header(title, 'utf8')  # 邮件主题
    msg['From'] = send_usr  # 发件人
    msg['To'] = Header('Training Logger', 'utf8')  # 收件人--这里是昵称
    # msg.attach(MIMEText(content,'html','utf-8'))  # 构建邮件正文,不能多次构造
    attchment = MIMEApplication(open(r'Experiment/TaxiPT/PTmodel.png', 'rb').read())  # 文件
    attchment.add_header('Content-Disposition', 'attachment', filename=r'Experiment/TaxiPT/PTmodel.png')
    msg.attach(attchment)  # 添加附件到邮件
    # attchment2 = MIMEApplication(
    #     open(os.path.join('experiment/taxiPT/', hyperparams_name + '/best.h5'), 'rb').read())  # 文件
    # attchment2.add_header('Content-Disposition', 'attachment',
    #                       filename=os.path.join(r'experiment/taxiPT/', hyperparams_name + '/best.h5'))
    # msg.attach(attchment2)  # 添加附件到邮件
    attchment3 = MIMEApplication(
        open(os.path.join(r'experiment\taxiPT', hyperparams_name + '/best1.png'), 'rb').read())  # 文件
    attchment3.add_header('Content-Disposition', 'attachment',
                          filename=os.path.join(r'experiment\taxiPT', hyperparams_name + '/best1.png'))
    msg.attach(attchment3)  # 添加附件到邮件
    f = open(r'Experiment/TaxiPT/PTmodel.png', 'rb')  # 打开图片
    msgimage = MIMEImage(f.read())
    f.close()
    msgimage.add_header('Content-ID', '<image1>')  # 设置图片
    msg.attach(msgimage)
    # f1 = open(os.path.join(r'experiment\taxiPT', hyperparams_name + '/pretrain.png'), 'rb')  # 打开图片
    # msgimage1 = MIMEImage(f1.read())
    # f1.close()
    # msgimage1.add_header('Content-ID', '<image2>')  # 设置图片
    # msg.attach(msgimage1)
    f2 = open(os.path.join(r'experiment\taxiPT', hyperparams_name + '/best1.png'), 'rb')  # 打开图片
    msgimage2 = MIMEImage(f2.read())
    f2.close()
    msgimage2.add_header('Content-ID', '<image3>')  # 设置图片
    msg.attach(msgimage2)
    msg.attach(MIMEText(html_img, 'html', 'utf-8'))  # 添加到邮件正文
    # 添加功能：保存本地csv
    import pandas as pd
    if not os.path.exists("training_process.csv"):
        df_read = pd.DataFrame(
            {"exp_count": "", "finish_time": "", "hyperparams_name": "", "nb_epoch": "", "finetune_epoch": "",
             "mse": "", "mae": "",
             "mape": "", "r2": ""}, index=['test'])
    else:
        df_read = pd.read_csv('training_process.csv')
    df = pd.DataFrame({"exp_count": exp_count, "finish_time": time.ctime(), "hyperparams_name": hyperparams_name,
                       "nb_epoch": nb_epoch,
                       "finetune_epoch": finetune_epoch, "mse": mse, "mae": mae, "mape": mape, "r2": r2},
                      index=[exp_count])
    df = pd.concat([df_read, df])
    df.to_csv("training_process.csv", index=False)
    try:
        smtp = SMTP_SSL(email_server)  # 指定邮箱服务器
        smtp.ehlo(email_server)  # 部分邮箱需要
        smtp.login(send_usr, send_pwd)  # 登录邮箱
        smtp.sendmail(send_usr, receive, msg.as_string())  # 分别是发件人、收件人、格式
        smtp.quit()  # 结束服务
        print(receive_mail, '邮件发送成功,mailflag已经改成False!', time.ctime())
    except Exception as E:
        print('发送失败', E)
        return 'sent'
def sendmail163(receive_mail, check_start_stamp, **kwargs):
    print('before sendmail163, check keys:', kwargs.keys())
    hyperparams_name = kwargs.get('hyperparams_name', None)
    nb_epoch = kwargs.get('nb_epoch', None)
    finetune_epoch = kwargs.get('finetune_epoch', None)
    exp_count = kwargs.get('exp_count', None)
    mse = kwargs.get('mse', None)
    mae = kwargs.get('mae', None)
    mape = kwargs.get('mape', None)
    r2 = kwargs.get('r2', None)
    title = kwargs.get('title', None)
    extra_content = kwargs.get('extra_content', None)
    print('inside sendmail,', hyperparams_name, nb_epoch, finetune_epoch, mse, mae, mape, r2, title, extra_content)
    check_end_stamp = time.time()
    total_seconds = check_end_stamp - check_start_stamp
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60
    send_usr = 'midy2001@163.com'  # 发件人
    send_pwd = r"BNwRP5VYvvgFUsf8"
    receive = receive_mail  # '782568799@qq.com'  # 接收者
    content0 = '发送于{}<p>10月训练已完成{}</p>训练模型结果：Epoch={}+{}, MSE={:.6f}, MAE={:.6f}, MAPE={:.6f}, R2={:.6f}'.format(
        time.ctime(), hyperparams_name, nb_epoch, finetune_epoch, mse, mae, 100 * mape, 100 * r2)
    # content 内容设置
    html_img = f'<p>{content0}<br><img src="cid:image1"><img src="cid:image3"></br><br></br></p><p>{extra_content}</p>'  # html格式添加图片
    email_server = 'smtp.163.com'
    msg = MIMEMultipart()  # 构建主体
    msg['Subject'] = Header(title, 'utf8')  # 邮件主题
    msg['From'] = send_usr  # 发件人
    msg['To'] = Header('Training Logger', 'utf8')  # 收件人--这里是昵称
    # msg.attach(MIMEText(content,'html','utf-8'))  # 构建邮件正文,不能多次构造
    attchment = MIMEApplication(open(r'Experiment/TaxiPT/PTmodel.png', 'rb').read())  # 文件
    attchment.add_header('Content-Disposition', 'attachment', filename=r'Experiment/TaxiPT/PTmodel.png')
    msg.attach(attchment)  # 添加附件到邮件
    # attchment2 = MIMEApplication(
    #     open(os.path.join('experiment/taxiPT/', hyperparams_name + '/best.h5'), 'rb').read())  # 文件
    # attchment2.add_header('Content-Disposition', 'attachment',
    #                       filename=os.path.join(r'experiment/taxiPT/', hyperparams_name + '/best.h5'))
    # msg.attach(attchment2)  # 添加附件到邮件
    attchment3 = MIMEApplication(
        open(os.path.join(r'experiment\taxiPT', hyperparams_name + '/best1.png'), 'rb').read())  # 文件
    attchment3.add_header('Content-Disposition', 'attachment',
                          filename=os.path.join(r'experiment\taxiPT', hyperparams_name + '/best1.png'))
    msg.attach(attchment3)  # 添加附件到邮件
    f = open(r'Experiment/TaxiPT/PTmodel.png', 'rb')  # 打开图片
    msgimage = MIMEImage(f.read())
    f.close()
    msgimage.add_header('Content-ID', '<image1>')  # 设置图片
    msg.attach(msgimage)
    # f1 = open(os.path.join(r'experiment\taxiPT', hyperparams_name + '/pretrain.png'), 'rb')  # 打开图片
    # msgimage1 = MIMEImage(f1.read())
    # f1.close()
    # msgimage1.add_header('Content-ID', '<image2>')  # 设置图片
    # msg.attach(msgimage1)
    f2 = open(os.path.join(r'experiment\taxiPT', hyperparams_name + '/best1.png'), 'rb')  # 打开图片
    msgimage2 = MIMEImage(f2.read())
    f2.close()
    msgimage2.add_header('Content-ID', '<image3>')  # 设置图片
    msg.attach(msgimage2)
    msg.attach(MIMEText(html_img, 'html', 'utf-8'))  # 添加到邮件正文
    # 添加功能：保存本地csv
    import pandas as pd
    if not os.path.exists("training_process.csv"):
        df_read = pd.DataFrame(
            {"exp_count": "", "finish_time": "", "hyperparams_name": "", "nb_epoch": "", "finetune_epoch": "",
             "mse": "", "mae": "",
             "mape": "", "r2": ""}, index=['test'])
    else:
        df_read = pd.read_csv('training_process.csv')
    df = pd.DataFrame({"exp_count": exp_count, "finish_time": time.ctime(), "hyperparams_name": hyperparams_name,
                       "nb_epoch": nb_epoch,
                       "finetune_epoch": finetune_epoch, "mse": mse, "mae": mae, "mape": mape, "r2": r2},
                      index=[exp_count])
    df = pd.concat([df_read, df])
    df.to_csv("training_process.csv", index=False)
    try:
        smtp = SMTP_SSL(email_server)  # 指定邮箱服务器
        smtp.ehlo(email_server)  # 部分邮箱需要
        smtp.login(send_usr, send_pwd)  # 登录邮箱
        smtp.sendmail(send_usr, receive, msg.as_string())  # 分别是发件人、收件人、格式
        smtp.quit()  # 结束服务
        print(receive_mail, '邮件发送成功,mailflag已经改成False!', time.ctime())
    except Exception as E:
        print('发送失败', E)
        return 'sent'
# sendmail('782568799@qq.com', 1, "TaxiPT_ep50_C3_P1_T1_L4_bs128_lr1.0e-04_ext", 0, 0, 0, 0, "try标题")
# sendmail163('782568799@qq.com', 1, "TaxiPT_ep50_C3_P1_T1_L4_bs128_lr1.0e-04_ext", 0, 0, 0, 0, "try")
def sendwarning(receive_mail, **kwargs):
    print('before sendmail-warning, keyw args:', kwargs.keys())
    E = kwargs.get('E', None)
    title = "【Error】训练中止。报错：{}".format(E)
    print('before sendmailwarning, check error:', E)
    send_usr = 'midy2001@163.com'  # 发件人
    send_pwd = r"BNwRP5VYvvgFUsf8"  # 授权码，邮箱设置
    receive = receive_mail
    content0 = "【Warning】.\n程序报错，训练终止。报错：{}".format(E)
    email_server = 'smtp.163.com'
    msg = MIMEMultipart()  # 构建主体
    msg['Subject'] = Header(title, 'utf8')  # 邮件主题
    msg['From'] = send_usr  # 发件人
    msg['To'] = Header('Error Report', 'utf8')  # 收件人--这里是昵称
    try:
        smtp = SMTP_SSL(email_server)  # 指定邮箱服务器
        smtp.ehlo(email_server)  # 部分邮箱需要
        smtp.login(send_usr, send_pwd)  # 登录邮箱
        smtp.sendmail(send_usr, receive, msg.as_string())  # 分别是发件人、收件人、格式
        smtp.quit()  # 结束服务
        print(receive_mail, '报错通知已发送。Error Notice Sent.', time.ctime())
    except Exception as E:
        print('Error报错邮件发送失败', E)
        return 'sent'

