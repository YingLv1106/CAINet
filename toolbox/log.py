"""
    日志记录
    同时输出到屏幕和文件
    可以通过日志等级，将训练最后得到的结果发送到邮箱，参考下面example

"""

import logging
import os
import sys
import time


def get_logger(logdir):

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logname = f'run-{time.strftime("%Y-%m-%d-%H-%M")}.log'
    log_file = os.path.join(logdir, logname)

    # create log
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    # Formatter 设置日志输出格式
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # StreamHandler 日志输出1 -> 到控制台
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler 日志输出2 -> 保存到文件log_file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# # example 输出到邮箱
# from logging.handlers import SMTPHandler
#
# logger = logging.getLogger('train')
# logger.setLevel(logging.INFO)
#
# SMTP_handler = SMTPHandler(
#     mailhost=('smtp.163.com', 25),
#     fromaddr='xxx163emailxxx@163.com',
#     toaddrs=['xxxqqemailxxx@qq.com', 'or other emails you want to send'],
#     subject='send title',
#     credentials=('fromaddr email', 'fromaddr passwd')
# )
#
# formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# SMTP_handler.setFormatter(formatter)
# SMTP_handler.setLevel(logging.WARNING)  # 设置等级为warning, logger.warning('infos')将会把重要结果信息输出到邮箱
# logger.addHandler(SMTP_handler)
#
# logging.warning('information need to be send to email. the final results_old or errors')

