import os
import logging

logger = logging.getLogger('training log')
logger.setLevel(logging.INFO)
if __name__ == '__main__':
    def A1():
        while 1:
            try:
                global A
                A = float(input('请输入A，只保留数字，支持小数'))
                break
            except:
                print('输入数据类型有误,请重新输入')


    while 1:
        A1()
        print(A1)
