#-*-coding: utf-8-*-
from typing import List

from pylab import *
import os
import copy
import logging
import datetime
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 日志记录类，初始化日志记录模组
class LOG(object):
    def __init__(self):
        # 获取logger对象,LogRecord
        self.logger = logging.getLogger("LogRecord")
        # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
        self.logger.setLevel(level=logging.DEBUG)

    def LogInitialize(self,name=""):
        """日志头初始化"""
        # 获取当前时间
        curr_time = datetime.datetime.now()
        print(curr_time.year, curr_time.month, curr_time.day)

        # 格式化日期字符串，建议使用补零格式 (例如 20230505 而不是 202355)
        # 也可以保持你原来的 %s%s%s 格式
        year = int(curr_time.year)
        month = int(curr_time.month)
        day = int(curr_time.day)

        # 定义日志目录和文件路径
        log_dir = "./log"
        log_name = os.path.join(log_dir, name+"：%s%s%s.log" % (year, month, day))

        # 检查日志目录是否存在，不存在则创建
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # ==========================================
        # 核心修改：如果日志文件已存在，则删除它
        # ==========================================
        if os.path.exists(log_name):
            try:
                os.remove(log_name)
                print(f"Old log file '{log_name}' removed.")
            except OSError as e:
                print(f"Error removing log file: {e}")

        # 获取文件日志句柄并设置日志级别
        # 这里的 mode='a' (append) 是默认值，但因为我们刚才删除了文件，所以相当于新建
        # 也可以显式设置 mode='w' (write) 来覆盖，这样就不需要上面的 os.remove 了
        # 但为了完全符合你的"删除重建"逻辑，保留 os.remove 是最直观的
        handler = logging.FileHandler(log_name, mode='w')
        handler.setLevel(logging.INFO)

        # 生成并设置文件日志格式
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)

        # 获取流句柄并设置日志级别
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)

        # 为logger对象添加句柄
        # 建议先清空旧的 handlers，防止重复添加导致日志重复打印
        self.logger.handlers = []

        self.logger.addHandler(handler)
        self.logger.addHandler(console)

        return None

    def LogRecord(self, str_record, bPrint = True):
        """记录日志"""
        # 是否输出到控制台
        if bPrint:
            print(str_record)

        # 记录日志
        self.logger.info(str_record)


# 通用功能函数类
class CommonFun(object):
    @staticmethod
    def ReadConfig(configPath='config.txt'):
        configDict = {}
        file = open(configPath, 'r', encoding='utf-8')
        while True:
            getText = file.readline()
            if getText == '':
                break
            res = getText.replace(';', ' ').split(' ')
            if len(res) >= 3:
                if res[2] == 'False' or res[2] == 'True':
                    configDict[res[0]] = bool(res[2])
                elif '[' in res[2] and ']' in res[2]:
                    res[2] = res[2].replace('[', '')
                    res[2] = res[2].replace(']', '')
                    res[2] = res[2].replace('\n', '')
                    res[2] = res[2].strip()
                    nums = res[2].split(',')
                    result = []
                    for i, num in enumerate(nums):
                        if '.' in res[2] or 'e' in num:
                            result.append(float(num))
                        else:
                            result.append(int(num))
                    configDict[res[0]] = result
                elif '.' in res[2] or 'e' in res[2]:
                    configDict[res[0]] = float(res[2])
                else:
                    configDict[res[0]] = int(res[2])
        file.close()
        return configDict


    @staticmethod
    def Time2Second(Time):
        """将系统时间转换成秒计时"""
        return Time.hour*3600 + Time.minute * 60 + Time.second

    @staticmethod
    def ZoommingNormalization(input, region=None):
        """将数据缩放到[0,1]范围内"""
        res = np.zeros(len(input))
        if not region:
            region = max(input)-min(input)
        if region == 0: region = 1
        for dataIdx in range(len(input)):
            res[dataIdx] = (input[dataIdx] - min(input)) / region
        return res


    @staticmethod
    def Continues2Discrete(actionLen, splitNums=4):
        """
        将连续动作转换成离散动作数组
        返回转换后的离散数组长度和数组本身
        """
        resultList = []
        currList = np.linspace(0, 1, splitNums)
        if actionLen == 1: return currList.shape[0], currList[:, np.newaxis]
        _, tempResList = CommonFun.Continues2Discrete(actionLen - 1, splitNums)
        for idx in range(splitNums):
            temp = np.full(tempResList.shape[0], currList[idx])
            temp = np.insert(tempResList, 0, temp, axis=1)
            if idx == 0:
                resultList = copy.copy(temp)
            else:
                resultList = np.vstack((resultList, temp))
        return resultList.shape[0], resultList

    @staticmethod
    def ContinuesChoice(totalLen, choiceLen):
        """
        将连续动作转换成离散选择动作数组
        返回转换后的离散数组长度和数组本身
        """
        if totalLen <= choiceLen:
            return 1, np.full((1, totalLen), 1)
        if choiceLen == 0:
            return 1, np.full((1, totalLen), 0)
        resultList = [[]]
        # 以 0 开头的递归结果
        if totalLen > 1:
            sh, tempResList = CommonFun.ContinuesChoice(totalLen - 1, choiceLen)
            temp = np.full(tempResList.shape[0], 0)
            temp = np.insert(tempResList, 0, temp, axis=1)
            resultList = copy.deepcopy(temp)
        # resultList = np.vstack((resultList, temp))
        # 以 1 开头的递归结果
        if totalLen > 1 and choiceLen >= 1:
            sh, tempResList = CommonFun.ContinuesChoice(totalLen - 1, choiceLen - 1)
            temp = np.full(tempResList.shape[0], 1)
            temp = np.insert(tempResList, 0, temp, axis=1)
            resultList = np.vstack((resultList, temp))

        return resultList.shape[0], resultList


# 数据处理类，用于加载文件数据、检查文件目录、保存数据、绘图等
class DataProcess(object):
    """数据处理类，包含数据保存、数据读取、绘折线图保存等"""
    def __init__(self):
        mpl.rcParams['font.sans-serif'] = ['SimHei']

    @staticmethod
    def text_save(filename, data):  # filename为写入txt文件的路径，data为要写入数据列表.
        """保存列表文件"""
        file = open(filename, 'w')
        for i in range(len(data)):
            s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
            s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
            file.write(s)
        file.close()
        print("保存文件成功")

    @staticmethod
    def CheckAndBuildDirections(str_build_direction):
        """检查文件夹路径是否存在，若不存在则创建目标文件夹"""
        str_build_only_direction = str_build_direction.split('/')
        direction_length = len(str_build_only_direction) - 1
        for index in range(0, direction_length):
            str_temp1 = str_build_only_direction[0]

            if index > 0:
                for str_index in range(1, index + 1):
                    str_temp1 = str_temp1 + '/' + str_build_only_direction[str_index]

            if not os.path.exists(str_temp1):
                os.makedirs(str_temp1)

    @staticmethod
    def ReadText2List(filename):
        """输入文件目录，返回数据列表"""
        file = open(filename, 'r')
        returnList = []
        while True:
            getText = file.readline()
            if getText == '':
                break
            returnList.append(float(getText))
        file.close()
        return returnList

    @staticmethod
    def DrawBoxesPlot(plt_scores:List[List[int]], Labels, savePath="fig.png", figTitle=None, yLabel="AoI"):
        fig = plt.figure()
        ax = plt.subplot()
        fig.add_subplot(111)
        linesColors = ['red', 'blue', 'darkmagenta', 'green', 'yellow', 'lawngreen', 'darkred']
        if figTitle: plt.title(figTitle)
        bplot = plt.boxplot(plt_scores, meanline=True, showmeans=False, medianprops={'color': 'red'},
                            meanprops={'color': 'blue'}, showfliers=False)
        ax.set_xticklabels(Labels, fontsize = 7)
        for idx in range(len(bplot['boxes'])):
            bplot['boxes'][idx].set(color=linesColors[idx])
        plt.ylabel(yLabel)
        plt.grid(True)
        plt.savefig(savePath, dpi=500)
        del fig

    @staticmethod
    def DrawSingleLine(plt_scores, savePath="fig.png", figTitle=None, xLabel="Episode", yLabel="Reward"):
        """画一条线并保存"""
        fig = plt.figure()
        fig.add_subplot(111)
        if figTitle: plt.title(figTitle)
        plt.plot(np.arange(len(plt_scores)), plt_scores)    #  plt.plot(x, y)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(True)
        plt.savefig(savePath, dpi=500)
        del fig

    @staticmethod
    def DrawGapMeanLine(plt_scores, mean_gap=10, savePath="fig.png", figTitle=None, xLabel="Episode", yLabel="Reward"):
        """输入数据画间隔均值折线图"""
        fig = plt.figure()
        fig.add_subplot(111)
        # 求间隔均值
        gap_mean_scores = []
        temp, doubles = 0, 0
        for index in range(0, len(plt_scores)):
            temp += plt_scores[index]
            if (index + 1) % mean_gap == 0:
                doubles += 1
                gap_mean_scores.append(temp / (doubles * mean_gap))
                #temp = 0

        if figTitle: plt.title(figTitle)
        plt.plot(np.arange(len(gap_mean_scores)), gap_mean_scores)    #  plt.plot(x, y)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(True)
        plt.savefig(savePath, dpi=500)
        del fig, gap_mean_scores, temp

    @staticmethod
    def DrawLinesWithLabels(targetList, Labels, savePath="fig.png", bMarker=False,
                      figTitle=None, xLabel="Episode", yLabel="Reward"):
        """画多条线，每条线的数据和标签以多维列表的形式传入，但数量需≤8"""
        fig = plt.figure()
        fig.add_subplot(111)
        if figTitle: plt.title(figTitle)
        makers = ['*', 'o', '^', 'x', '+', '<', '>', 'p']
        linesColors = ['yellow', 'blue', 'darkmagenta', 'green', 'red', 'yellow', 'lawngreen', 'darkred']
        if len(Labels)==0:
            for index in range(0, len(targetList)):
                if not bMarker:
                    plt.plot(np.arange(len(targetList[index])), targetList[index], color=linesColors[index])
                else:
                    plt.plot(np.arange(len(targetList[index])), targetList[index], maker=makers[index], color=linesColors[index])
        else:
            for index in range(0, len(targetList)):
                if not bMarker:
                    plt.plot(np.arange(len(targetList[index])), targetList[index], color=linesColors[index], label=Labels[index])
                else:
                    plt.plot(np.arange(len(targetList[index])), targetList[index], maker=makers[index],
                             color=linesColors[index], label=Labels[index])
                plt.legend()
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(True)
        plt.savefig(savePath, dpi=500)
        del fig

    @staticmethod
    def DrawGapMeanLinesWithLabels(targetList, Labels, mean_gap=10, regionalMean=False,
                                   savePath="fig.png", figTitle=None, xLabel="Episode", yLabel="Reward"):
        """
        功能：输入数据画间隔均值折线图
        形参：mean_gap：均值间隔长度，默认为10
                  regionalMean：布尔值，默认为False。是否为区域均值
        """
        fig = plt.figure()
        fig.add_subplot(111)
        gap_mean_scores = []
        linesColors = ['red', 'blue', 'darkmagenta', 'green', 'yellow', 'lawngreen', 'darkred']
        for targeIdx in range(0, len(targetList)):
            temp, doubles = 0, 0
            gap_mean_scores.append([])
            for meanIdx in range(0, len(targetList[targeIdx])):
                temp += targetList[targeIdx][meanIdx]
                if (meanIdx + 1) % mean_gap == 0:
                    if not regionalMean:
                        # 全局均值
                        doubles += 1
                        gap_mean_scores[targeIdx].append(temp / (doubles * mean_gap))
                    else:
                        # 间隔区域均值
                        gap_mean_scores[targeIdx].append(temp / mean_gap)
                        temp = 0
            plt.plot(np.arange(len(gap_mean_scores[targeIdx]))*mean_gap, gap_mean_scores[targeIdx], color=linesColors[targeIdx], label=Labels[targeIdx])
        plt.legend()
        if figTitle: plt.title(figTitle)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(True)
        plt.savefig(savePath, dpi=500)
        del fig, gap_mean_scores, temp

    @staticmethod
    def DrawLinesWithVLines(targetList, Labels, meanGap=10,
                            regionalMean=False, tolerance=0.99, savePath="fig.png",
                            figTitle=None,  xLabel="Episode", yLabel="AoI"):
        fig = plt.figure()
        fig.add_subplot(111)
        linesColors = ['red', 'blue', 'darkmagenta', 'green', 'yellow', 'lawngreen', 'darkred']
        maxNum, minNum = max(targetList[0]), min(targetList[0])
        # -----------------------------------------------------------------------
        # 求区域均值
        gap_mean_scores = []
        for targeIdx in range(0, len(targetList)):
            temp, doubles = 0, 0
            gap_mean_scores.append([])
            for meanIdx in range(0, len(targetList[targeIdx])):
                temp += targetList[targeIdx][meanIdx]
                if (meanIdx + 1) % meanGap == 0:
                    if not regionalMean:
                        # 全局均值
                        doubles += 1
                        gap_mean_scores[targeIdx].append(temp / (doubles * meanGap))
                    else:
                        # 间隔区域均值
                        gap_mean_scores[targeIdx].append(temp / meanGap)
                        temp = 0
        # -----------------------------------------------------------------------
        # 完整遍历，求最终收敛的点
        tailored, vPoints = list(), list()
        for listIdx in range(len(gap_mean_scores)):
            curMin = min(gap_mean_scores[listIdx])
            vPoints.append(0)
            for dataIdx in range(len(gap_mean_scores[listIdx])):
                # 这里的处理就是记录最后一个
                # if gap_mean_scores[listIdx][dataIdx] >= curMin + curMin * (1-tolerance):
                #     vPoints[-1] = dataIdx
                # 这里的处理就是记录第一个就行
                if gap_mean_scores[listIdx][dataIdx] < curMin + curMin * (1 - tolerance):
                    vPoints[-1] = dataIdx
                    break
        # -----------------------------------------------------------------------
        for index in range(0, len(gap_mean_scores)):
            maxNum = max(maxNum, max(gap_mean_scores[index]))
            minNum = min(minNum, min(gap_mean_scores[index]))
            plt.plot(np.arange(len(gap_mean_scores[index]))*meanGap, gap_mean_scores[index], color=linesColors[index],
                     label=Labels[index])
            plt.legend()
        for targeIdx in range(0, len(gap_mean_scores)):
            plt.vlines(vPoints[targeIdx]*meanGap, minNum, maxNum, linestyles='dashed', colors=linesColors[targeIdx])
            # plt.axvline(vPoints[targeIdx], linestyles='dashed', color=linesColors[targeIdx])
        plt.legend()
        if figTitle: plt.title(figTitle)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(True)
        plt.savefig(savePath, dpi=500)
        del fig

# --------------------------------------------------------------------------------------------------------
    @staticmethod
    def DrawGapMeanLinesWithLabels_2(targetList, Labels, mean_gap=10, regionalMean=False,
                                   savePath="fig.png", figTitle=None, xLabel="Episode", yLabel="Reward"):
        """
        功能：输入数据画间隔均值折线图
        形参：mean_gap：均值间隔长度，默认为10
                  regionalMean：布尔值，默认为False。是否为区域均值
        """
        fig = plt.figure()
        fig.add_subplot(111)
        gap_mean_scores = []
        linesColors = ['red', 'blue', 'darkmagenta', 'green', 'yellow', 'lawngreen', 'darkred']
        for targeIdx in range(0, len(targetList)):
            temp, doubles = 0, 0
            gap_mean_scores.append([])
            for meanIdx in range(0, len(targetList[targeIdx])):
                temp += targetList[targeIdx][meanIdx]
                if (meanIdx + 1) % mean_gap == 0:
                    if not regionalMean:
                        # 全局均值
                        doubles += 1
                        gap_mean_scores[targeIdx].append(temp / (doubles * mean_gap))
                    else:
                        # 间隔区域均值
                        gap_mean_scores[targeIdx].append(temp / mean_gap)
                        temp = 0
            plt.plot(np.arange(len(gap_mean_scores[targeIdx]))*mean_gap, gap_mean_scores[targeIdx], color=linesColors[targeIdx], label=Labels[targeIdx])
        plt.legend()
        if figTitle: plt.title(figTitle)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(True)
        plt.savefig(savePath, dpi=500)
        del fig, gap_mean_scores, temp

    @staticmethod
    def DrawLinesWithVLines_2(targetList, Labels, meanGap=10,
                            regionalMean=False, tolerance=0.99, savePath="fig.png",
                            figTitle=None,  xLabel="Episode", yLabel="AoI"):
        fig = plt.figure()
        fig.add_subplot(111)
        linesColors = ['red', 'blue', 'darkmagenta', 'red', 'yellow', 'lawngreen', 'darkred']
        maxNum, minNum = max(targetList[0]), min(targetList[0])
        # -----------------------------------------------------------------------
        # 求区域均值
        gap_mean_scores = []
        for targeIdx in range(0, len(targetList)):
            temp, doubles = 0, 0
            gap_mean_scores.append([])
            for meanIdx in range(0, len(targetList[targeIdx])):
                temp += targetList[targeIdx][meanIdx]
                if (meanIdx + 1) % meanGap == 0:
                    if not regionalMean:
                        # 全局均值
                        doubles += 1
                        gap_mean_scores[targeIdx].append(temp / (doubles * meanGap))
                    else:
                        # 间隔区域均值
                        gap_mean_scores[targeIdx].append(temp / meanGap)
                        temp = 0
        # -----------------------------------------------------------------------
        # 完整遍历，求最终收敛的点
        tailored, vPoints = list(), list()
        for listIdx in range(len(gap_mean_scores)):
            curMin = min(gap_mean_scores[listIdx])
            vPoints.append(0)
            for dataIdx in range(len(gap_mean_scores[listIdx])):
                # 这里的处理就是记录最后一个
                # if gap_mean_scores[listIdx][dataIdx] >= curMin + curMin * (1-tolerance):
                #     vPoints[-1] = dataIdx
                # 这里的处理就是记录第一个就行
                if gap_mean_scores[listIdx][dataIdx] < curMin + curMin * (1 - tolerance):
                    vPoints[-1] = dataIdx
                    break
        # -----------------------------------------------------------------------
        for index in range(0, len(gap_mean_scores)):
            maxNum = max(maxNum, max(gap_mean_scores[index]))
            minNum = min(minNum, min(gap_mean_scores[index]))
            plt.plot(np.arange(len(gap_mean_scores[index]))*meanGap, gap_mean_scores[index], color=linesColors[index],
                     label=Labels[index])
            plt.legend()
        for targeIdx in range(0, len(gap_mean_scores)):
            plt.vlines(vPoints[targeIdx]*meanGap, minNum, maxNum, linestyles='dashed', colors=linesColors[targeIdx])
            # plt.axvline(vPoints[targeIdx], linestyles='dashed', color=linesColors[targeIdx])
        plt.legend()
        if figTitle: plt.title(figTitle)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(True)
        plt.savefig(savePath, dpi=500)
        del fig

