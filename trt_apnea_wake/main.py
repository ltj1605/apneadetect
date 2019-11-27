from lib.apnea_sleep_detection_ import ApneaDetector, commaParsor, str2Int, np, list2intlist, re_sample_wake_sleep
from lib.TRT_detection_ import detection_trt, pd, position_change_, position_percentage_
import os
import sys
from datetime import datetime, timedelta
import json
from collections import OrderedDict
if __name__ == '__main__':
    ########################################## parameter load ##########################################
    code = 1
    trt_start_str = sys.argv[1]
    trt_end_str = sys.argv[2]
    trt_start = datetime.strptime(trt_start_str, "%Y-%m-%d %H:%M:%S")
    trt_end = datetime.strptime(trt_end_str, "%Y-%m-%d %H:%M:%S")
    modelFolderName = 'model96'
    modelFolderNameSleep = 'modelSleepTest'
    ## algorithm option setting
    options = []
    fSetting = open('./setting/setting.txt', 'r')
    modelName = os.listdir('./model/' + modelFolderName)
    modelNameSleep = os.listdir('./model/' + modelFolderNameSleep)
    for tmpSetting in fSetting.readlines():
        options.append(tmpSetting.strip())
    options = commaParsor(options)
    ## model & normalization parameter (mean & std) load
    model = np.load('./model/' + modelFolderName + '/' + modelName[0],
                    allow_pickle=True).item()
    parameter = np.load('./model/' + modelFolderName + '/' + modelName[1],
                        allow_pickle=True).item()
    sleepModel = np.load('./model/' + modelFolderNameSleep + '/' + modelNameSleep[0],
                         allow_pickle=True).item()
    sleepParameter = np.load('./model/' + modelFolderNameSleep + '/' + modelNameSleep[1],
                             allow_pickle=True).item()
    currentDirMain = os.getcwd()
    ########################################## TRT & position detection ##########################################
    sampleACCData = pd.read_csv('./data/ConvertedACC.csv').values
    acc_data = sampleACCData[:,1:4]
    start_idx, end_idx, position = detection_trt(acc_data)
    ## sleep info calculation
    trt = len(acc_data) / (options[1] * options[4])
    tib_start = trt_start + timedelta(minutes=start_idx)
    tib_end = trt_start + timedelta(minutes=end_idx)
    ## position info
    position_change_num, position_change_idx = position_change_(position)
    pos_stand_per, pos_left_per, pos_supine_per, pos_right_per, pos_prone_per = position_percentage_(position)
    posture_raw_data = list2intlist(position)
    posture_raw_data_len = len(posture_raw_data[0])
    ## acc data slice
    acc_data = acc_data[int(start_idx * float(options[1]) * float(options[4])):int(end_idx * float(options[1]) * float(options[4]))]
    ########################################## apnea & wake detection ##########################################
    ApneaDetectorObject = ApneaDetector(acc_data, acc_data, currentDir=currentDirMain,
                                        FsACC=float(options[1]), FsECG=float(options[0]),
                                        positionDetectSec=float(options[2]),
                                        positionDetectThreshold=float(options[3]),
                                        featureEpoch=options[4], model=model, parameter=parameter, sleepModel=sleepModel,
                                        sleepParam=sleepParameter,
                                        smoothingOrder=options[6], countOrder=options[7])

    ## artifact detection
    artifacts = ApneaDetectorObject.extractPositionIdx()
    ## feature extraction
    sleepAnalysisFeatures = ApneaDetectorObject.sleepAnalysisFeatureEx()
    features_wake_sleep, acc_data_list = ApneaDetectorObject.positionBasedWaveLetFeatures(artifacts)
    ## estimation AHI
    prop, pksMins, count, time10, sleepEstimation, TST, SOL, WASO, SE, AHI = ApneaDetectorObject.estimateTestDataPosition(
        testDataList=features_wake_sleep, ACCrawDatasTest=acc_data_list, optimalThreshold=options[5], optimalThresholdSleep=options[8],
        labelSmoothIdx=1)
    time_to_sleep_datetime = tib_start + timedelta(minutes=SOL)
    sleep_wake_raw_data = list2intlist(re_sample_wake_sleep(sleepEstimation, position))
    sleep_wake_raw_data_len = len(sleep_wake_raw_data[0])
    ########################################## print result ##########################################
    # ## ahi info
    # print('apnea_count ' + str(count))
    # print('ahi_index ' + str(AHI))
    # ## sleep info (min)
    # print('sleep_efficiency ' + str(SE*100))
    # print('total_record_time ' + str(((tib_end - tib_start).seconds / 60)))
    # print('total_sleep_time ' + str(TST))
    # print('time_to_sleep_min ' + str(SOL))
    # print('wake_time_during_sleep ' +str(WASO))
    # ## sleep info (date time)
    # print('in_bed_time ' + str(tib_start))
    # print('out_bed_time ' + str(tib_end))
    # print('time_to_sleep_datetime ' + str(time_to_sleep_datetime))
    # ## position info
    # print('posture_change_num ' + str(position_change_num))
    # print('posture_per_stand ' + str(pos_stand_per))
    # print('posture_per_left ' + str(pos_left_per))
    # print('posture_per_supine ' + str(pos_supine_per))
    # print('posture_per_right ' + str(pos_right_per))
    # print('posture_per_prone ' + str(pos_prone_per))
    # print('posture_raw_data ' + str(position))
    # ## error code
    # print('code ' + str(code))
    ##
    ########################################## make json file ##########################################
    json_output = {'apnea_info':[{'apnea_count':int(count),
                                  'ahi_index':float(AHI)}],

                   'sleep_info':[{'sleep_efficiency':float(SE * 100),
                                  'total_record_time':float(((tib_end - tib_start).seconds / 60)),
                                  'total_sleep_time':int(TST),
                                  'time_to_sleep_min':int(SOL),
                                  'wake_time_during_sleep':int(WASO),
                                  'in_bed_time':tib_start.strftime("%Y-%m-%d %H:%M:%S"),
                                  'out_bed_time':tib_end.strftime("%Y-%m-%d %H:%M:%S"),
                                  'time_to_sleep_datetime':time_to_sleep_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                                  'sleep_wake_raw_data':sleep_wake_raw_data,
                                  'sleep_wake_raw_data_len':sleep_wake_raw_data_len
                                  }],

                   'posture_info':[{'posture_change_num':position_change_num,
                                    'posture_toss_idx': list2intlist(position_change_idx),
                                    'posture_per_stand':pos_stand_per,
                                    'posture_per_left':pos_left_per,
                                    'posture_per_supine':pos_supine_per,
                                    'posture_per_right':pos_right_per,
                                    'posture_per_prone':pos_prone_per,
                                    'posture_raw_data':posture_raw_data,
                                    'posture_raw_data_len':posture_raw_data_len}],

                   'code':[{'error_code':code}]}

    with open('./data/json_output2.json', 'w', encoding='utf-8') as make_file:
        json.dump(json_output, make_file, ensure_ascii=False, indent='\t' )
