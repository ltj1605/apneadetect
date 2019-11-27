import numpy as np

def str2Int(data):
    '''

    :param data:
    :return:
    '''
    newData = [0,0,0,0]
    for idx, val in enumerate(data):
        newData[idx] = val
    return str(newData)

def list2intlist(data):
    new_data = []

    for idx, val in enumerate(data):
        new_data.append(int(val))
    return [new_data]

def commaParsor(data):
    """

    :param data:
    :return:
    """
    newData = []
    for tmpData in data:
        newnewData = []
        for tmptmpData in tmpData:
            if tmptmpData == ',':
                break
            else:
                newnewData.append(tmptmpData)
        newData.append(float(''.join(newnewData)))
    return newData

def isNaN(num):
    return num != num

def findNan(data):
    NanIdx = []
    for idx, tmpData in enumerate(data):
        if isNaN(np.sum(tmpData)):
            NanIdx.append(idx)
    return NanIdx

def re_sample_wake_sleep(data, position, order = 2):
    new_data_final = np.zeros([len(position),])
    new_data = []
    for tmp_data in data:
        for idx in range(order):
            new_data.append(tmp_data)
    new_data_final[0:len(new_data)] = np.array(new_data)
    return new_data_final

if __name__ == '__main__':
    print('utills main function')