from scipy import stats
from.filters_ import *
from .position_detection_ import *
import pandas as pd

def detection_trt(data, Fs = 32, winodw = 60, window_postion = 30,
                  filtering = 0, cut_low = 0.1, cut_high = 1,
                  threshold_resp_normal = 0.50, threshold_resp_apnea = 0.70,
                  detection_start_order_posture = 5, detection_end_order_resp = 3,
                  smoothing_filter = signal.savgol_filter, smoothing_filter_order = 11,
                  over_est_backward_window = 30):

    features = acc2resp(data,
                        winodw, Fs,
                        filtering=filtering, cutOffLow=cut_low, cutOffHigh=cut_high)
    roll_1min, pitch_1min = do_pos_analys(data, winodw=winodw)
    posture = get_posture(roll_1min, pitch_1min)

    roll_30sec, pitch_30sec = do_pos_analys(data, winodw=window_postion)
    posture_30sec = get_posture(roll_30sec, pitch_30sec)

    count_resp_1 = smoothing_filter(features[:, -1], smoothing_filter_order, 3) >= threshold_resp_normal
    count_resp_2 = smoothing_filter(features[:, -2], smoothing_filter_order, 3) >= threshold_resp_apnea
    count_resp_3 = smoothing_filter(features[:, -3], smoothing_filter_order, 3) >= threshold_resp_apnea

    count_resp = np.logical_or(count_resp_1, count_resp_2)
    count_resp = np.logical_or(count_resp, count_resp_3)

    start_idx_resp, end_idx_resp = detection_start_end(count_resp, posture,
                                             order_resp = detection_end_order_resp, order_posture = detection_start_order_posture)
    end_idx_resp = over_est_filter(count_resp, end_idx_resp, backward_window=over_est_backward_window)
    end_idx_resp = detection_end_posture(posture[start_idx_resp:end_idx_resp]) + 1 + start_idx_resp

    posture_30sec = posture_30sec[int(start_idx_resp*2):int(end_idx_resp*2)]

    return start_idx_resp, end_idx_resp, posture_30sec


def acc2resp(data, window, Fs, filtering = 0, cutOffLow = 0.1, cutOffHigh = 1):

    if filtering == 1:
        newData = np.zeros(data.shape)
        newData[:, 0] = butter_bandpass_filter(data[:, 0], cutOffLow, cutOffHigh, Fs)
        newData[:, 1] = butter_bandpass_filter(data[:, 1], cutOffLow, cutOffHigh, Fs)
        newData[:, 2] = butter_bandpass_filter(data[:, 2], cutOffLow, cutOffHigh, Fs)

    elif filtering == 2:
        newData = np.zeros(data.shape)
        newData[:, 0] = smoothing(data[:, 0])
        newData[:, 1] = smoothing(data[:, 1])
        newData[:, 2] = smoothing(data[:, 2])
    else:
        newData = data
    windowRaw =  int(window * Fs)
    ACCX = newData[:,0]
    ACCY = newData[:,1]
    ACCZ = newData[:,2]
    Features = []
    for idx in range(0, len(newData), windowRaw):
        tmpACCX = ACCX[idx:idx+windowRaw]
        tmpACCY = ACCY[idx:idx + windowRaw]
        tmpACCZ = ACCZ[idx:idx + windowRaw]
        ## discard mean
        if filtering == 0:
            tmpACCX = tmpACCX - np.mean(tmpACCX)
            tmpACCY = tmpACCY - np.mean(tmpACCY)
            tmpACCZ = tmpACCZ - np.mean(tmpACCZ)

        if len(tmpACCZ) < windowRaw:
            continue

        f, Pxx_den_x = signal.welch(signal.detrend(stats.zscore(tmpACCX)), fs=Fs, nperseg  = 1024, nfft = 2048)
        f, Pxx_den_y = signal.welch(signal.detrend(stats.zscore(tmpACCY)), fs=Fs, nperseg  = 1024, nfft = 2048)
        f, Pxx_den_z = signal.welch(signal.detrend(stats.zscore(tmpACCZ)), fs=Fs, nperseg  = 1024, nfft = 2048)
        Pxx_den_y_list = [Pxx_den_x, Pxx_den_y, Pxx_den_z]

        ## normal
        VLFIdx = np.where(np.logical_and(f>=0.01,
                                     f<0.05))[0]
        LFIdx = np.where(np.logical_and(f>=0.05,
                                     f<0.15))[0]
        HFIdx = np.where(np.logical_and(f>=0.15,
                                     f<0.5))[0]
        respFreIdx = np.where(np.logical_and(f>=0.15,
                                     f<0.4))[0]
        upStreamIdx = np.where(np.logical_and(f>=0.4,
                                     f<1))[0]
        ## abnormal
        VLFIdx2 = np.where(np.logical_and(f>=0.01,
                                     f<0.05))[0]
        LFIdx2 = np.where(np.logical_and(f>=0.05,
                                     f<0.2))[0]
        HFIdx2 = np.where(np.logical_and(f>=0.2,
                                     f<0.58))[0]
        upStreamIdx2 = np.where(np.logical_and(f>=0.58,
                                     f<1))[0]
        ## slow breathing
        respFreIdx_slow = np.where(np.logical_and(f>=0.10,
                                     f<0.35))[0]
        upStreamIdx_slow = np.where(np.logical_and(f>=0.35,
                                     f<1))[0]
        LFIdx_slow = np.where(np.logical_and(f>=0.05,
                                     f<0.10))[0]

        ratio_x = (np.sum(Pxx_den_x[respFreIdx]) * (np.max(Pxx_den_x[respFreIdx]) + np.max(Pxx_den_x[HFIdx2])))
        ratio_y = (np.sum(Pxx_den_y[respFreIdx]) * (np.max(Pxx_den_y[respFreIdx]) + np.max(Pxx_den_y[HFIdx2])))
        ratio_z = (np.sum(Pxx_den_z[respFreIdx]) * (np.max(Pxx_den_z[respFreIdx]) + np.max(Pxx_den_z[HFIdx2])))

        ratio_list = [ratio_x , ratio_y , ratio_z]
        ratio_list_idx = np.argmax(ratio_list)
        Pxx_den = Pxx_den_y_list[ratio_list_idx]

        ratio = np.sum(Pxx_den[respFreIdx]) / (
                    np.sum(Pxx_den[respFreIdx]) + np.sum(Pxx_den[upStreamIdx]) + np.sum(Pxx_den[LFIdx]))
        ratio2 = (np.sum(Pxx_den[HFIdx2]) + np.sum(Pxx_den[VLFIdx]))  / (
                    np.sum(Pxx_den[HFIdx2]) + np.sum(Pxx_den[upStreamIdx2]) + np.sum(Pxx_den[VLFIdx]) + np.sum(Pxx_den[LFIdx2]))
        ratio_slow = np.sum(Pxx_den[respFreIdx_slow]) / (
                    np.sum(Pxx_den[respFreIdx_slow]) + np.sum(Pxx_den[upStreamIdx_slow]) + np.sum(Pxx_den[LFIdx_slow]))
        Features.append([ratio_slow, ratio2, ratio])
    return np.array(Features)


def detection_start_end(data, posture, order_resp = 2, order_posture = 5):
    num = 0
    start_idx = 0
    end_idx = len(data) -1
    for idx, val in enumerate(posture):
        if val != 0:
            num += 1
            if num >= order_posture:
                start_idx = idx - order_posture + 1
                num = 0
                break
            else:
                pass
        else:
            num = 0
    num = 0
    for idx in reversed(range(len(data))):
        val = data[idx]
        if val == 1:
            num += 1
            if num >= order_resp:
                end_idx = idx + order_resp - 1
                num = 0
                break
            else:
                pass
        else:
            num = 0
    num = 0
    return start_idx, end_idx


def detection_end_posture(data, order_end = 5):
    num = 0
    for idx in reversed(range(len(data))):
        val = data[idx]
        if val != 0:
            num += 1
            if num >= order_end:
                end_idx = idx + order_end - 1
                num = 0
                break
            else:
                pass
        else:
            num = 0
    num = 0
    return end_idx


def over_est_filter(data, end_idx, backward_window = 60,
                    min_resp_len = 6, min_artifact_len = 3):
    data_info = non_wear_algorithm_short_window(data)
    data_info_ndarray = data_info.values
    data_valid_array = np.ones([len(data_info,)])

    data_info_ndarray_start_idx = data_info_ndarray[:,0]
    data_info_ndarray_end_idx = data_info_ndarray[:,1]
    data_info_ndarray_num = data_info_ndarray[:,2]

    first_end_idx_fitering_idx = np.where(data_info_ndarray_end_idx <=end_idx)[0]
    data_info_ndarray = data_info_ndarray[first_end_idx_fitering_idx]
    data_info_ndarray_start_idx = data_info_ndarray_start_idx[first_end_idx_fitering_idx]
    data_info_ndarray_end_idx = data_info_ndarray_end_idx[first_end_idx_fitering_idx]
    data_info_ndarray_num = data_info_ndarray_num[first_end_idx_fitering_idx]
    naji_break = True
    for idx, tmp_data_info in zip(reversed(range(len(data_info_ndarray))), reversed(data_info_ndarray)):
        tmp_start_idx = tmp_data_info[0]
        tmp_end_idx = tmp_data_info[1]
        tmp_num = tmp_data_info[2]
        end_idx = tmp_end_idx

        if data_valid_array[idx] == 0:
            continue
        else:
            pass

        if tmp_num < min_artifact_len:
            data_valid_array[idx] = 0
            continue
        else:
            pass

        if tmp_num >= min_resp_len:
            end_idx = tmp_end_idx
            break
        else:
            tmp_idx_backward = np.where(np.logical_and(data_info_ndarray_end_idx >= tmp_start_idx - backward_window,
                                                       data_info_ndarray_end_idx < tmp_start_idx))[0]
            for idx_idx, tmp_tmp_num, tmp_tmp_end_idx in zip(tmp_idx_backward,
                                            data_info_ndarray_num[tmp_idx_backward],
                                            data_info_ndarray_end_idx[tmp_idx_backward]):
                if data_info_ndarray_num[idx_idx] < min_resp_len:
                    data_valid_array[idx_idx] = 0
                else:
                    end_idx = data_info_ndarray_end_idx[idx]
                    naji_break = False
                    break
            data_valid_array[idx] = 0

        if naji_break == False:
            break
        else:
            pass

    return end_idx


def non_wear_algorithm_short_window(data):
    '''
    make binary data to data frame with data info
    example ::
    data = [0,1,1,1,0,0,1,1]
    --> data frame like this
    start_idx   |   end_idx |   number
    1               3           3
    6               7           2
    :param data: binary data consist of 1 & 0
    :return: data frame with binary data info
    '''
    data_buffer = list(data)
    data_buffer.append(0)
    data_buffer = np.array(data_buffer)
    data_numpy = np.array(data)
    answer = data_numpy > 0
    data_buffer_anwser = data_buffer > 0
    idx_start = []
    idx_end = []
    val = []
    tmpVal = 0
    tmp_idx_start = None
    tmp_idx_end = None
    for idx, tmpData in enumerate(answer):
        if tmpData == 1:
            if tmp_idx_start == None:
                tmp_idx_start = idx
            tmpVal += 1
            if data_buffer_anwser[idx+1] == 0:
                if tmp_idx_end == None:
                    tmp_idx_end = idx
                val.append(tmpVal)
                idx_start.append(tmp_idx_start)
                idx_end.append(tmp_idx_end)

                tmpVal = 0
                tmp_idx_start = None
                tmp_idx_end = None
        else:
            pass
    dict = {'idx_start':idx_start,
            'idx_end':idx_end,
            'number':val}
    return pd.DataFrame(dict)

if __name__ == '__main__':
    print('TRT detection main function')