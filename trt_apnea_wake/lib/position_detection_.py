import numpy as np
import cmath

def acosd(number):
    res = np.degrees(cmath.acos(number).real)
    return res

def asind(number):
    res = np.degrees(cmath.asin(number).real)
    return res

def asind_list(number):
    res = []
    for tmp_val in number:
        tmp_res = np.degrees(cmath.asin(tmp_val).real)
        res.append(tmp_res)
    return np.array(res)

def do_pos_analys(ACC, fs_acc = 32, ts_stt = 0, winodw = 30):
    epoch_raw = int(winodw * fs_acc)
    data_len = len(ACC)
    acc_mean = []
    for idx in range(0, data_len, epoch_raw):
        tmpData = ACC[idx:idx+epoch_raw]
        if len(tmpData) < epoch_raw:
            continue
        tmp_acc_mean = np.mean(tmpData, 0)
        acc_mean.append(tmp_acc_mean)

    pitch, roll = get_body_angle(np.vstack(acc_mean))
    return pitch, roll

def get_body_angle(acc_mean):
    idx_X, idx_Y, idx_Z = [0, 1, 2]
    data_len = len(acc_mean)
    roll = np.zeros([data_len, 1])
    ag_y = np.zeros([data_len, 2])
    ag_z = np.zeros([data_len, 2])
    pitch = np.real(asind_list(acc_mean[:,idx_X]))

    for idx in range(0, data_len):
        ag_y[idx,0] = np.real(asind(acc_mean[idx,idx_Y]))
        if acc_mean[idx,idx_Y] >= 0:
            ag_y[idx,1] = 180 - ag_y[idx,0]
        else:
            ag_y[idx,1] = -180 - ag_y[idx,0]
        ag_z[idx,0] = np.real(acosd(-acc_mean[idx,idx_Z]))
        ag_z[idx,1] = -ag_z[idx,0]

        vy = np.array([ag_y[idx,0], ag_y[idx,0], ag_y[idx,1], ag_y[idx,1]])
        vz = np.array([ag_z[idx,0], ag_z[idx,1], ag_z[idx,0], ag_z[idx,1]])
        diff_vy_vz = np.abs(vy - vz)
        idx_min = diff_vy_vz.argmin(0)
        roll[idx] = np.mean([vy[idx_min],
                             vz[idx_min]])
    return pitch, roll

def get_posture(pitch, roll):
    data_len = len(pitch)
    pitch = np.array(pitch)
    roll = np.array(roll)

    idx_lay_supine = pitch >= -40
    idx_lay_non_supine = np.zeros([len(pitch,)])

    i_pos_laying  = np.logical_or(idx_lay_supine, idx_lay_non_supine)

    ag_sup_left = -30
    ag_left_prn = -150
    ag_sup_right = 30
    ag_right_prn = 150

    i_pos_supine_pre = np.logical_and(i_pos_laying.reshape([data_len,]), (ag_sup_left<=roll).reshape([data_len,]))
    i_pos_supine = np.logical_and(i_pos_supine_pre, (roll<=ag_sup_right).reshape([data_len,]))
    i_pos_left_pre = np.logical_and(i_pos_laying.reshape([data_len,]), (ag_left_prn<roll).reshape([data_len,]))
    i_pos_left = np.logical_and(i_pos_left_pre, (roll<ag_sup_left).reshape([data_len,]))
    i_pos_right_pre = np.logical_and(i_pos_laying.reshape([data_len,]), (ag_sup_right<roll).reshape([data_len,]))
    i_pos_right = np.logical_and(i_pos_right_pre, (roll<ag_right_prn).reshape([data_len,]))
    i_pos_prone_prepre = np.logical_and(i_pos_laying.reshape([data_len,]), ~i_pos_supine)
    i_pos_prone_pre = np.logical_and(i_pos_prone_prepre, ~i_pos_left)
    i_pos_prone = np.logical_and(i_pos_prone_pre, ~i_pos_right)

    posture = np.zeros([len(pitch),])
    posture[i_pos_left] = 1
    posture[i_pos_supine] = 2
    posture[i_pos_right] = 3
    posture[i_pos_prone] = 4

    return posture

def position_change_(data):
    change_num = 0
    change_idx = []
    for idx in range(len(data)-1):
        if data[idx] != int(data[idx+1]):
            change_num += 1
            change_idx.append(idx)
    return change_num, change_idx

def position_percentage_(data):
    data_len = len(data)

    pos_stand_per = len(np.where(data == 0)[0]) / data_len
    pos_left_per = len(np.where(data == 1)[0]) / data_len
    pos_supine_per = len(np.where(data == 2)[0]) / data_len
    pos_right_per = len(np.where(data == 3)[0]) / data_len
    pos_prone_per = len(np.where(data == 4)[0]) / data_len

    return pos_stand_per*100, pos_left_per*100, pos_supine_per*100, pos_right_per*100, pos_prone_per*100

if __name__ == '__main__':
    print('position detection main function')