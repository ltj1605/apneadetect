from scipy import stats
from scipy.interpolate import interp1d
from pywt import wavedec
from .peak_detection_ import peakdetect, _smooth
from .respiration_detection_ import *
from .filters_ import *
from .utills_ import *


class ApneaDetector:
    def __init__(self, ECG, ACC, model, parameter, sleepModel, sleepParam,
                 currentDir, FsECG=256, FsACC=32, positionDetectSec=10, positionDetectThreshold=0.5, featureEpoch=60,
                 smoothingOrder=32 * 15, countOrder=32 * 25):
        """
        :param ECG:
        :param ACC:
        :param model:
        :param parameter:
        :param sleepModel:
        :param sleepParam:
        :param currentDir:
        :param FsECG:
        :param FsACC:
        :param positionDetectSec:
        :param positionDetectThreshold:
        :param featureEpoch:
        :param smoothingOrder:
        :param countOrder:
        """
        self.ECG = ECG
        self.ACC_raw = self.buffer(ACC)
        self.ACC = np.zeros(ACC.shape)
        self.ACC[:, 0] = self.smoothing(ACC[:, 0])
        self.ACC[:, 1] = self.smoothing(ACC[:, 1])
        self.ACC[:, 2] = self.smoothing(ACC[:, 2])

        self.recordTime = len(ACC) / FsACC
        self.FsECG = FsECG
        self.FsACC = FsACC
        self.model = model
        self.parameter = parameter
        self.sleepModel = sleepModel
        self.sleepParam = sleepParam
        self.featureEpoch = featureEpoch
        self.positionDetectorSec = positionDetectSec
        self.positionDetectorThreshold = positionDetectThreshold
        self.logData = open(currentDir + './result/log.txt', 'w')
        self.smmothOrderCountAlgorithm = smoothingOrder
        self.countOrderCountAlgorithm = countOrder
        self.currentDir = currentDir
        self.TIB = (len(ACC) / (FsACC)) / 3600

        self.sleepFeatures = 0

    def buffer(self, data):
        return data

    def sleepClassification(self, optimalThreshold):
        mu = self.sleepParam['mean']
        std = self.sleepParam['std']
        sleepFeatures = self.zscoreTest(self.sleepFeatures, mu, std)
        if isNaN(np.sum(sleepFeatures)):
            nanIdx = findNan(sleepFeatures)
            dummyData = np.ones([sleepFeatures.shape[1]])
            sleepFeatures[nanIdx] = dummyData
            estimation = self.sleepModel.predict_proba(sleepFeatures)[:, 1] >= optimalThreshold
            estimation[nanIdx] = 0
        else:
            estimation = self.sleepModel.predict_proba(sleepFeatures)[:, 1] >= optimalThreshold
        estimationSOL = self.sleepModel.predict_proba(sleepFeatures)[:, 1] >= 0.1
        TST, SOL, WASO, SE = calSleepParam(estimation, estimationSOL)
        return estimation, TST, SOL, WASO, SE

    def sleepAnalysisFeatureEx(self):
        tmpACCFeatures = self.ACCFeatureEx()
        tmpActivityFeatures = self.calActivityCount()
        tmpRespFeatures = self.RespFeatureExByNeurokit()
        tmpTotalTestFeatuers = np.c_[tmpACCFeatures, tmpActivityFeatures, tmpRespFeatures]
        self.sleepFeatures = tmpTotalTestFeatuers
        return tmpTotalTestFeatuers

    def RespFeatureExByNeurokit(self, window=60, Fs=32, filtering=2, cutOffLow=0.1, cutOffHigh=1):
        data = self.ACC_raw
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
        windowRaw = int(window * Fs)
        ACCX = newData[:, 0]
        ACCY = newData[:, 1]
        ACCZ = newData[:, 2]
        features, skipIdx = [], []
        for idx in range(0, len(newData), windowRaw):
            tmpACCX = ACCX[idx:idx + windowRaw]
            tmpACCY = ACCY[idx:idx + windowRaw]
            tmpACCZ = ACCZ[idx:idx + windowRaw]

            ## covariance matrix
            tmpCbindMat = np.c_[tmpACCX, tmpACCY, tmpACCZ]
            tmpCov = np.cov(tmpCbindMat.T)
            ## eigen values
            tmpEigenVec = np.linalg.eig(tmpCov)[0]
            ## eigen vector-based weight
            n1 = tmpEigenVec[0] / np.sum(tmpEigenVec)
            n2 = tmpEigenVec[1] / np.sum(tmpEigenVec)
            n3 = tmpEigenVec[2] / np.sum(tmpEigenVec)
            ## calculation respiration
            if filtering == 1:
                tmpRasp = (n1 * (tmpACCX)) + (n2 * (tmpACCY)) + (n3 * (tmpACCZ))
            elif filtering == 2:
                tmpRasp = (n1 * signal.detrend(tmpACCX)) + (n2 * signal.detrend(tmpACCY)) + (
                        n3 * signal.detrend(tmpACCZ))
            else:
                tmpRasp = (n1 * signal.detrend(tmpACCX)) + (n2 * signal.detrend(tmpACCY)) + (
                        n3 * signal.detrend(tmpACCZ))

            if len(tmpRasp) < windowRaw:
                continue
            if filtering == 0:
                f, Pxx_den = signal.periodogram(smoothing(tmpRasp), fs=Fs)
            else:
                f, Pxx_den = signal.periodogram(tmpRasp, fs=Fs)

            VLFIdx = np.where(np.logical_and(f >= 0.01,
                                             f < 0.05))[0]
            LFIdx = np.where(np.logical_and(f >= 0.05,
                                            f < 0.15))[0]
            HFIdx = np.where(np.logical_and(f >= 0.15,
                                            f < 0.5))[0]
            respFreIdx = np.where(np.logical_and(f >= 0.15,
                                                 f < 0.4))[0]
            Pxx_denResp = Pxx_den[respFreIdx]
            tmpMaxF = f[respFreIdx][np.argmax(Pxx_denResp)]
            tmpMaxFAmp = Pxx_den[respFreIdx][np.argmax(Pxx_denResp)]
            # print(idx)
            try:
                neuroKitRespObject = rsp_process(tmpRasp, sampling_rate=Fs)
            except:
                neuroKitRespObject = rsp_process(tmpRasp, sampling_rate=int(Fs * (9 / 10)))
            # tmpFilteredResp = neuroKitRespObject['df']['RSP_Filtered'].values
            # tmpCycleOnset = neuroKitRespObject['RSP']['Cycles_Onsets']
            # tmpExpirationOnset = neuroKitRespObject['RSP']['Expiration_Onsets']
            # tmpCycleLength = neuroKitRespObject['RSP']['Cycles_Length']

            ## respiratory variability features
            tmpVariablityObject = neuroKitRespObject['RSP']['Respiratory_Variability']
            tmpRSPV_SD = tmpVariablityObject['RSPV_SD']
            tmpRSPV_RMSSD = tmpVariablityObject['RSPV_RMSSD']
            tmpRSPV_RMSSD_Log = tmpVariablityObject['RSPV_RMSSD_Log']

            features.append(
                ## time domain features of respiration raw signal
                np.array([
                    ## time domain featuers 2
                    tmpRSPV_SD,
                    tmpRSPV_RMSSD,
                    tmpRSPV_RMSSD_Log,
                    ## spectral features of respiration
                    np.log(np.sum(Pxx_den[VLFIdx])),
                    np.log(np.sum(Pxx_den[LFIdx])),
                    np.log(np.sum(Pxx_den[HFIdx])),
                    (np.sum(Pxx_den[LFIdx])) / (np.sum(Pxx_den[HFIdx])),
                    ## respiratory features
                    tmpMaxF,
                    tmpMaxFAmp])
            )
        features = np.vstack(features)
        return features

    def calTmpACCFeatures(self, data, dataNofilt, Fs, lowCut, highCut):
        """

        :param data:
        :param dataNofilt:
        :return:
        mean, std, kurtosis, skewness, entropy, bandpower
        """
        ## time domain
        mean = np.mean(data)
        std = np.std(data)
        kurtosis = stats.kurtosis(data)
        skewess = stats.skew(data)
        entropy = stats.entropy(np.exp(data))
        ## frequency domain
        tmpPeriodogramFrame = signal.periodogram(dataNofilt,
                                                 fs=Fs,
                                                 scaling='density')
        tmpPeriodogram = tmpPeriodogramFrame[1]
        tmpPeriodogramX = tmpPeriodogramFrame[0]

        idxBand = np.where(np.logical_and(tmpPeriodogramX >= lowCut,
                                          tmpPeriodogramX < highCut))[0]

        bandPower = (np.sum(tmpPeriodogram[idxBand])) / (np.sum(tmpPeriodogram))

        return mean, std, kurtosis, skewess, entropy, bandPower

    def ACCFeatureEx(self, Fs=32, epoch=60, lowCut=0.5, highCut=11, normalization=1):
        ACCData = self.ACC_raw
        totalFeatures = []
        dataX = butter_bandpass_filter(ACCData[:, 0], lowCut, highCut, Fs)
        dataY = butter_bandpass_filter(ACCData[:, 1], lowCut, highCut, Fs)
        dataZ = butter_bandpass_filter(ACCData[:, 2], lowCut, highCut, Fs)
        dataSqurt = squarRoot(np.c_[dataX,
                                    dataY,
                                    dataZ])

        if normalization:
            dataSqurt = stats.zscore(dataSqurt)
        else:
            pass

        dataXNoFilt = ACCData[:, 0]
        dataYNoFilt = ACCData[:, 1]
        dataZNoFilt = ACCData[:, 2]
        dataSqurtNoFilt = squarRoot(np.c_[dataXNoFilt,
                                          dataYNoFilt,
                                          dataZNoFilt])
        dataPoint = len(dataX)
        epochDataLen = int(epoch * Fs)
        for tmpIdx in range(0, dataPoint, epochDataLen):
            tmpACCX = dataX[tmpIdx:tmpIdx + epochDataLen]
            tmpACCXNoFilt = dataXNoFilt[tmpIdx:tmpIdx + epochDataLen]
            if len(tmpACCX) < epochDataLen:
                continue
            tmpMeanSqurt, tmpStdSqurt, tmpKurtosisSquirt, tmpSkewessSquirt, tmpEntropySqurt, tmpBandPowerSqurt = self.calTmpACCFeatures(
                tmpACCX,
                tmpACCXNoFilt, Fs=Fs, lowCut=lowCut, highCut=highCut)

            totalFeatures.append([tmpMeanSqurt, tmpStdSqurt, tmpKurtosisSquirt, tmpSkewessSquirt, tmpEntropySqurt,
                                  tmpBandPowerSqurt])
        totalFeatures = np.vstack(totalFeatures)
        return totalFeatures

    def calActivityCount(self, window=60, subWindow=1, Fs=32, cutOffLow=0.5, cutOffHigh=11, filtering=1):
        data = self.ACC_raw
        dataPoint = len(data)
        epoch = int(window * Fs)
        subEpoch = int(Fs * subWindow)
        activityCountFeatures = []
        activityCountFeatures2 = []
        activityCount = []
        if filtering:
            newData = np.zeros(data.shape)
            newData[:, 0] = butter_bandpass_filter(data[:, 0], cutOffLow, cutOffHigh, Fs)
            newData[:, 1] = butter_bandpass_filter(data[:, 1], cutOffLow, cutOffHigh, Fs)
            newData[:, 2] = butter_bandpass_filter(data[:, 2], cutOffLow, cutOffHigh, Fs)
        else:
            newData = data

        newData = squarRoot(newData)

        for Idx in range(0, dataPoint, epoch):
            subCount = []
            tmpData = newData[Idx:Idx + epoch]
            if len(tmpData) < epoch:
                continue

            for tmpIdx in range(0, len(tmpData), subEpoch):
                tmptmpData = tmpData[tmpIdx:tmpIdx + subEpoch]
                subCount.append(np.sum(tmptmpData))
            ## 5time
            threshold = np.mean(subCount)
            fiveTimeNum = np.sum(np.array(subCount) >= threshold)

            activityCountFeatures.append([np.sum(subCount),
                                          np.max(subCount),
                                          np.min(subCount),
                                          np.std(subCount),
                                          np.log(np.max(subCount) + 1),
                                          np.log(np.sum(subCount) + 1),
                                          fiveTimeNum
                                          ])
            activityCount.append(np.sum(subCount))
        activityCountFeatures = np.vstack(activityCountFeatures)

        for q in range(len(activityCount)):
            cursor10and5 = q - 10
            cursor5and5 = q - 5

            if cursor10and5 < 0:
                cursor10and5 = 0
            else:
                pass
            if cursor5and5 < 0:
                cursor5and5 = 0
            else:
                pass
            centroidCount10and5 = activityCount[cursor10and5:q + 5]
            centroidCount5and5 = activityCount[cursor5and5:q + 5]

            sum10and5 = np.sum(centroidCount10and5)
            sum5and5 = np.sum(centroidCount5and5)
            std10and5 = np.std(centroidCount10and5)
            std5and5 = np.std(centroidCount5and5)
            max10and5 = np.max(centroidCount10and5)
            max5and5 = np.max(centroidCount5and5)
            min10and5 = np.min(centroidCount10and5)
            min5and5 = np.min(centroidCount5and5)

            activityCountFeatures2.append([sum10and5, sum5and5,
                                           std10and5, std5and5,
                                           max10and5, max5and5,
                                           min10and5, min5and5])
        activityCountFeatures2 = np.vstack(activityCountFeatures2)
        totalFeatures = np.c_[activityCountFeatures, activityCountFeatures2]
        return totalFeatures

    def envelope1D(self, data, peakDetectionMethod='f1', interPolKind='linear'):
        if peakDetectionMethod == 'f1':
            pksMax = signal.argrelmax(data)[0]
            pksMin = signal.argrelmin(data)[0]
        elif peakDetectionMethod == 'f2':
            pksMax = np.vstack(peakdetect(data, lookahead=32 * 2)[0])[:, 0]
            pksMin = np.vstack(peakdetect(data, lookahead=32 * 2)[1])[:, 0]
        else:
            print('peakDetectionMethod = ["f1", "f2"]')
            raise ValueError

        pksMax = np.concatenate([np.zeros([1, ]), pksMax])
        pksMax = np.concatenate([pksMax, np.array(len(data) - 1).reshape([1, ])])

        pksMin = np.concatenate([np.zeros([1, ]), pksMin])
        pksMin = np.concatenate([pksMin, np.array(len(data) - 1).reshape([1, ])])

        YMax = data[np.int64(pksMax)]
        YMin = data[np.int64(pksMin)]

        fMax = interp1d(pksMax, YMax, kind=interPolKind)
        fMin = interp1d(pksMin, YMin, kind=interPolKind)

        XnewMax = np.linspace(np.min(pksMax), np.max(pksMax), len(data))
        XnewMin = np.linspace(np.min(pksMin), np.max(pksMin), len(data))

        YnewMax = fMax(XnewMax)
        YnewMin = fMin(XnewMin)

        return YnewMax, YnewMin

    def envelopeApneaCount(self, data, peakDetectionMethod='f2', interPolKind='linear', countMethod='f1',
                           smoothingOrder=32 * 15, countOrder=32 * 25, visuallization=0):
        YnewMax, YnewMin = self.envelope1D(data, peakDetectionMethod=peakDetectionMethod, interPolKind=interPolKind)
        if countMethod == 'f1':
            smoothData = _smooth(YnewMax, window_len=smoothingOrder, modes='same')
            pksMax = signal.argrelmax(smoothData, order=countOrder)[0]
            pksMin = signal.argrelmin(smoothData, order=countOrder)[0]
            count = len(pksMin)
        elif countMethod == 'f2':
            smoothData = _smooth(YnewMax, s=smoothingOrder, modes='same')
            pksMax = np.vstack(peakdetect(smoothData, lookahead=countOrder)[0])[:, 0]
            pksMin = np.vstack(peakdetect(smoothData, lookahead=countOrder)[1])[:, 0]
            count = len(pksMin)
        else:
            print('peakDetectionMethod = ["f1", "f2"]')
            raise ValueError
        if visuallization:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.plot(data, label='ACC Raw Data')
            plt.plot(YnewMax, label='Envelope Data')
            plt.plot(smoothData, label='Smoothing Data')
            plt.plot(pksMin, smoothData[np.int64(pksMin)], 'o', label='Est Apnea Section')
            plt.grid()
            plt.legend()
        return count, pksMin

    def apneaCounting2(self, apneaIdxs, ACCrawData):
        totalApneaCountEnvelope = 0
        pksMins = []
        if len(apneaIdxs) == 0:
            pass
        else:
            for idxApnea in apneaIdxs:
                if idxApnea[0] == idxApnea[1]:
                    tmpACCrawData = ACCrawData[idxApnea[0]]
                else:
                    tmpACCrawData = np.concatenate(ACCrawData[idxApnea[0]:idxApnea[1]])
                tmpApneaCountEnvelope, pksMin = self.envelopeApneaCount(tmpACCrawData)
                if idxApnea[0] == 0:
                    if len(pksMin) == 0:
                        pass
                    else:
                        for tmpPksMin in pksMin:
                            pksMins.append(tmpPksMin)
                else:
                    pksMin += len(np.concatenate(ACCrawData[0:idxApnea[0]]))
                    for tmpPksMin in pksMin:
                        pksMins.append(tmpPksMin)
                totalApneaCountEnvelope += tmpApneaCountEnvelope

        return totalApneaCountEnvelope, pksMins

    def featureSmoothing(self, feature, s=1):
        """
        smooths a feature averages surround feature of number of s
        watch out train - test this functions output
        """
        dataPoint = len(feature)
        newFeature = np.zeros(feature.shape)
        for i in range(dataPoint):
            if (i < s):
                newFeature[i] = np.mean(feature[0:i + s + 1], 0)
                len(feature[0:i + s + 1])
            elif (i == (dataPoint - 1)):
                newFeature[i] = np.mean(feature[-(1 + s)::], 0)
            else:
                newFeature[i] = np.mean(feature[i - s:i + (s + 1)], 0)
        return newFeature

    def list2ndarrayVstack(self, data):
        newList = []
        for i in data:
            newList.append(np.vstack(i))
        return newList

    def zscoreTest(self, a, mns, sstd):
        return (a - mns) / sstd

    def smoothing(self, data):
        data = signal.savgol_filter(data, polyorder=3, window_length=33)
        data = signal.savgol_filter(data, polyorder=3, window_length=33)
        data = signal.savgol_filter(data, polyorder=3, window_length=33)
        return data

    def position2Idx(self, dataArray, PositionIdxVal=0):
        dataArray = np.array(dataArray)
        data = list(np.where(dataArray == PositionIdxVal)[0])
        PositonIdx = np.where(data == PositionIdxVal)[0]
        totalIdx = []
        idxList = []
        while not (len(data) == 0):
            tmpIdx = data.pop(0)
            if len(data) == 0:
                idxList.append(tmpIdx)
                totalIdx.append([min(idxList), max(idxList)])
                idxList = []
                return totalIdx
            elif (data[0] - tmpIdx == 1):
                idxList.append(tmpIdx)
            else:
                idxList.append(tmpIdx)
                totalIdx.append([min(idxList), max(idxList)])
                idxList = []
        return totalIdx

    def extractPositionIdx(self):
        ACCMAXMINx = []
        ACCMAXMINy = []
        ACCMAXMINz = []
        for tmpIdx in range(0, len(self.ACC), int(self.FsACC * self.positionDetectorSec)):
            tmpACC = self.ACC[tmpIdx:int(tmpIdx + int(self.FsACC * self.positionDetectorSec))]

            if len(tmpACC) < int(self.FsACC * self.positionDetectorSec):
                continue
            ACCMAXMINx.append(np.max(tmpACC[:, 0]) - np.min(tmpACC[:, 0]))
            ACCMAXMINy.append(np.max(tmpACC[:, 1]) - np.min(tmpACC[:, 1]))
            ACCMAXMINz.append(np.max(tmpACC[:, 2]) - np.min(tmpACC[:, 2]))
        MAXMINIDX = np.float32(np.array(ACCMAXMINx) >= self.positionDetectorThreshold) + np.float32(
            np.array(ACCMAXMINy) >= self.positionDetectorThreshold) + np.float32(
            np.array(ACCMAXMINz) >= self.positionDetectorThreshold)
        MAXMINIDX = np.array(MAXMINIDX) >= 1

        return MAXMINIDX

    def extractWaveLetFeature(self, data, wavelets, levels):
        """

        extraction of feature from waveleted signal
        wavelet function list --> pywt.wavelist()
        featureName = ['mean',
                       'std',
                       'entropy'
                       'kurtosis'
                       'skewness']
        """

        waveletData = wavedec(data, wavelet=wavelets, level=levels, mode='per')
        feature = []
        for idx, i in enumerate(waveletData):
            feature.append(np.mean(i))
            feature.append(np.std(i))
            feature.append(stats.kurtosis(i))
            feature.append(stats.skew(i))
            feature.append(stats.entropy(stats.norm.pdf(i, np.mean(i), np.std(i))))
        return np.array(feature)

    def featureExtraction(self, waveletFunction='sym3', waveLevel=8):
        totalFeatures = []
        ACCData = []

        for tmpIdx in range(0, len(self.ACC), int(self.FsACC * self.featureEpoch)):
            tmpACCData = self.ACC[tmpIdx:int(tmpIdx + self.FsACC * self.featureEpoch), 0]

            if len(tmpACCData) < int(self.FsACC * self.featureEpoch):
                continue
            tmpEDRipFeatures = self.extractWaveLetFeature(tmpACCData, waveletFunction, waveLevel)
            totalFeatures.append(tmpEDRipFeatures)
            ACCData.append(tmpACCData)
        return np.vstack(totalFeatures), ACCData

    def positionBasedWaveLetFeatures(self, position, epochSec=10):
        idxPosition = self.position2Idx(position)
        if len(idxPosition) == 0:
            self.logData.write('Artifact detection : no artifact' + '\n')
            tmpApneaObject = ApneaDetector(self.ECG, self.ACC,
                                           FsACC=self.FsACC, FsECG=self.FsECG,
                                           positionDetectSec=self.positionDetectorSec,
                                           positionDetectThreshold=self.positionDetectorThreshold,
                                           model=0,
                                           parameter=0,
                                           sleepModel=0,
                                           sleepParam=0,
                                           currentDir=self.currentDir)
            tmpFeatures, tmpACCData = tmpApneaObject.featureExtraction()
            self.logData.write('Feature Extraction : Finish' + '\n')
            return tmpFeatures, tmpACCData
        else:
            totalFeatures = []
            totalACCData = []
            for tmpIdx in idxPosition:
                if tmpIdx[1] - tmpIdx[0] < (int(60 / epochSec) + 1) or tmpIdx[1] - tmpIdx[0] < 0:
                    self.logData.write(
                        'Artifact detection : pass : ' + str(tmpIdx[0]) + ' finishIdx : ' + str(tmpIdx[1]) + '\n')
                else:
                    self.logData.write(
                        'Artifact detection : startIdx : ' + str(tmpIdx[0]) + ' finishIdx : ' + str(tmpIdx[1]) + '\n')
                    tmpApneaObject = ApneaDetector(
                        self.ECG[int(tmpIdx[0] * self.FsECG * epochSec):int(tmpIdx[1] * self.FsECG * epochSec)],
                        self.ACC[int(tmpIdx[0] * self.FsACC * epochSec):int(tmpIdx[1] * self.FsACC * epochSec)],
                        FsACC=self.FsACC, FsECG=self.FsECG,
                        positionDetectSec=self.positionDetectorSec,
                        positionDetectThreshold=self.positionDetectorThreshold,
                        model=0,
                        parameter=0,
                        sleepModel=0,
                        sleepParam=0,
                        currentDir=self.currentDir)
                    tmpFeatures, tmpACCData = tmpApneaObject.featureExtraction()
                    totalFeatures.append(tmpFeatures)
                    totalACCData.append(tmpACCData)
            self.logData.write('Feature Extraction : Finish' + '\n')
        return totalFeatures, totalACCData

    def StepCounterIdx2(self, data):
        epochidx = []
        cursor = 0
        while cursor < len(data):
            if data[cursor] == 1:
                if cursor == len(data) - 1:
                    epochidx.append([cursor, cursor])
                    break
                epochidxx = []
                epochidxx.append(cursor)
                while not (data[cursor + 1] == 0):
                    cursor += 1
                    if cursor == len(data) - 1:
                        epochidxx.append(cursor)
                        epochidx.append(epochidxx)
                        return epochidx
                epochidxx.append(cursor)
                epochidx.append(epochidxx)
                cursor += 1
            elif data[cursor] == 0:
                cursor += 1
        return epochidx

    def estimateTestDataPosition(self, testDataList, ACCrawDatasTest,
                                 optimalThreshold, optimalThresholdSleep, ROCcurveIdx=1, labelSmoothIdx=1,
                                 labelSmoothLength=1, featureSmoothingidx=1, featureSmoothLength=1, normalization=1):
        parameters = self.parameter
        ## data Len compare between ACC data and test data
        ACCList = ACCrawDatasTest
        testX = testDataList
        testDataLen = 0
        for tmpTestX in testX:
            testDataLen += len(tmpTestX)
        ACCDataLen = 0
        for tmpACCData in ACCList:
            ACCDataLen += len(tmpACCData)
        ## raise Error if dismatch between ACC data and test data
        if not (testDataLen == ACCDataLen):
            self.logData.write('Error : ACCData Len --> ' + str(testDataLen))
            self.logData.write(' , Data Len --> ' + str(ACCDataLen))
            self.logData.write(' size dismatch with ACC & features \n')
            raise ValueError

        if normalization:
            for tmpTestXIdx, tmpTestX in enumerate(testX):
                testX[tmpTestXIdx] = self.zscoreTest(tmpTestX, parameters['mean'], parameters['Std'])

        if featureSmoothingidx:
            for tmpTestXIdx, tmpTestX in enumerate(testX):
                testX[tmpTestXIdx] = self.featureSmoothing(tmpTestX, s=featureSmoothLength)

        tmpProp = []
        for tmpTestX in testX:
            tmpProp.append(self.model.predict_proba(tmpTestX))
        if labelSmoothIdx:
            for tmptmpPropIdx, tmptmpProp in enumerate(tmpProp):
                tmpProp[tmptmpPropIdx][:, 1] = self.featureSmoothing(tmptmpProp[:, 1], s=labelSmoothLength)

        tmpPredict = []
        if ROCcurveIdx:
            for tmptmpProp in tmpProp:
                tmpPredict.append(tmptmpProp[:, 1] >= optimalThreshold)
        else:
            for tmptmpProp in tmpProp:
                tmpPredict.append(tmptmpProp[:, 1] >= 0.5)
        ## apnea count
        apneaIdxss = []
        totalApneaCountEnvelopess = []
        pksMins = []
        tmpDataTime = []
        countIndexOffSet = 0
        for idxPredict in range(len(tmpPredict)):
            tmpDataTime.append(len(tmpPredict[idxPredict]))
            apneaIdxss.append(self.StepCounterIdx2(tmpPredict[idxPredict]))
            tmpACCList = ACCrawDatasTest[idxPredict]
            if len(apneaIdxss[idxPredict]) == 0:
                totalApneaCountEnvelope = 0
                totalApneaCountEnvelopess.append(totalApneaCountEnvelope)
                countIndexOffSet += len(np.concatenate(tmpACCList))
            else:
                totalApneaCountEnvelope, pksMin = self.apneaCounting2(apneaIdxss[idxPredict], tmpACCList)
                totalApneaCountEnvelopess.append(totalApneaCountEnvelope)
                if len(pksMin) == 0:
                    pass
                else:
                    pksMins.append(np.array(pksMin) + countIndexOffSet)
                countIndexOffSet += len(np.concatenate(tmpACCList))

        totalRecordTimeHour = self.recordTime / 3600
        AHIwithTST = np.sum(totalApneaCountEnvelopess) / totalRecordTimeHour
        # mean-TST
        AHIwithTSTwithMeanTST = AHIwithTST * (1 / 0.85294)
        # print(np.sum(totalApneaCountEnvelopess))
        # print(str(AHIwithTSTwithMeanTST))
        sleepEstimation, TST, SOL, WASO, SE = self.sleepClassification(optimalThresholdSleep)
        # print(TST)
        # print(SOL)
        # print(WASO)
        # print(SE)
        # print(str(((np.sum(totalApneaCountEnvelopess)) / (self.recordTime / (3600)) ) * (1 / 0.85294)))
        self.logData.write('Result : Total Sleep Time ' + str(np.sum(tmpDataTime)) + ' Min ' + '\n')
        self.logData.write(
            'Result : Apnea & Hypopnea Count ' + str(np.sum(totalApneaCountEnvelopess)) + ' Count' + '\n')
        self.logData.write('Result : Apnea & Hypopnea Index ' + str(
            np.sum(totalApneaCountEnvelopess) / (np.sum(tmpDataTime) / 60)) + ' /Hour' + '\n')
        self.logData.write('Result : Sleep Efficiency ' + str(
            (np.sum(tmpDataTime) / (len(self.ACC) / int(self.FsACC * self.featureEpoch))) * 100) + ' %' + '\n')
        self.logData.close()
        return tmpProp, pksMins, np.sum(totalApneaCountEnvelopess), np.sum(
            tmpDataTime), sleepEstimation, TST, SOL, WASO, SE, np.sum(totalApneaCountEnvelopess) / (
                           np.sum(tmpDataTime) / 60)

def extractionSleepLactancy(data, wakeLabel = 1):
    for tmpDataIdxm, tmpData in enumerate(data):
        if tmpDataIdxm ==0:
            if not(tmpData == wakeLabel):
                return 0
        else:
            if not (tmpData == wakeLabel):
                return tmpDataIdxm
    return 1

def calSleepParam(data, dataSOL):
    SOL = extractionSleepLactancy(dataSOL)
    data[0:SOL] = 1
    TST = len(data) - np.sum(data)
    WASO = len(data) - TST - SOL
    SE = 1 - (np.sum(data) / len(data))
    return TST, SOL, WASO, SE

def squarRoot(data):
    dataRow = data.shape[0]
    dataCol = data.shape[1]
    sumData = np.zeros([dataRow,])
    for colNum in range(dataCol):
        tmpData = data[:, colNum]**2
        sumData += tmpData
    return sumData**(1/2)

if __name__ == '__main__':
    print('apnea_sleep_detection main function')
