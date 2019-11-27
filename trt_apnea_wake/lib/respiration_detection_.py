import numpy as np
from scipy import interpolate
import pandas as pd
import biosppy

def discrete_to_continuous(values, value_times, sampling_rate=1000):
    """
    3rd order spline interpolation.

    Parameters
    ----------
    values : dataframe
        Values.
    value_times : list
        Time indices of values.
    sampling_rate : int
        Sampling rate (samples/second).

    Returns
    ----------
    signal : pd.Series
        An array containing the values indexed by time.

    Example
    ----------
    >>> import neurokit as nk
    >>> signal = discrete_to_continuous([4, 5, 1, 2], [1, 2, 3, 4], sampling_rate=1000)
    >>> pd.Series(signal).plot()

    Notes
    ----------
    *Authors*

    - `Dominique Makowski <https://dominiquemakowski.github.io/>`_

    *Dependencies*

    - scipy
    - pandas
    """
#    values=RRis.copy()
#    value_times=beats_times.copy()
    # Preprocessing
    initial_index = value_times[0]
    value_times = np.array(value_times) - initial_index

    # fit a 3rd degree spline on the data.
    spline = interpolate.splrep(x=value_times, y=values, k=3, s=0)  # s=0 guarantees that it will pass through ALL the given points
    x = np.arange(0, value_times[-1], 1)
    # Get the values indexed per time
    signal = interpolate.splev(x=x, tck=spline, der=0)
    # Transform to series
    signal = pd.Series(signal)
    signal.index = np.array(np.arange(initial_index, initial_index+len(signal), 1))

    return(signal)

def rsp_find_cycles(signal):
    """
    Find Respiratory cycles onsets, durations and phases.

    Parameters
    ----------
    signal : list or array
        Respiratory (RSP) signal (preferably filtered).


    Returns
    ----------
    rsp_cycles : dict
        RSP cycles features.

    Example
    ----------
    >>> import neurokit as nk
    >>> rsp_cycles = nk.rsp_find_cycles(signal)

    Notes
    ----------
    *Authors*

    - Dominique Makowski (https://github.com/DominiqueMakowski)

    *Dependencies*

    - biosppy

    *See Also*

    - BioSPPY: https://github.com/PIA-Group/BioSPPy

    """
    # Compute gradient (sort of derivative)
    gradient = np.gradient(signal)
    # Find zero-crossings
    zeros, = biosppy.tools.zero_cross(signal=gradient, detrend=True)

    # Find respiratory phases
    phases_indices = []
    for i in zeros:
        if gradient[i+1] > gradient[i-1]:
            phases_indices.append("Inspiration")
        else:
            phases_indices.append("Expiration")

    # Select cycles (inspiration) and expiration onsets
    inspiration_onsets = []
    expiration_onsets = []
    for index, onset in enumerate(zeros):
        if phases_indices[index] == "Inspiration":
            inspiration_onsets.append(onset)
        if phases_indices[index] == "Expiration":
            expiration_onsets.append(onset)


    # Create a continuous inspiration signal
    # ---------------------------------------
    # Find initial phase
    if phases_indices[0] == "Inspiration":
        phase = "Expiration"
    else:
        phase = "Inspiration"

    inspiration = []
    phase_counter = 0
    for i, value in enumerate(signal):
        if i == zeros[phase_counter]:
            phase = phases_indices[phase_counter]
            if phase_counter < len(zeros)-1:
                phase_counter += 1
        inspiration.append(phase)

    # Find last phase
    if phases_indices[len(phases_indices)-1] == "Inspiration":
        last_phase = "Expiration"
    else:
        last_phase = "Inspiration"
    inspiration = np.array(inspiration)
    inspiration[max(zeros):] = last_phase

    # Convert to binary
    inspiration[inspiration == "Inspiration"] = 1
    inspiration[inspiration == "Expiration"] = 0
    inspiration = pd.to_numeric(inspiration)

    cycles_length = np.diff(inspiration_onsets)

    rsp_cycles = {"RSP_Inspiration": inspiration,
                  "RSP_Expiration_Onsets": expiration_onsets,
                  "RSP_Cycles_Onsets": inspiration_onsets,
                  "RSP_Cycles_Length": cycles_length}

    return(rsp_cycles)

def rsp_process(rsp, sampling_rate=1000):
    """
    Automated processing of RSP signals.

    Parameters
    ----------
    rsp : list or array
        Respiratory (RSP) signal array.
    sampling_rate : int
        Sampling rate (samples/second).

    Returns
    ----------
    processed_rsp : dict
        Dict containing processed RSP features.

        Contains the RSP raw signal, the filtered signal, the respiratory cycles onsets, and respiratory phases (inspirations and expirations).

    Example
    ----------
    >>> import neurokit as nk
    >>>
    >>> processed_rsp = nk.rsp_process(rsp_signal)

    Notes
    ----------
    *Authors*

    - Dominique Makowski (https://github.com/DominiqueMakowski)

    *Dependencies*

    - biosppy
    - numpy
    - pandas

    *See Also*

    - BioSPPY: https://github.com/PIA-Group/BioSPPy
    """
    processed_rsp = {"df": pd.DataFrame({"RSP_Raw": np.array(rsp)})}

    biosppy_rsp = dict(biosppy.signals.resp.resp(rsp, sampling_rate=sampling_rate, show=False))
    processed_rsp["df"]["RSP_Filtered"] = biosppy_rsp["filtered"]


#   RSP Rate
#   ============
    rsp_rate = biosppy_rsp["resp_rate"]*60  # Get RSP rate value (in cycles per minute)
    rsp_times = biosppy_rsp["resp_rate_ts"]   # the time (in sec) of each rsp rate value
    rsp_times = np.round(rsp_times*sampling_rate).astype(int)  # Convert to timepoints
    try:
        rsp_rate = discrete_to_continuous(rsp_rate, rsp_times, sampling_rate)  # Interpolation using 3rd order spline
        processed_rsp["df"]["RSP_Rate"] = rsp_rate
    except TypeError:
        # print("NeuroKit Warning: rsp_process(): Sequence too short to compute respiratory rate.")
        processed_rsp["df"]["RSP_Rate"] = np.nan


#   RSP Cycles
#   ===========================
    rsp_cycles = rsp_find_cycles(biosppy_rsp["filtered"])
    processed_rsp["df"]["RSP_Inspiration"] = rsp_cycles["RSP_Inspiration"]

    processed_rsp["RSP"] = {}
    processed_rsp["RSP"]["Cycles_Onsets"] = rsp_cycles["RSP_Cycles_Onsets"]
    processed_rsp["RSP"]["Expiration_Onsets"] = rsp_cycles["RSP_Expiration_Onsets"]
    processed_rsp["RSP"]["Cycles_Length"] = rsp_cycles["RSP_Cycles_Length"]/sampling_rate

#   RSP Variability
#   ===========================
    rsp_diff = processed_rsp["RSP"]["Cycles_Length"]

    processed_rsp["RSP"]["Respiratory_Variability"] = {}
    processed_rsp["RSP"]["Respiratory_Variability"]["RSPV_SD"] = np.std(rsp_diff)
    processed_rsp["RSP"]["Respiratory_Variability"]["RSPV_RMSSD"] = np.sqrt(np.mean(rsp_diff ** 2))
    processed_rsp["RSP"]["Respiratory_Variability"]["RSPV_RMSSD_Log"] = np.log(processed_rsp["RSP"]["Respiratory_Variability"]["RSPV_RMSSD"])


    return(processed_rsp)

if __name__ == '__main__':
    print('main')