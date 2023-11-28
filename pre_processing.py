import numpy as np
import scipy.signal as ss
import scipy.fft as fft
import pywt
import polars as pl
import math as m
import os
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
import itertools
import json
from icecream import ic

CPU_COUNT = os.cpu_count()

def find_sampling_freq(signal_df)->float:
    _del_t = signal_df[-1,'Timestamp'] - signal_df[0,'Timestamp']
    num_samples = signal_df.shape[0]
    return num_samples/_del_t

def delta_t(signal_df)->float:
    return signal_df[-1,'Timestamp'] - signal_df[0,'Timestamp']


def signal_metadata_fn(file_loc:str, files_list:list)->tuple:
    remove_ext = lambda x: x[:-4]
    signal_dict = {}
    signal_len = []
    signal_freq = []
    signal_time = []
    for f in files_list:
        # print(f)
        signal_dict[remove_ext(f)] = {}
        signal_dict[remove_ext(f)]['signal']  = pl.read_csv(os.path.join(file_loc, f))[4000:-4000]
        signal_dict[remove_ext(f)]['length']  = signal_dict[remove_ext(f)]['signal'].shape[0]
        signal_dict[remove_ext(f)]['s_freq']  = find_sampling_freq(signal_dict[remove_ext(f)]['signal'])
        signal_dict[remove_ext(f)]['sig_dur'] = delta_t(signal_dict[remove_ext(f)]['signal'])
        signal_len.append(signal_dict[remove_ext(f)]['length'])
        signal_freq.append(signal_dict[remove_ext(f)]['s_freq'])
        signal_time.append(signal_dict[remove_ext(f)]['sig_dur'])
        signal_dict[remove_ext(f)].pop('signal')
    signal_len = np.asarray(signal_len)
    signal_freq = np.asarray(signal_freq)
    signal_time = np.asarray(signal_time)

    min_signal_len = min(signal_len)
    min_signal_freq = min(signal_freq)
    
    ic(signal_time.max(), signal_time.min())
    assert np.abs(signal_time.max() - signal_time.min())< 12

    ic((max(signal_time)-min(signal_time))/signal_time.mean())
    assert (max(signal_time)-min(signal_time))/signal_time.mean() < 0.2

    ic((max(signal_len)-min(signal_len))/signal_len.mean())
    assert (max(signal_len)-min(signal_len))/signal_len.mean() < 0.2

    ic((max(signal_freq)-min(signal_freq))/signal_freq.mean())
    assert (max(signal_freq)-min(signal_freq))/signal_freq.mean() < 0.2
    
    return (signal_dict, min_signal_len, min_signal_freq)

def generate_components(signal_df):
    sig_components = {}
    component_map = {'ax':'0', 'ay':'1', 'az':'2' ,'gx':'3', 'gy':'4', 'gz':'5' }
    for key in component_map:
        sig_components[key] = signal_df[:,component_map[key]].to_numpy().reshape(-1)
    return sig_components

def bandpass(raw_sig:dict, min_sample_freq:float, fs:float, signal_type:str='vib', )->dict:
    if signal_type == 'vib':
        sos = ss.butter(1, Wn=(0.5, min_sample_freq/2-1), btype='bandpass', analog=False, output='sos', fs=fs)
    filt_sig = {}
    for key in raw_sig:
        filt_sig[key] = ss.sosfilt(sos, raw_sig[key])
    return filt_sig

def make_sig_len_equal(filt_sig:dict, min_signal_len:int)->dict:
    filt_cut_sig = {}
    for key in filt_sig:
        filt_cut_sig[key] = filt_sig[key][:min_signal_len]
    return filt_cut_sig

def cut_in_10(filt_cut_sig:dict)->dict:
    usable_sig ={}
    for key in filt_cut_sig:
        s1 = filt_cut_sig[key]
        for i in range(10):
            usable_sig[f"{key}_{i:02d}"] = s1[i*m.floor(len(s1)/10):(i+1)*m.floor(len(s1)/10)]
    return usable_sig

def calc_fourier_transform(usable_sig:dict, freq_vals, fs:float)->dict:
    fourier_reg = {}
    for key in usable_sig:
        _fft  = np.abs(fft.rfft(usable_sig[key]))
        _fft_freq = fft.rfftfreq(usable_sig[key].size, d =1/fs)
        
        fourier_reg[key] = np.interp(freq_vals, _fft_freq, _fft)
    return fourier_reg

def get_fft(file_name:str, file_loc:str, min_sample_freq:float, freq_vals:list, fs:float)->tuple:
    fft_array = np.zeros(len(freq_vals))
    remove_ext = lambda x: x[:-4]
    f = file_name
    signal_demo = pl.read_csv(os.path.join(file_loc, f))[4000:-4000]
    signal_str = remove_ext(f)
    raw_sig = generate_components(signal_demo)
    filt_sig = bandpass(raw_sig, min_sample_freq, fs=fs, signal_type='vib')
    usable_sig = cut_in_10(filt_sig)
    fft_calc = calc_fourier_transform(usable_sig, freq_vals, fs=fs)
    return (fft_calc,remove_ext(f))        

def create_df_dict(min_sample_freq:float, min_signal_len:int, freq_vals:list)->dict:
    col_names = [f"f_{i:07.3f}" for i in freq_vals]
    dummy = np.zeros(len(col_names))+0.1
    col_dummy_dict = dict(zip(col_names, dummy))
    col_dummy_dict['fault_code']  ='some_fault'
    col_dummy_dict['segment_num']  ='random_seg_num'
    component_map = {'ax':'0', 'ay':'1', 'az':'2' ,'gx':'3', 'gy':'4', 'gz':'5' }

    df_dict = {}
    for key in component_map: 
        df_dict[key] = pl.DataFrame(data=col_dummy_dict)
        df_dict[key] = df_dict[key].drop("segment_num").insert_at_idx(0, df_dict[key].get_column("segment_num"))
        df_dict[key] = df_dict[key].drop("fault_code").insert_at_idx(0, df_dict[key].get_column("fault_code"))
    return df_dict


def add_fft_to_df(df_dict:dict, fft_calc:dict, fault_code:str)-> tuple:
    component_map = {'ax':'0', 'ay':'1', 'az':'2' ,'gx':'3', 'gy':'4', 'gz':'5' }
    freq_cols = df_dict['ax'].columns[2:]
    
    for key in fft_calc:
        data_dict = dict(zip(freq_cols, fft_calc[key]))
        data_dict['segment_num'] = key[3:]
        data_dict['fault_code'] = fault_code
        new_rows = pl.DataFrame(data_dict)
        new_rows = new_rows.drop("segment_num").insert_at_idx(0, new_rows.get_column("segment_num"))
        new_rows = new_rows.drop("fault_code").insert_at_idx(0, new_rows.get_column("fault_code"))
        df_dict[key[:2]] = pl.concat([df_dict[key[:2]], new_rows])
    return df_dict

def write_to_files(df_dict:dict)-> None:
    write_path = r"C:\Users\Sudhendu\Documents\IITB\0. Courses PhD\Fundamentals of Machine Learning\Project\Code_git_controlled\Data dump"
    for key in df_dict:
        df_dict[key].write_ipc(os.path.join(write_path, f"{key}.feather"))

def read_from_files(path:str)-> dict:
    feather_files = os.listdir(path)
    df_dict = {}
    for key in feather_files:
        df_dict[key[0:2]] = pl.read_ipc(os.path.join(path, key))
    
    return df_dict


if __name__ == "__main__":
    file_loc = r"C:\Users\Sudhendu\Documents\IITB\0. Courses PhD\Fundamentals of Machine Learning\Project\categorized_data_updated\VIB_on"
    files_list_all = os.listdir(path=file_loc)
    bad_files_no_ext = ['VIB_0000001@2@0_on', 'VIB_0000001@2@2_on', 'VIB_0000200@0@2_on', 'VIB_0000300@0@3_on', 'VIB_0002000@1@0_on', 'VIB_0002200@0@1_on', 'VIB_0002300@0@2_on', 'VIB_0020000@2@2_on', 'VIB_0021000@1@3_on', 'VIB_0021000@2@2_on', 'VIB_0031000@1@1_on', 'VIB_0032000@1@2_on', 'VIB_0201000@1@0_on', 'VIB_0201000@1@1_on', 'VIB_0201000@2@4_on', 'VIB_0300000@1@1_on', 'VIB_0300000@1@3_on', 'VIB_0300000@1@4_on', 'VIB_1000000@0@0_on', 'VIB_2000000@1@3_on', 'VIB_2002000@1@0_on']
    bad_files = [f"{x}.csv" for x in bad_files_no_ext]
    files_list = [x for x in files_list_all if x not in bad_files]

    ic(len(files_list))
    signal_metadata, min_signal_len, min_sample_freq= signal_metadata_fn(file_loc=file_loc, files_list=files_list)
    with open('file.txt', 'w') as f_json:
        f_json.write(json.dumps(signal_metadata))
    ic(signal_metadata, min_sample_freq, min_signal_len)
    freq_vals = np.linspace(0,2000,int(min_signal_len/20))
    df_dict = create_df_dict(min_sample_freq, min_signal_len, freq_vals)

    remove_ext = lambda x: x[:-4]
    number_files = len(files_list)
    step = max(m.ceil(number_files/10),1)
    for j in range(0, number_files,step):
        ic(j, number_files, step )
        if j!=0: 
            df_dict=read_from_files(r"C:\Users\Sudhendu\Documents\IITB\0. Courses PhD\Fundamentals of Machine Learning\Project\Code_git_controlled\Data dump")
        with Pool(CPU_COUNT) as pool:
            args = [(i, file_loc, min_sample_freq, freq_vals, signal_metadata[remove_ext(i)]['s_freq']) for i in files_list[j:min(j+step,number_files)]]
            for result in pool.starmap(get_fft, args, chunksize=5):
                _data_dict, fault_code = result
                df_dict = add_fft_to_df(df_dict, _data_dict, fault_code)
        write_to_files(df_dict)
        ic("writing done")
    df_ult = read_from_files(path = r"C:\Users\Sudhendu\Documents\IITB\0. Courses PhD\Fundamentals of Machine Learning\Project\Code_git_controlled\Data dump")
    ic(df_ult)