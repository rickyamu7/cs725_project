import ot
import polars as pl
import math as m
import numpy as np
from icecream import ic
import ot
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
pl.Config.set_fmt_str_lengths(100)
import optuna
import pickle
import time
import datetime
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
seed = 37
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.metrics import classification_report


seg_lim_Hz = [0, 125, 188 ,250, 750, 875, 938 ,1000, 1062 ,1125 ,1251, 1501, 1751, 1875, 1937 ,2000]
seg_lims = [int(a*10451/2000) for a in seg_lim_Hz]
ic(seg_lims)

def dfun(u, v)-> float:
    u = u/u.sum()
    v = v/v.sum()
    return ot.emd2_1d(u,v)*100

def get_distances(df):
    global seg_lims
    dist_m = {}
    get_seg = lambda x,y: list(range(x,y))
    freq_segments = [get_seg(seg_lims[i],seg_lims[i+1]) for i in range(len(seg_lims)-1)]
    for seg in freq_segments:
        X = df[:,seg]
        key = f'{seg[:1][0]/10451*2000:.0f}'
        dist_m[key] = squareform(pdist(X, dfun))
    return dist_m

def weighted_distance(dist_m:dict,weights:dict,p:float):
    s = 0
    for key in dist_m:
        s += (dist_m[key]**p)*weights[key]
    return s

def create_train_test_split(df, row_ind):
    random.seed(42) #donot change, if changed please recompute distance matrix
    f_codes = df['fault_code'].to_list()
    f_str = set([f[4:-7] for f in f_codes])
    index_fstr = {}
    fstr_index = {}
    val_list = []
    for f in f_str:
        index_fstr[f] = max([code[-6:-5] for code in f_codes if f in code])
        key = f"{f}@{index_fstr[f]}"
        fstr_index[key] = [i for i,code in enumerate(f_codes) if key in code]
        val_list += fstr_index[key]
    row_ind_5 = set(list(range(0, df.shape[0], 5)))
    val_set = (set(val_list)&row_ind_5)
    full_set = (set(list(range(df.shape[0])))&row_ind_5)
    train_list = list(full_set-val_set)
    val_list = list(val_set)
    test_list = random.sample([ind+2 for ind in val_list], int(len(val_list)*0.4))
    return train_list, val_list, test_list

def create_dataset(df, train_list, val_list, test_list):
    col_ind = list(range(10451))
    X_train = df[train_list, col_ind]
    X_val = df[val_list, col_ind]
    X_test = df[test_list, col_ind]

    train_list_y = df["fault_code"][train_list].to_list()
    y_train = [t[7:8] for t in train_list_y]
    val_list_y = df["fault_code"][val_list].to_list()
    y_val = [t[7:8] for t in val_list_y]
    test_list_y = df["fault_code"][test_list].to_list()
    y_test = [t[7:8] for t in test_list_y]

    ic(len(y_train), len(y_val))
    return(X_train, y_train, X_val, y_val, X_test, y_test)

def create_energy_dataset(row_ind ,df, train_list, val_list, test_list):
    global seg_lims
    train_ind = [row_ind.index(i) for i in train_list]
    val_ind = [row_ind.index(i) for i in val_list]
    test_ind = [row_ind.index(i) for i in test_list]
    row_ind_ = list(range(df.shape[0]))
    get_seg = lambda x,y: list(range(x,y))
    freq_segments = [get_seg(seg_lims[i],seg_lims[i+1]) for i in range(len(seg_lims)-1)]
    X = df.to_numpy()
    energy_ = np.zeros((len(df),len(seg_lims)-1))
    for i,seg in enumerate(freq_segments):
        energy_[:, i] = X[np.ix_(row_ind_,seg)].sum(axis=1)/1e14
    return (energy_[train_ind,:], energy_[val_ind,:], energy_[test_ind,:])


def create_dist_dataset(row_ind, train_list, val_list, test_list, X_dist):
    train_ind = [row_ind.index(i) for i in train_list]
    val_ind = [row_ind.index(i) for i in val_list]
    test_ind = [row_ind.index(i) for i in test_list]
    train_dist = {}
    val_dist = {}
    test_dist = {}
    for key in X_dist:
        train_dist[key] = X_dist[key][np.ix_(train_ind,train_ind)]
        val_dist[key] = X_dist[key][np.ix_(val_ind,train_ind)]
        test_dist[key] = X_dist[key][np.ix_(test_ind,train_ind)]
    return train_dist, val_dist, test_dist

def objective(trial,train_dist, val_dist, dataset, energy_dataset):
    global seg_lims
    X_train, y_train, X_val, y_val, X_test, y_test = dataset
    Xe_train, Xe_val, Xe_test = energy_dataset
    row_ind = list(range(0, df.shape[0], 5))
    k = trial.suggest_int("k",3,30)
    p = trial.suggest_float("p",0.5,2)
    C = trial.suggest_float("C", 0.1, 10, log=True)
    W_dict = {}
    keys = [f'{s/10451*2000:.0f}' for s in seg_lims[:-1]]
    for i,key in enumerate(keys):
        W_dict[key] = trial.suggest_float(f"w_{key}",0,1)
    w_sum = sum(list(W_dict.values()))
    W_dict_n = {key:W_dict[key]/w_sum for key in W_dict}
    neigh = KNeighborsClassifier(n_neighbors=k, metric= "precomputed", weights='distance')
    Train_dist = weighted_distance(train_dist, W_dict_n,p)
    Val_dist = weighted_distance(val_dist, W_dict_n,p)
    neigh.fit(Train_dist, y_train)
    y_pred_1 = neigh.predict_proba(Val_dist)

    Lr = LogisticRegression(random_state=0, C=C).fit(Xe_train,y_train)
    y_pred_2 = Lr.predict_proba(Xe_val)

    y_pred = np.argmax(y_pred_1+y_pred_2, axis=1)
    y_val = [int(x) for x in y_val]

    return f1_score(y_val,y_pred, average='micro')

if __name__ == "__main__":
    path = r"C:\Users\Sudhendu\Documents\IITB\0. Courses PhD\Foundations of Machine Learning\Project\Code_git_controlled\Data dump\gx_2.feather"
    df = pl.read_ipc(path)
    col_ind = list(range(10451))
    row_ind = list(range(0, df.shape[0], 5))
    train_list, val_list, test_list = create_train_test_split(df, row_ind)
    row_ind = row_ind+test_list
    ic(test_list[:10])
    X = df[row_ind, col_ind]
    ic(X)
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    # X_dist = get_distances(X)
    # with open('X_dist.pickle', 'wb') as handle:
    #     pickle.dump(X_dist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('X_dist.pickle', 'rb') as handle:
        X_dist = pickle.load(handle)
    
    # ic(len(train_list), len(val_list), len(test_list))
    dataset = create_dataset(df, train_list, val_list, test_list)
    train_dist, val_dist, test_dist = create_dist_dataset(row_ind, train_list, val_list,test_list, X_dist)
    energy_dataset = create_energy_dataset(row_ind,X, train_list, val_list,test_list)
    study = optuna.create_study(direction="maximize")  # Create a new study.
    n_trials = 100
    ic(n_trials)
    study.optimize(lambda trial: objective(trial, train_dist, val_dist, dataset, energy_dataset), n_trials=n_trials, n_jobs=-1)  # Invoke optimization of the objective function.
    best_params = study.best_params
    ic(best_params['k'], best_params['p'], best_params['C'])

    neigh = KNeighborsClassifier(n_neighbors=best_params['k'], metric= "precomputed", weights='distance')
    p = best_params['p']
    best_params.pop('k')
    best_params.pop('p')
    best_params.pop('C')

    W_dict = {f'{key[2:]}':best_params[key] for key in best_params}
    w_sum = sum(list(W_dict.values()))
    W_dict_n = {key:W_dict[key]/w_sum for key in W_dict}
    keys = sorted([int(key) for key in W_dict_n])
    for key in keys:
        print(f"{key:04d}: {W_dict_n[f'{key}']:.02f}")

    Train_dist = weighted_distance(train_dist, W_dict_n,p)
    Val_dist = weighted_distance(val_dist, W_dict_n,p)
    Test_dist = weighted_distance(test_dist, W_dict_n,p)
    X_train, y_train, X_val, y_val, X_test, y_test = dataset
    
    neigh.fit(Train_dist, y_train)
    y_val_pred = neigh.predict(Val_dist)
    y_test_pred = neigh.predict(Test_dist)
    ic(f1_score(y_val,y_val_pred, average='macro'), f1_score(y_test,y_test_pred, average='macro'))
    ic(accuracy_score(y_val,y_val_pred), accuracy_score(y_test,y_test_pred))
    print(classification_report(y_val,y_val_pred))
    conf = confusion_matrix(y_val,y_val_pred, normalize='all')
    ax = sns.heatmap(conf, annot=True, cmap="Blues")
    ax.set(xlabel="Predictions", ylabel="True Labels")
    plt.show()