import os
import numpy as np
import pandas as pd
import mne
from tqdm import tqdm

pd.options.display.float_format = '{:.9f}'.format

raw_path = '/workspace/Competition/PSG/01_DATA/raw'
raw_files = os.listdir(raw_path)
raw_files = [x for x in raw_files if not x.startswith('.')]
tsv_files = [x for x in raw_files if 'tsv' in x]
edf_files = [x for x in raw_files if 'edf' in x]
ids = [x.split('.')[0] for x in tsv_files]

master_channels = ['EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2']

### 1channel - full numpy ###
'''
for rec in tqdm(edf_files):
    savename = rec.split('.')[0]
    edfpath = os.path.join(raw_path, rec)
    rawedf = mne.io.read_raw_edf(edfpath, verbose=False)
    edfchannels = rawedf.ch_names
    edfchannels.sort()
    for eeg in edfchannels:
        #raw_eeg = rawedf[eeg][0]
        if eeg.upper() == 'EEG F3-M2':
            raw_eeg = rawedf[eeg][0]
            np.save('/workspace/Competition/PSG/01_DATA/eeg_f3m2/fullnpy/'+savename,raw_eeg)
        elif eeg.upper() == 'EEG C3-M2':
            raw_eeg = rawedf[eeg][0]
            np.save('/workspace/Competition/PSG/01_DATA/eeg_c3m2/fullnpy/'+savename,raw_eeg)
        elif eeg.upper() == 'EEG O1-M2':
            raw_eeg = rawedf[eeg][0]
            np.save('/workspace/Competition/PSG/01_DATA/eeg_o1m2/fullnpy/'+savename,raw_eeg)
'''        
### Make 30 second splits
'''
for tfile in tqdm(tsv_files):
    recid = tfile.split('.')[0]
    tdata = pd.read_csv(os.path.join(raw_path, tfile), delimiter='\t')
    #process dataframe
    subdf = tdata[tdata['description'].str.contains('stage')].reset_index(drop=True)
    stages = list(subdf['description'])
    stages = [x[-1] for x in stages]
    onsets = list(subdf['onset'])
    durations = list(subdf['duration'])
    # load arrays
    eeg1_array = np.load(os.path.join('/workspace/Competition/PSG/01_DATA/eeg_f3m2/fullnpy',recid+'.npy'))
    eeg2_array = np.load(os.path.join('/workspace/Competition/PSG/01_DATA/eeg_c3m2/fullnpy',recid+'.npy'))
    eeg3_array = np.load(os.path.join('/workspace/Competition/PSG/01_DATA/eeg_o1m2/fullnpy',recid+'.npy'))
    # process array and save
    wcnt = 0
    rcnt = 0
    n1cnt = 0
    n2cnt = 0
    n3cnt = 0
    for idx,stg in enumerate(stages):
        dur = durations[idx]
        if (stg in ['W','1','2','3','R'])&(dur==30):
            startind = int(onsets[idx]*256)
            endind = startind + int(dur*256)
            sub1array = eeg1_array[:,np.arange(startind,endind)]
            sub2array = eeg2_array[:,np.arange(startind,endind)]
            sub3array = eeg3_array[:,np.arange(startind,endind)]
            if stg == 'W':
                savename = recid+'_W_'+str(wcnt)
                wcnt+=1
            elif stg == '1':
                savename = recid+'_N1_'+str(n1cnt)
                n1cnt+=1
            elif stg == '2':
                savename = recid+'_N2_'+str(n2cnt)
                n2cnt+=1
            elif stg == '3':
                savename = recid+'_N3_'+str(n3cnt)
                n3cnt+=1
            elif stg == 'R':
                savename = recid+'_R_'+str(rcnt)
                rcnt+=1
            np.save('/workspace/Competition/PSG/01_DATA/eeg_f3m2/split_30/'+savename, sub1array)
            np.save('/workspace/Competition/PSG/01_DATA/eeg_c3m2/split_30/'+savename, sub2array)
            np.save('/workspace/Competition/PSG/01_DATA/eeg_o1m2/split_30/'+savename, sub3array)
'''    
###

### 3 channel - make 90 second splits

for tfile in tqdm(tsv_files):
    #load data
    recid = tfile.split('.')[0]
    tdata = pd.read_csv(os.path.join(raw_path, tfile), delimiter='\t')
    #process dataframe
    subdf = tdata[tdata['description'].str.contains('stage')].reset_index(drop=True)
    stages = list(subdf['description'])
    stages = [x[-1] for x in stages]
    onsets = list(subdf['onset'])
    durations = list(subdf['duration'])
    # load array
    eeg1_array = np.load(os.path.join('/workspace/Competition/PSG/01_DATA/eeg_f3m2/fullnpy',recid+'.npy'))
    eeg2_array = np.load(os.path.join('/workspace/Competition/PSG/01_DATA/eeg_c3m2/fullnpy',recid+'.npy'))
    eeg3_array = np.load(os.path.join('/workspace/Competition/PSG/01_DATA/eeg_o1m2/fullnpy',recid+'.npy'))
    # process array and save
    wcnt = 0
    rcnt = 0
    n1cnt = 0
    n2cnt = 0
    n3cnt = 0
    for i in range(len(subdf)-3): # exclude very end
        # Make sure they are consecutive
        time1 = onsets[i]
        time2 = onsets[i+1]
        time3 = onsets[i+2]
        isConsecutive = (time1+30==time2)&(time2+30==time3)
        # Make sure last stage isn't ?
        lstage = stages[i+2]
        validStage = lstage in ['W','1','2','3','R']
        # Now subset the array
        if isConsecutive & validStage:
            startind = int(time1*256)
            endind = startind + 90*256
            sub1array = eeg1_array[:,np.arange(startind,endind)]
            sub2array = eeg2_array[:,np.arange(startind,endind)]
            sub3array = eeg3_array[:,np.arange(startind,endind)]
            if lstage == 'W':
                savename = recid+'_W_'+str(wcnt)
                wcnt+=1
            elif lstage == '1':
                savename = recid+'_N1_'+str(n1cnt)
                n1cnt+=1
            elif lstage == '2':
                savename = recid+'_N2_'+str(n2cnt)
                n2cnt+=1
            elif lstage == '3':
                savename = recid+'_N3_'+str(n3cnt)
                n3cnt+=1
            elif lstage == 'R':
                savename = recid+'_R_'+str(rcnt)
                rcnt+=1
            np.save('/workspace/Competition/PSG/01_DATA/eeg_f3m2/split_90/'+savename, sub1array)
            np.save('/workspace/Competition/PSG/01_DATA/eeg_c3m2/split_90/'+savename, sub2array)
            np.save('/workspace/Competition/PSG/01_DATA/eeg_o1m2/split_90/'+savename, sub3array)
        

