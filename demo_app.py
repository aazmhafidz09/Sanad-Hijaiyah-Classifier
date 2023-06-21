from tkinter import *
from tkinter import ttk, filedialog
import xgboost
import numpy as np
import pygame
import pandas as pd
import scipy.io.wavfile
from scipy.signal import find_peaks
import spafe.utils.vis as vis
from spafe.features.mfcc import mfcc
from spafe.features.rplp import rplp
from spafe.features.lpc import lpc
from spafe.utils.preprocessing import SlidingWindow
import pickle, wave
import pywt

# model_path = '//Users//aazmhafidzazis//Documents//Semhas//Model//'
# sound_path = '//Users//aazmhafidzazis//Documents//Semhas//Suara//'
# data_path = '//Users//aazmhafidzazis//Documents//Semhas//Ekstrak//'

model_path = "C:\\TA Aaz\\Sidang Tesis\\Semhas\\Model\\"
sound_path = 'C:\\TA Aaz\\Sidang Tesis\\Semhas\\Suara\\'
data_path = 'C:\\TA Aaz\\Sidang Tesis\\Semhas\\Ekstrak\\'


OPTIONS = {
    'Alif' : 0,
    'Ba' : 1,
    'Ta' : 2,
    'Tsa' : 3,
    'Jim' : 4,
    'ha' : 5,
    'Kho' : 6,
    'Dal' : 7,
    'Dzal' : 8,
    'Ro' : 9,
    'Zay' : 10,
    'Sin' : 11,
    'Syin' : 12,
    'Shod' : 13,
    'Dhod' : 14,
    'Tho' : 15,
    'Zho' : 16,
    'Ghain' : 17,
    'Fa' : 18,
    'Qof' : 19,
    'Kaf' : 20,
    'Lam' : 21,
    'Mim' : 22,
    'Nun' : 23,
    'Waw' : 24,
    'HA' : 25,
    'Ya' : 26
    }

index_sifat=[
    'S1',
    'S2',
    'S3',
    'S4',
    'S5',
    'T1',
    'T2',
    'T3',
    'T4',
    'T5',
    'T6',
    'T7'
    ]

information_dict = {
    'S1': {
        0 : 'Jahr',
        1 : 'Hams',
        'ekstrak': 'mfcc',
        'model': 'cnn'
    },
    'S2': {
        0 : 'Rakhawah',
        1 : 'Bayniyyah',
        2 : 'Syiddah',
        'ekstrak': 'mfcc',
        'model': 'cnn'
    },
    'S3': {
        0 : 'Istifal',
        1 : "Isti'la",
        'ekstrak': 'rasta',
        'model': 'xgboost'
    },
    'S4': {
        0 : 'Infitah',
        1 : 'Ithbaq',
        'ekstrak': 'mfcc',
        'model': 'cnn'
    },
    'S5': {
        0 : 'Ishmat',
        1 : 'Idzlaq',
        'ekstrak': 'mfcc',
        'model': 'cnn'
    },
    'T1': {
        0 : 'Tidak Shafir',
        1 : 'Shafir',
        'ekstrak': 'rasta',
        'model': 'cnn'
    },
    'T2': {
        0 : 'Tidak Qalqalah',
        1 : 'Qalqalah',
        'ekstrak': 'mfcc',
        'model': 'cnn'
    },
    'T3': {
        0 : 'Tidak Liin',
        1 : 'Liin',
        'ekstrak': 'rasta',
        'model': 'cnn'
    },
    'T4': {
        0 : 'Tidak Inhirah',
        1 : 'Inhirah',
        'ekstrak': 'mfcc',
        'model': 'cnn'
    },
    'T5': {
        0 : 'Tidak Takrir',
        1 : 'Takrir',
        'ekstrak': 'mfcc',
        'model': 'cnn'
    },
    'T6': {
        0 : 'Tidak Tafafsysyi',
        1 : 'Tafasysyi',
        'ekstrak': 'mfcc',
        'model': 'cnn'
    },
    'T7': {
        0 : 'Tidak Istithaalah',
        1 : 'Istithaalah',
        'ekstrak': 'rasta',
        'model': 'xgboost'
    }
}

pengelompokan = np.array([
    ## -------- huruf 1 (alif) ----------------------------------- huruf 28 (ya)----------##
    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], ## sifat 1 (S1) [0]Jahr    -   [1]Hams
    [2, 2, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 2, 2, 1, 1, 1, 0, 0, 0], ##     | [0]Rakhawah-   [1]Bayniyyah-   [2]Syiddah
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ,0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], ##     | [0]Istifal -   [1]Isti'la
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ##     | [0]Infitah -   [1]Ithbaq
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0 ,0, 0], ##     | [0]Ishmat  -   [1]Idzlaq 
                                                                                          ##     sampai di sini, ada 11 sifat 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ##     | Shafir (huruf yang berdesis)    [0] tidak ada [1] ada
    [0, 1, 0, 0, 1, 0, 0, 1, 0 ,0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], ##     | Qalqalah (memantul)             [0] tidak ada [1] ada 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], ##     | Liin (lembut)                   [0] tidak ada [1] ada
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], ##     | Inhirah (menyimpangnya makhraj) [0] tidak ada [1] ada
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ##     | Takrir (berulang)               [0] tidak ada [1] ada
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ##     | Tafasysyi (udara yang berhembus deras di dalam mulut) [0] tidak ada [1] ada
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])## Sifat 12 Istithaalah (memanjang)    [0] tidak ada [1] ada

def get_suara(data):
    data_suara_raw = wave.open(data, 'r')
    fs = data_suara_raw.getframerate()   #pengambilan frame suara per detiknya

    data_suara = data_suara_raw.readframes(-1) # ubah data dari wav ke binary
    data_suara_raw.close()
    data_suara = np.frombuffer(data_suara, np.int16) #ubah data dari binary ke int16

    max_data_suara = max(abs(data_suara))
    data_suara = data_suara/max_data_suara
    
    return data_suara, fs, max_data_suara

def non_overlapping_frame(data_suara, fs, ms):
    ## non-overlapping frame
    n = int(fs/1000*ms) ## 20ms = 882 data
    suara_len = len(data_suara)
    frame_G_N = int(suara_len/n) #banyaknya frame
    if suara_len % n:
        frame_G = np.zeros([n,frame_G_N+1])
        ## frame terakhir kode terpisah
        frame_G_NZ = suara_len-frame_G_N*n
        frame_G[:frame_G_NZ,frame_G_N] = [j for j in data_suara[frame_G_N*n:suara_len]]
    else:
        frame_G = np.zeros([n,frame_G_N])
    for i in range(frame_G_N):
        frame_G[:,i] =  data_suara[i*n:i*n+n]
    frame_G_N = len(frame_G[0,:])
    
    return frame_G, frame_G_N, n

def spectral_centroid(x, fs, b):
    centroid = []
    for i in range(b):
        sum_x = np.sum(x[:,i])
        if sum_x == 0: #jika frame isinya kosong/zeros
            centroid.append(0)
        else:
            window = x[:,i] #* hamming
            magnitudes = np.abs(np.fft.rfft(window)) # magnitudes of positive frequencies
            freqs = np.abs(np.fft.rfftfreq(len(window), 1.0/fs)) # positive frequencies
            centroid.append((np.sum(freqs * (magnitudes / np.sum(magnitudes)))) / (fs/2)) # return weighted mean
    return np.array(centroid)

def fejer_korovkin():
    ## mendefinisikan wavelet Fejer-Korovkin 6
    dec_lo = [0.0406258144232379, -0.0771777574069701, -0.146438681272577, 0.356369511070187, 0.812919643136907, 0.42791503242231]
    dec_hi = [-0.42791503242231, 0.812919643136907, -0.356369511070187, -0.146438681272577, 0.0771777574069701, 0.0406258144232379]
    rec_lo = [0.42791503242231, 0.812919643136907, 0.356369511070187, -0.146438681272577, -0.0771777574069701, 0.0406258144232379]
    rec_hi = [0.0406258144232379, 0.0771777574069701, -0.146438681272577, -0.356369511070187, 0.812919643136907, -0.42791503242231]
    filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi]
    wavelet = pywt.Wavelet(name="fk6",filter_bank=filter_bank)
    return wavelet

def wavelet_denoising(data_suara, wavelet, fs, flags_and):
    
    ############
    def thresholding_minimaxi(a,b):
        x = len(a)
        if any(not i for i in b):
            c = np.empty((1,0))
            for i in range(len(b)):
                if not b[i]:
                    c = np.append(c,a[i])
            # max_a = max(abs(a))
            # if lev < 11:
            x_var = np.median(abs(c))/0.6745 # kalkulasi nilai noise variance
            # else:
            #     x_var = 1/10**(  24  /10) # noise 30 dB
                # x_var = b
            mnmx = x_var * (0.3936 + 0.1892 * np.log2(x))
        else:
            mnmx = 0
        return pywt.threshold(a, mnmx)

    def wavelet_D(a, wavelet, lev):
        wav = pywt.wavedec(a, wavelet, level=lev)
        wav_f = pywt.wavedec(flags_and, wavelet, level=lev)
        # skala = 1
        # if skala0 > 0.5:
        #     skala = (skala-skala0)*2
        for i in range(lev+1):
            wav[i] = thresholding_minimaxi(wav[i],wav_f[i])
        return pywt.waverec(wav, wavelet)
        # wav_Ai, wav_Di = pywt.dwt(a, wavelet)
        # if lev > 0:
        #     wav_Ai = wavelet_D(wav_Ai, wavelet, lev-1)
        #     wav_Di = wavelet_D(wav_Di, wavelet, lev-1)
        # thre = thresholding_minimaxi(wav_Di, lev)
        # wavT_D = pywt.idwt(wav_Ai, thre, wavelet)
        # return wavT_D
    ############
    
    batas_lev = [19, 33, 61, 117, 229, 453, 901, 1797, 3589, 7173]
    len_batas = len(batas_lev)
    for i in range(len_batas):
        if len(data_suara) > batas_lev[-1*i-1]:
            level = len_batas-i
            break
    
    wavelet_data = data_suara
    idwt = wavelet_D(wavelet_data, wavelet, lev=level)
    
    return idwt #Inverse Discrete Wavelet Transform (IDWT)

def proses(alamat, b):
    data_suara, fs, scale_factor = get_suara(alamat)
    frame_G, frame_G_N, n = non_overlapping_frame(data_suara, fs, 20)
    frame_G_Ei = np.array([np.sum(np.square(frame_G[:,i]))/n for i in range(frame_G_N)])
    Ei_Mean = np.mean(frame_G_Ei)/5
    frame_G_Ci = spectral_centroid(frame_G, fs, frame_G_N)
    h = frame_G_N
    frame_G_Ei_H,bin_edge_Ei = np.histogram(frame_G_Ei,h)
    Hs_Ei = frame_G_Ei_H
    
    lo_ma_Ei0 = find_peaks(Hs_Ei*-1)[0][0:1]
    lo_ma_Ei1 = find_peaks(Hs_Ei)[0][0:1]
    if (len(lo_ma_Ei0) == 1) and (len(lo_ma_Ei1) == 1):
        if lo_ma_Ei0 < lo_ma_Ei1:
            lo_ma_Ei = lo_ma_Ei0
        else:
            lo_ma_Ei = lo_ma_Ei1
    elif (len(lo_ma_Ei0) == 1):
        lo_ma_Ei = lo_ma_Ei0
    elif (len(lo_ma_Ei1) == 1):
        lo_ma_Ei = lo_ma_Ei1
    else:
        lo_ma_Ei = []
    if len(lo_ma_Ei) < 1:
        T_Ei = Ei_Mean
    else:
        M_Ei = bin_edge_Ei[lo_ma_Ei]
        T_Ei = M_Ei      
        
    M_Ci0 = 3500 / (fs/2)
    M_Ci1 = 5000 / (fs/2)
        
    Flags_Ei = frame_G_Ei >= T_Ei
    
    Flags_Ci = np.logical_or((frame_G_Ci <= M_Ci0),(frame_G_Ci >= M_Ci1))
    
    flags_and = Flags_Ei & Flags_Ci
    
    flags_and_f = np.zeros(frame_G_N)
    min04 = [4, 6, 7, 9, 11, 12, 13, 14, 15, 17, 19, 20, 24, 25, 27]
    flags_tambah = np.zeros(frame_G_N)
    if b in min04:
        for i in range(frame_G_N):
            break_f = 0
            if flags_and[i]:
                if frame_G_N-i >= 20:
                    flags_tambah[i:i+20] = np.ones(20)
                else:
                    flags_tambah[i:] = np.ones(frame_G_N-i)
                break_f = 1
            if break_f:
                break
    flags_and = np.logical_or(flags_and,flags_tambah)
    
    for i in range(frame_G_N-5):
        if sum(flags_and[i:i+5]) > 0.5 and flags_and[i] and flags_and[i+5]:
            flags_and_f[i:i+5] = np.ones(5)
    flags_and = np.logical_or(flags_and,flags_and_f)
    
    frame_G_Ei *= flags_and
    
    flags_and_pjg = np.ones(frame_G_N*n)
    for i in range(len(flags_and)):
        flags_and_pjg[i*n:i*n+n] = [flags_and[i]]*n
        
    for i in range(len(frame_G_Ei)-3):
        if (np.mean(frame_G_Ei[i:i+2])) > (max(frame_G_Ei)/5):
            # flags_and = flags_and[i:]
            j = (i)*n
            if j > (fs*0.01):
                data_suara = data_suara[j-int(fs*0.01):]
                flags_and_pjg = flags_and_pjg[j-int(fs*0.01):]
            break
        
    flags_and = flags_and_pjg[:len(data_suara)]
    
    be_te = [3,22]
    for i in range(len(data_suara)):
        awal = i+n*3
        akhir = i+n*4
        if all(flags_and[i:awal]) and not any(flags_and[awal:akhir]):
            if len(data_suara) > awal+fs*0.1 and (np.sum(flags_and[:awal])/np.sum(flags_and)) > .8:
                if b in be_te:
                    data_suara = data_suara[:int(awal+fs*0.25)]
                else:
                    data_suara = data_suara[:int(awal+fs*0.1)]
                break

    flags_and = flags_and_pjg[:len(data_suara)]

    data_suara = wavelet_denoising(data_suara, wavelet, fs, flags_and)
    
    return data_suara, fs, scale_factor

def ekstrak_mfcc(al, huruf):
    data_suara, fs, scale_factor = proses(sound_path+al, huruf)
    mfcc_data = mfcc(data_suara, 
                     fs=fs, 
                     num_ceps=20,
                     low_freq=0,
                     high_freq=(2595 * np.log10(1 + (fs / 2) / 700)),
                     use_energy=True,
                     window=SlidingWindow(win_len = 0.02, win_hop = 0.01, win_type='hamming'),
                     nfft=22051
                     )
    the_data = pd.DataFrame(mfcc_data)
    if len(the_data.index) > 201:
        the_data = the_data.drop(the_data.index[201:])
    elif len(the_data.index) < 201:
        for l in range(len(the_data.index),201):
            the_data = pd.concat([the_data,mfcc_zeros],ignore_index=True)
    return the_data
    
def ekstrak_rasta(al, huruf):
    data_suara, fs, scale_factor = proses(sound_path+al, huruf)
    rplp_data = rplp(data_suara, 
                     fs=fs,
                     order=20,
                     window=SlidingWindow(win_len = 0.02, win_hop = 0.01, win_type='hamming'),
                     nfft=22051,
                     low_freq=0,
                     high_freq=(2595 * np.log10(1 + (fs / 2) / 700)),
                     )
    the_data = pd.DataFrame(rplp_data)
    if len(the_data.index) > 201:
        the_data = the_data.drop(the_data.index[201:])
    elif len(the_data.index) < 201:
        for l in range(len(the_data.index),201):
            the_data = pd.concat([the_data,mfcc_zeros],ignore_index=True)
    return the_data
    
def ekstrak_lpc(al, huruf):
    data_suara, fs, scale_factor = proses(sound_path+al, huruf)
    lpc_data = lpc(data_suara,
                   fs=fs,
                   order=20,
                   window=SlidingWindow(win_len = 0.02, win_hop = 0.01, win_type='hamming'),
                   )
    the_data = pd.DataFrame(lpc_data)
    if len(the_data.index) > 201:
        the_data = the_data.drop(the_data.index[201:])
    elif len(the_data.index) < 201:
        for l in range(len(the_data.index),201):
            the_data = pd.concat([the_data,mfcc_zeros],ignore_index=True)
    return the_data

def open_file():
    global file
    file_object = filedialog.askopenfile(mode='r')
    if file_object:
        strvarError.set(file_object.name)
        file = file_object.name.split('/')[-1].split('.')[0]
        print(file)

def play_sound():
    global file
    pygame.mixer.music.load(sound_path+file+'.wav')
    pygame.mixer.music.play(loops=0)

def predict():
    global file
    tree.delete(*tree.get_children())
    temp_nilai = 0
    for idx, index in enumerate(index_sifat):
        if information_dict[index]['ekstrak'] == 'mfcc':
            ekstrak_mfcc(file, huruf.get())
        elif information_dict[index]['ekstrak'] == 'rasta':
            ekstrak_rasta(file, huruf.get())
        else:
            ekstrak_lpc(file, huruf.get())
        temp_model = xgboost.XGBClassifier(learning_rate=0.01, n_estimators=300, min_child_weight=5, objective='binary:logistic', gpu_id=0)
        temp_model.load_model(model_path+f"xgboost_{index}.json")
        temp_pred = temp_model.predict([temp_data.flatten()])
        temp_tag = ''
        if temp_pred[0] == pengelompokan[idx][OPTIONS[huruf.get()]]:
            temp_tag = 'benar'
            temp_nilai = temp_nilai + 1
        else:
            temp_tag = 'salah'
        tree.insert('', 'end', text=index, values=(information_dict[index][temp_pred[0]],), tags=(temp_tag,))
    strvarNilai.set("Nilai : {:.2f}".format(temp_nilai/len(index_sifat)))
    tree.pack()

def open_popup():
   top = Toplevel(win)
   top.geometry("500x400")
   top.title("Keteranga Mengenai Sifat")
   Label(top, text= "Petunjuk Untuk Seluruh Sifat Huruf", font=('Georgia 13')).pack(pady=5)
   temp_text = """
    ----------------------------------------------------------------------
    Sifat S1:
    
    ----------------------------------------------------------------------
    Sifat S2:
    
    ----------------------------------------------------------------------
    Sifat S3:
    
    ----------------------------------------------------------------------
    Sifat S4:
    
    ----------------------------------------------------------------------
    Sifat S4:
    
    ----------------------------------------------------------------------
    Sifat S5:
    
    ----------------------------------------------------------------------
    Sifat T1:
    
    ----------------------------------------------------------------------
    Sifat T2:
    
    ----------------------------------------------------------------------
    Sifat T3:
    
    ----------------------------------------------------------------------
    Sifat T4:
    
    ----------------------------------------------------------------------
    Sifat T5:
    
    ----------------------------------------------------------------------
    Sifat T6:
    
    ----------------------------------------------------------------------
    Sifat T7:
    
    ----------------------------------------------------------------------
   """
   # Label(top, borderwidth=2, font=("HelvLight", 10), height=200, width=60, text=temp_text, relief='solid', justify='left').pack(pady=5)
   
   myscroll = ttk.Scrollbar(top, orient='vertical')
   myscroll.pack(side=RIGHT, fill='y')
   myentry = Text(top, font="Georgia 10", yscrollcommand=myscroll.set)
   myentry.insert(END, temp_text)
   myentry.config(state=DISABLED)
   myscroll.config(command=myentry.yview)
   myentry.pack(pady=5)

wavelet = fejer_korovkin()
mfcc_zeros = pd.DataFrame([[0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

win = Tk()
win.title('Demo Application')
win.geometry("700x700")
pygame.mixer.init()

label = Label(win, text="Klasifikasi Hijaiyah berdasarkan Sanad", font=('Georgia 13'))
label.pack(pady=5)

ttk.Button(win, text="Browse", command=open_file).pack(pady=5)
ttk.Button(win, text="Play Sound", command=play_sound).pack(pady=5)

strvarError = StringVar()
lblError = Label(win, textvar=strvarError, font=('Georgia 12'))
lblError.pack(pady=5)

huruf = StringVar(win)
huruf.set(list(OPTIONS.keys())[0])
w = OptionMenu(win, huruf, *OPTIONS.keys())
w.pack(pady=5)

ttk.Button(win, text="Predict", command=predict).pack(pady=5)

strvarNilai = StringVar()
lblNilai = Label(win, textvar=strvarNilai, font=('Georgia 12'))
lblNilai.pack(pady=5)

# Create an object of Style widget
style = ttk.Style()
style.theme_use('clam')

# Add a Treeview widget
tree = ttk.Treeview(win, column=("Sifat"), show='headings', height=12)
tree.tag_configure('benar', background='green')
tree.tag_configure('salah', background='red')
tree.column("# 1", anchor=CENTER)
tree.heading("# 1", text="Sifat")

for index in index_sifat:
    tree.insert('', 'end', text="-", values=('-'))
tree.pack()

keterangan_1 = Label(win, text="Sifat Benar", font=('Georgia 12'), background='green')
keterangan_1.pack(pady=5)

keterangan_2 = Label(win, text="Sifat Salah", font=('Georgia 12'), background='red')
keterangan_2.pack(pady=5)

keterangan_3 = Label(win, text="Harap ditingkatkan kembali pengucapan dalam sifat yang berwarna merah dari huruf tersebut", font=('Georgia 12'))
keterangan_3.pack(pady=5)

# ttk.Button(win, text= "Petunjuk Sifat", command= open_popup).pack(pady=5)

win.mainloop()