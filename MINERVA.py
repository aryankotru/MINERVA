
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
import math
import csv
import time
import os
from collections import Counter
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import butter, sosfilt,filtfilt,  sosfreqz, lfilter, iirnotch, spectrogram
from sklearn.decomposition import PCA
from scipy.spatial import distance

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)


epoch = 710 #calc epoch len from fs and epoch time len
num_epochs = 12
# Sample rate and desired cutoff frequencies (in Hz)*
fs = 250
lowcut = 5
highcut = 30
filter_order = 3
notch_quality_factor = 10
wavelet = 'sym9'
wavelet_decomp_level = 5

freq_bands = {"delta": [0.3, 4], "theta": [4, 8], "alpha": [8, 13], "beta": [13, 30], "gamma": [30, 50]}
chars = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "*", "#"]
targets = [1 , 2 ,3 , 4, 5, 6,7,8,9,10,11,12]
ssvep_targets = [24, 0, 18, 0, 12, 0, 8, 0]

feature_names = ['Mean', 'Standard Deviation', 'Skewness', 'Entropy', 'Energy', 'Kurtosis', 'Median' ,'Variance']
mu_feature_names = ['Mean', 'Kurtosis' ,'Energy', 'Skewness']

tsinghua_channel_names = ['POz', 'PO3', 'PO4', 'PO5', 'PO6', 'Oz', 'O1', 'O2']
channel_names = ["O1", "PO3", "CP3", "C5", "CP4", "C6", "PO4", "O2"]
final_channels = ["PO3", "PO4", "O1", "O2"]
channel_numbers = [1,2,3,4,5,6,7,8]
ssvep_channels = [1,2,3,4]

reference_data_path = f"C://Users//arkot//Downloads//24 6s epoch 5 mins w other buttons on screen alt betn class and 0.txt"
GUI_data_path = f"C://Users//arkot//Documents//OpenBCI_GUI//Recordings"
game_output_path = r"C://Users//arkot//Desktop//output.txt"


T =  2.84
nsamples = T * fs
t = np.arange(0, epoch) / fs

def read_input_file(input_file):
    data = []
    fs = 250
    header = []
    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('%Sample Rate'):
                key_value = line[1:].strip().split(' = ')
                value = key_value[1]
                fs = int(value.strip().split(' ')[0])

            elif line.startswith('Sample Index'):
                header = line.split(', ')
            else:
                data.append(line.split(', ')) #typecast as int'

    data = [sublist[1:][:8] for sublist in data[3:]]
    converted_array = np.array(data, dtype=float).tolist()

    return fs, header[:9], converted_array

def clear_file(input_file):
  with open(input_file, 'w') as file:
    file.writelines(" ")
  return

def get_latest_subdirs(b):
  result = []
  for d in os.listdir(b):
    bd = os.path.join(b, d)
    if os.path.isdir(bd): result.append(bd)
    latest_subdir = max(result, key=os.path.getmtime)
  return latest_subdir


def get_input_raw_file(latest_subdir):
  file_names = []
  for x in os.listdir(latest_subdir):
    if x.endswith(".txt"):
      filepath = os.path.join(latest_subdir, x)
    else:
      return "No matching file"
  return filepath



def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a
def bandstop_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop', analog = False)#, fs, )
    y = filtfilt(b, a, data)
    return y


"""data filtering"""

def notch_filter(data, rate, freq, quality):
      x = scipy.signal.filtfilt(*scipy.signal.iirnotch(freq / (rate / 2), quality), data)
      #https://neuraldatascience.io/7-eeg/erp_filtering.html
      return x

def butter_bandpass(lowcut, highcut, fs, order=filter_order):
    return butter(order, [lowcut, highcut], fs=fs, btype='bandpass', analog = False)

def butter_bandpass_filter(data, lowcut, highcut, fs, order = filter_order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y =  filtfilt(b, a, data)
    return y

def butter_bandpass_sos(lowcut, highcut, fs, order=filter_order):
        nyq = 0.5 * fs #
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
        return sos

def butter_bandpass_filter_sos(data, lowcut, highcut, fs, order=filter_order):
        sos = butter_bandpass_sos(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

"""data normalization"""

def dataset_scaler(dataframe, column_names):
  scaler = MinMaxScaler() #normalizing data points, object of class StandardScaler
  minmax_scaled_data = scaler.fit_transform((dataframe))
  np.reshape(minmax_scaled_data, dataframe.shape)
  scaled_data = np.vstack((minmax_scaled_data))
  dfa = pd.DataFrame(minmax_scaled_data)
  dfa.columns = column_names #index for columns
  return dfa

def eta(data, unit='shannon'):
    base = {
        'shannon' : 2.,
        'natural' : math.exp(1),
        'hartley' : 10.}
    if len(data) <= 1:
        return 0
    counts = Counter()
    for d in data:
        d = round(d, 3)
        counts[d] += 1
    ent = 0
    probs = [float(c) / len(data) for c in counts.values()]
    for p in probs:
        if p > 0.:
            ent -= p * math.log(p, base[unit])
    return ent
  #https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python

def calculate_features(coeffs):
    mean_val = np.mean(coeffs)
    std_val = np.std(coeffs) #maybe try variance
    skewness_val = skew(coeffs)
    entropy_val = eta(coeffs)
    energy = np.sum(coeffs**2)
    kurtosis_value = kurtosis(coeffs)
    median_value = np.median(coeffs)
    variance_value = np.var(coeffs)
    #return mean_val, std_val, skewness_val, entropy_val, energy, kurtosis_value, median_value, variance_value #iteration lambda
    return mean_val, kurtosis_value, energy, skewness_val #iteration mu

"""wavelet transform & feature extraction"""

def feature_generator(df, epoch_len, wavelet_decomp_level):
    num_epochs = int(len(df) / epoch_len)
    all_features = []

    for n in range(num_epochs):
        #for channel in channel_numbers:
        for channel in ssvep_channels:
            data = df.iloc[n * epoch_len:(n + 1) * epoch_len, channel-1]
            coeffs = pywt.wavedec(data, wavelet, level = wavelet_decomp_level) #returns level+1 values, level detail coeffs and 1 approx
            """Frequency Bands:
              The frequency range corresponding to each level is calculated. For a sampling frequency of 250 Hz, the levels correspond roughly to:
              Level 1: 62.5 - 125 Hz
              Level 2: 31.25 - 62.5 Hz
              Level 3: 15.625 - 31.25 Hz
              Level 4: 7.8125 - 15.625 Hz
              Level 5: 3.9062 - 7.8125 Hz
              Approximation: 0 - 3.9062 Hz
              The alpha band (8-13 Hz) lies within Level 4."""
            #for i, coeff in enumerate(coeffs[2:4], 1): iteration lambda
            for i, coeff in enumerate(coeffs[0:wavelet_decomp_level], 1): #iteration mu, take all 5 relevant bands
                features = calculate_features(coeff)
                all_features.append([channel] + list(features))
            #features = calculate_features(coeffs[0])
            #all_features.append([channel] + list(features))

    feature_df = pd.DataFrame(all_features, columns=['Channel'] + mu_feature_names)
    feature_df.set_index('Channel', inplace=True)

    return feature_df

"""dataset labeller"""

level_map = [1,2,3,4,5]
feature_map = [1,2,3,4]
chars = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

#num = 20 for label, num = 1 for feature,  num = 4 for level
#n = 0 for label, n = 1 for feature, n = 2 for level, in order

def drop_column(dataframe, column_name):
  try:
    df = dataframe.drop(columns=column_name, axis =1)
    return df
  except Exception as e:
    return dataframe

def dataset_labeller(dataframe, num_features_per_epoch, targs, n, label):
  num_samples = len(dataframe) // num_features_per_epoch
  num_labels = len(targs)  # 12 labels

  # Create a list of labels for each 32-row segment
  labels = []
  for i in range(num_samples):
      labels.extend([targs[i % num_labels]] * num_features_per_epoch)
  # Add the labels to the DataFrame
  dataframe.insert(dataframe.shape[1]-n, label , labels, allow_duplicates = False) #df.shape[1] returns the number of columns
  return dataframe

def preprocessing(dataframe, epoch_len):

   filtered_df = pd.DataFrame(columns=dataframe.columns)
   num_epochs = round(int(len(dataframe.index)/epoch_len))
   #print(num_epochs)

   for channel, i in enumerate(dataframe.columns):
      filtered_col_data = []
      for n in range(num_epochs):
          data = dataframe.iloc[n*epoch:(n+1)*epoch, channel]
          data = notch_filter(data, fs, 50, notch_quality_factor)
          data = butter_bandpass_filter(data, lowcut, highcut, fs, order=filter_order)
          filtered_col_data.extend(data)
      filtered_df[i] = filtered_col_data
   feature_df = feature_generator(filtered_df, epoch_len, wavelet_decomp_level) #extract and store features
   #scaled_df = dataset_scaler(feature_df,  feature_names) #scale dataset AFTER filtering


   return feature_df #EXPERIMENT

def flatten(dataframe, num_channels, wavelet_decomp, num_features):
  # Define the segment size
  # Calculate the number of segments
  segment_height = num_channels * wavelet_decomp
  segment_width = num_features
  num_segments = dataframe.shape[0] // segment_height

  # Initialize a list to hold the flattened vectors
  flattened_vectors = []

  for i in range(num_segments):
      # Extract the 8x32 segment
      segment = dataframe.iloc[i * segment_height:(i + 1) * segment_height, :]
      #print(segment)

      # Flatten the segment into a 1D vector of length 256
      flattened_vector = segment.to_numpy().flatten()

      # Add the flattened vector to the list
      flattened_vectors.append(flattened_vector)

  # Convert the list of vectors to a NumPy array
  flattened_vectors = np.array(flattened_vectors)

  da =  pd.DataFrame(flattened_vectors)
  return da

def principal_ca(df):
  # Initialize PCA
  pca = PCA()


  X = df.drop(columns='Label')
  y = df['Label']
  # Fit PCA on the scaled data
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  pca.fit(X)

  # Calculate the explained variance ratio
  explained_variance_ratio = pca.explained_variance_ratio_

  # Calculate cumulative explained variance
  cumulative_explained_variance = np.cumsum(explained_variance_ratio)

  # Determine the number of components to retain 95% variance
  n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1

  # Transform the data using the selected number of principal components
  pca = PCA(n_components=n_components_95)
  transformed_data = pca.fit_transform(X)

  # Create a DataFrame for the transformed data
  columns = [f'PC{i+1}' for i in range(n_components_95)]
  transformed_df = pd.DataFrame(transformed_data, columns=columns)

  datafr = dataset_labeller(transformed_df, 1, ssvep_targets, 0, 'Label')
  return datafr



def write_model_output(model_output):
  
  with open(game_output_path, "w") as file:
    file.write(str(model_output))
    return


def euclidean_dist(data1, data2):
  arr1 = data1.iloc[:,:-1]
  arr2 = data2.iloc[24, :-1]
  #print(arr1)
  dist = distance.euclidean(arr1, arr2)
  print(f'Euclidean distance: {dist}')
  return dist


def main(input_method):
  fs1, header1, ref_data = read_input_file(reference_data_path)
  reference_dataset = pd.DataFrame(ref_data, columns = channel_names)
  reference_dataset = reference_dataset[final_channels]
  model_output_list = []
  dist_aggregate = [0, 0, 0, 0]

  if input_method == 'system_read':
    fs = 250
    print("Finding file path")
    dir = get_latest_subdirs(GUI_data_path)
    input_file = get_input_raw_file(dir)  # Replace with your input file path
    print("Reading EEG data")
    fs, header, data = read_input_file(input_file)
    print("Clearing file")
    #clear_file(input_file)
    epoch_len = fs * 6
    df = pd.DataFrame(data, columns = channel_names)
    df = df[final_channels]


  elif input_method == 'manual_read':
    fs = 250
    epoch_len = 710
    df = pd.read_csv(input_file, header = None)
    df = np.transpose(df)
    df.columns = tsinghua_channel_names
    num_rows, num_cols = df.shape
    num_epochs = int((num_rows)/epoch)
  
  print("Processing data")
  preprocessed_df = preprocessing(df, epoch_len = epoch_len)
  dataset = flatten(preprocessed_df, num_channels = 4, wavelet_decomp = wavelet_decomp_level, num_features = 4)
  labelled_dataset = dataset_labeller(dataset, 1, ssvep_targets, 0, 'Label')
  print("Reducing dataset dimensions")
  reduced_dataset = principal_ca(labelled_dataset)
  reduced_dataset.to_csv('MINPCA95.csv', index=False)
  print("Calculating Euclidean distances")
  for i in range(int(len(df)/epoch_len)):
    dist0 = distance.euclidean(df.iloc[i*epoch_len:(i+1)*epoch_len, 0], reference_dataset.iloc[2*epoch_len:3*epoch_len,0]) #take new reference dataset
    dist1 = distance.euclidean(df.iloc[i*epoch_len:(i+1)*epoch_len,1], reference_dataset.iloc[2*epoch_len:3*epoch_len, 1])
    dist2 = distance.euclidean(df.iloc[i*epoch_len:(i+1)*epoch_len,2], reference_dataset.iloc[2*epoch_len:3*epoch_len, 2])
    dist3 = distance.euclidean(df.iloc[i*epoch_len:(i+1)*epoch_len,3], reference_dataset.iloc[2*epoch_len:3*epoch_len, 3])
    dist_list = [dist0, dist1, dist2, dist3]

    # Accumulate distances
    for j in range(4):
      dist_aggregate[j] += dist_list[j]

    # Calculate average distances
    avg_distances = [dist / int(len(df)/epoch_len) for dist in dist_aggregate]

    if avg_distances[0] < 100000 and avg_distances[1] < 100000 and avg_distances[2] < 100000:
        model_output = 1
    elif avg_distances[0] < 100000 and avg_distances[1] < 100000 and avg_distances[3] < 200000:
        model_output = 1
    elif avg_distances[0] < 200000 and avg_distances[2] < 200000 and avg_distances[3] < 200000:
        model_output = 1
    elif avg_distances[1] < 200000 and avg_distances[2] < 200000 and avg_distances[3] < 200000:
        model_output = 1
    else:
        model_output = 0

    pred_class = write_model_output(str(model_output))
    return model_output, avg_distances

if __name__ == '__main__':
  time.sleep(6) #move to within while loop
  while True:
    try:
        pred_class, dist_list = main('system_read')
        print("Predicted class " +str(pred_class)+" with Euclidean Distances: "+str(dist_list))
    except KeyboardInterrupt:
      quit()

