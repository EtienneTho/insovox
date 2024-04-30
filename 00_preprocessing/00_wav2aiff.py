import glob, os
import soundfile as sf
from scipy.signal import decimate, resample
import csv
import numpy as np
import pandas as pd


input_path = './originalRecordings/'


df = pd.read_csv('./timecodes.csv', sep='[;,]', engine='python')

os.chdir(input_path)
for file in glob.glob("*.WAV"):
	print(file)
	fs_target = 44100
	data, samplerate = sf.read(file)
	if samplerate != fs_target:
		# data = decimate(data[:,0],2 )
		# Decimation ratio
		decimation_ratio = fs_target / samplerate  # Example ratio, you can adjust this as needed

		# Calculate the new length of the signal after decimation
		new_length = int(len(data[:,0]) * decimation_ratio)

		# Resample the signal to the new length
		data = resample(data[:,0], new_length)

	parts = file.split('_') #.split(".")[0]
	filtered_df = df[(df['subjectId'] == int(parts[1])) & (df['sessionId'] == int(parts[2].split(".")[0]))]

	start1 = int(filtered_df['start1'].iloc[0] * fs_target)
	end1   = int(filtered_df['end1'].iloc[0] * fs_target)
	start2 = int(filtered_df['start2'].iloc[0] * fs_target)
	end2   = int(filtered_df['end2'].iloc[0] * fs_target)

	# print(os.path.splitext(file)[0]+".aiff")
	sf.write(os.path.splitext(file)[0]+"_task1.aiff", data[start1:end1]/np.amax(data[start1:end1]), fs_target)
	sf.write(os.path.splitext(file)[0]+"_task2.aiff", data[start2:end2]/np.amax(data[start2:end2]), fs_target)
	#os.remove(file)


