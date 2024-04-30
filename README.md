# insovox
This repository contains the script of the insovox project.

In './00_preprocessing/':
- A folder named 'originalRecordings' must be created.
- The WAV files with the following format 'Year_SubjectId_SessionId.WAV' must be added in 'originalRecordings'
- A folder named 'stmtf' must be created
- the csv file 'timecodes.csv' must be updated according to the files in 'originalRecordings'

In './00_preprocessing/':
- A folder named '01_acoustic_analyses' must be created

The scripts must be executed in the following order:
- '00_wav2aiff.py'
- '01_genereStrfCuted.py'
- '00_load_and_save_data.py'
- '01_acoustic_analysis_per_subject.py'
