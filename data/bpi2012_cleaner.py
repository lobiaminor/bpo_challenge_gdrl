import pandas as pd

# We need to clean the dataset. There are several challenges:
# - the tasks use lifecycles: they are started, completed, suspended, resumed we need to aggregate each task start-resume/complete-suspend combination to a single line
# - task complete events are actually single events with a start time and a complete time, but they may be suspended and resumed in between
# - there are tasks complete without explicitly starting: remove cases that have that
# - there are tasks that start twice: remove cases that have that
# After doing all that, we have a dataset that has about 95% of the cases left.
# All tasks now start and complete, so we have their duration.

# %% First we need to expand the complete events

rhandle = open("./bpi2012/BPI_Challenge_2012.csv", "r")
whandle = open("./bpi2012/BPI_Challenge_2012 - clean.csv", "w")

log = pd.read_csv(rhandle,  header=0)

#Add 'Start Timestamp' column
log['Start Timestamp'] = log['Complete Timestamp']

#Create a new empty log
new_log = pd.DataFrame(columns=log.columns)

#Drop SCHEDULE events
log = log[log['lifecycle:transition'] != 'SCHEDULE']

#Drop rows with nan values
log = log.dropna()

#Drop duplicate rows
log = log.drop_duplicates()

#Sort log by Timestamp
log = log.sort_values(by=['Case ID', 'Start Timestamp'])

#Unique activities
unique_act = log['Activity'].str.split('-').str[0].unique()

log['Activity'] = log['Activity'].str.split('-').str[0]

#In each case, check if the same resource has initiated and completed an activity
#If so, create a new row with the start timestamp of the start activity and the complete timestamp of the complete activity
for case in log['Case ID'].unique():
    case_log = log[log['Case ID'] == case]
    case_log = case_log.sort_values(by=['Start Timestamp'])
    for act in unique_act:
        act_log = case_log[case_log['Activity'].str.contains(act)]
        if len(act_log) > 1:
            for i in range(len(act_log)-1):
                for j in range(i+1, len(act_log)):
                    if act_log.iloc[i]['Resource'] == act_log.iloc[j]['Resource'] and act_log.iloc[i]['lifecycle:transition'] == 'START' and act_log.iloc[j]['lifecycle:transition'] == 'COMPLETE':

                        new_row = act_log.iloc[0]
                        new_row['Complete Timestamp'] = act_log.iloc[1]['Complete Timestamp']
                        new_row['Activity'] = act
                        new_row['Duration'] = pd.to_datetime(new_row['Complete Timestamp']) - pd.to_datetime(new_row['Start Timestamp'])
                        new_log = new_log.append(new_row)
                        break #We only need to create one row per activity




print(new_log)

new_log.to_csv(whandle, index=False)

