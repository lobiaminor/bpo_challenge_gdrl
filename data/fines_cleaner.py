import pandas as pd

# We need to clean the dataset. There are several challenges:
# - task complete events are actually single events with a start time and a complete time, but they may be suspended and resumed in between
# - there are tasks complete without explicitly starting: remove cases that have that
# - there are tasks that start twice: remove cases that have that
# After doing all that, we have a dataset that has about 95% of the cases left.
# All tasks now start and complete, so we have their duration.


if __name__ == '__main__':

    args = {
        'generate_reduced_cleaned_dataset': True
    }


    if args['generate_reduced_cleaned_dataset']:

        rhandle = open("fines/Road_Traffic_Fine_Management_Process.csv", "r")
        whandle = open("fines/fines_cleaned.csv", "w", newline='')\

        df = pd.read_csv(rhandle, header=0)

        df = df[['Case ID', 'Activity', 'Resource', 'Complete Timestamp', 'Variant', 'amount']]



        # Cases are always handled by a single resource, so we group by Case ID and copy the resource to nan values
        df['Resource'] = df.groupby('Case ID')['Resource'].transform(lambda x: x.fillna(x.mode()[0]))

        #The amount column contains the amount of the fine at each step in the process, so we need to aggregate it
        #We will sum the amount of the fine for each case
        df['amount'] = df.groupby('Case ID')['amount'].transform('sum')

        #pandas_data_cleaned.to_csv(whandle, index=False,  line_terminator='\n')
        print(df.head(10))

        #Rows where the Activity is 'Create Fine' are the start events
        #We use them to mark the start timestamp of the case
        #df['Start Timestamp'] = df.loc[df['Activity'] == 'Create Fine', 'Complete Timestamp']

        #The Start Timestamp of each activity (except 'Create Fine') is the Complete Timestamp of the previous activity
        #To do so, we must fill the NaN values in the Start Timestamp column with the previous value of Complete Timestamp
        df['Start Timestamp'] = df.groupby('Case ID')['Complete Timestamp'].shift()

        #We drop the rows where the Activity is 'Create Fine' because they are not actual activities
        df = df[df['Activity'] != 'Create Fine']


        #We drop rows where activity takes 0 time
        df = df[df['Start Timestamp'] != df['Complete Timestamp']]


        df['Duration'] = pd.to_datetime(df['Complete Timestamp']) - pd.to_datetime(df['Start Timestamp'])

        #Print, for each activity type, the average completion time (complete time - start time)
        print(df.groupby('Activity')['Duration'].mean())

        #Send Appeal to Prefecture and Send for Credit Collection are outliers in the dataset, so we remove them
        df = df[df['Activity'] != 'Send Appeal to Prefecture']
        df = df[df['Activity'] != 'Send for Credit Collection']

        #Convert resources to string
        df['Resource'] = df['Resource'].astype(str)

        #Save the reduced dataset
        df.to_csv(whandle, index=False, line_terminator='\n')


        whandle.flush()
        whandle.close()
        rhandle.close()

