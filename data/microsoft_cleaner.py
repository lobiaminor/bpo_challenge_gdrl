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

        rhandle = open("microsoft/SampleData_AP_Refunds_Financial_EventLog.csv", "r")
        whandle = open("microsoft/microsoft_cleaned.csv", "w", newline='')\

        df = pd.read_csv(rhandle, header=0)

        df = df[['Resource', 'StartTimestamp', 'EndTimestamp', 'CaseId', 'ActivityName']]

        #rename columns
        df.columns = ['Resource', 'Start Timestamp', 'Complete Timestamp', 'Case ID', 'Activity']

        df['Duration'] = pd.to_datetime(df['Complete Timestamp']) - pd.to_datetime(df['Start Timestamp'])

        #Drop cases that have duration equal to 0 (timestamp)
        df = df[df['Duration'] > pd.Timedelta(0)]

        #Print the average duration of every single task
        print(df.groupby('Activity')['Duration'].mean())

        #Check the average number of cases per resource
        print(f'Average number of tasks per case: {df.groupby("Resource")["Case ID"].nunique().mean()}')

        #Count the number of resources
        print(f'Number of resources: {df["Resource"].nunique()}')


        #Count the number of resources after filter
        print(f'Number of resources after filter: {df["Resource"].nunique()}')

        #print average number of tasks per case
        print(f"Average number of tasks per case: {df.groupby('Case ID')['Activity'].count().mean()}")
        #Remove cases that have too many tasks (more than 100)
        #df = df.groupby('Case ID').filter(lambda x: len(x) < 20)

        # print average number of tasks per case
        print(f"Average number of tasks per case after filter: {df.groupby('Case ID')['Activity'].count().mean()}")

        #Print, for each activity type, the average completion time (complete time - start time)
        print(df.groupby('Activity')['Duration'].mean())

        #Print

        #Convert resources to string
        df['Resource'] = df['Resource'].astype(str)

        #Save the reduced dataset
        df.to_csv(whandle, index=False, line_terminator='\n')


        whandle.flush()
        whandle.close()
        rhandle.close()

