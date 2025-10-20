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

        rhandle = open("consulta2018/ConsultaDataMining201618.csv", "r")
        whandle = open("consulta2018/consulta_cleaned.csv", "w", newline='')\

        df = pd.read_csv(rhandle, header=0)

        df = df[['Case ID', 'Activity', 'Resource', 'start:timestamp', 'Complete Timestamp', 'Variant']]

        #Check in how many rows the complete timestamp is prior to the start timestamp
        print(f'Number of rows where the complete timestamp is prior to the start timestamp: {len(df[df["Complete Timestamp"] < df["start:timestamp"]])}')

        #Rename
        df = df.rename(columns={'start:timestamp': 'Start Timestamp'})

        #Drop Start and End Events (not performed by humans)
        df = df[df['Activity'] != 'Start']
        df = df[df['Activity'] != 'End']

        #Remove cases that have tasks that complete without starting
        df = df[df['Start Timestamp'].notnull()]


        #Convert unix timestamp to datetime
        df['Start Timestamp'] = pd.to_datetime(df['Start Timestamp'], utc=True)
        df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'], utc=True)
        # Where the complete timestamp is prior to the start timestamp, we drop the row
        df = df[df["Complete Timestamp"] > df["Start Timestamp"]]



        df['Duration'] = pd.to_datetime(df['Complete Timestamp']) - pd.to_datetime(df['Start Timestamp'])



        #Drop cases that have duration equal to 0 (timestamp)
        df = df[df['Duration'] > pd.Timedelta(0)]

        #Remove tasks whose resource pool is less than 3
        df = df.groupby("Activity").filter(lambda x: len(x["Resource"].unique()) > 3)

        #Print the average duration of every single task
        print(df.groupby('Activity')['Duration'].mean())

        #Count the number of resources
        print(f'Number of resources: {df["Resource"].nunique()}')

        #Filter out resources that have less than 10000 tasks
        #df = df.groupby('Resource').filter(lambda x: len(x) > 10000)


        #Count the number of resources after filter
        print(f'Number of resources after filter: {df["Resource"].nunique()}')

        #print average number of tasks per case
        print(f"Average number of tasks per case: {df.groupby('Case ID')['Activity'].count().mean()}")
        #Remove cases that have too many tasks (more than 100)
        #df = df.groupby('Case ID').filter(lambda x: len(x) < 100)

        # print average number of tasks per case
        print(f"Average number of tasks per case after filter: {df.groupby('Case ID')['Activity'].count().mean()}")

        #Print, for each activity type, the average completion time (complete time - start time)
        print(df.groupby('Activity')['Duration'].mean())

        """
        i = 0
        while i < len(df) - 1:
            if df.iloc[i]['Resource'] == df.iloc[i + 1]['Resource'] and df.iloc[i + 1]['Start Timestamp'] < df.iloc[i]['Complete Timestamp']:
                df = df.drop(df.index[i + 1])
                df = df.reset_index(drop=True)
                print("Erased")
            else:
                i += 1
        """

        #Print

        #Convert resources to string
        df['Resource'] = df['Resource'].astype(str)

        #Save the reduced dataset
        df.to_csv(whandle, index=False, line_terminator='\n')


        whandle.flush()
        whandle.close()
        rhandle.close()

