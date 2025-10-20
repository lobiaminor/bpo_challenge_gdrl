import pandas as pd

# We need to clean the dataset. There are several challenges:
# - task complete events are actually single events with a start time and a complete time, but they may be suspended and resumed in between
# - there are tasks complete without explicitly starting: remove cases that have that
# - there are tasks that start twice: remove cases that have that
# After doing all that, we have a dataset that has about 95% of the cases left.
# All tasks now start and complete, so we have their duration.


if __name__ == '__main__':

    args = {
        'generate_cleaned_dataset': True
    }


    if args['generate_cleaned_dataset']:

        rhandle = open("production/Productions.csv", "r")
        whandle = open("production/productions_cleaned.csv", "w", newline='')\

        df = pd.read_csv(rhandle, header=0)

        df = df[['Case ID', 'Activity', 'Resource', 'Complete Timestamp.1', 'Start Timestamp']]

        #Rename columns
        df.columns = ['Case ID', 'Activity', 'Resource', 'Complete Timestamp', 'Start Timestamp']

        #Drop duplicated rows
        df = df.drop_duplicates()

        # Convert timestamps to datetime without specifying the unit
        df['Start Timestamp'] = pd.to_datetime(df['Start Timestamp'])
        df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])

        df = df.sort_values(by=['Case ID', 'Start Timestamp'])

        df['Duration'] = pd.to_datetime(df['Complete Timestamp']) - pd.to_datetime(df['Start Timestamp'])

        #Drop cases that have duration equal to 0 (timestamp)
        df = df[df['Duration'] > pd.Timedelta(0)]

        #Count the number of resources
        print(f'Number of resources: {df["Resource"].nunique()}')

        #print average number of tasks per case
        print(f"Average number of tasks per case: {df.groupby('Case ID')['Activity'].count().mean()}")

        #Print, for each activity type, the average completion time (complete time - start time)
        print(df.groupby('Activity')['Duration'].mean())

        #Convert resources to string
        df['Resource'] = df['Resource'].astype(str)
        df = df.reset_index(drop=True)


        #Make the df strictly sequential: if a resource is seen doing two activities in parallel, it is considered as two different resources
        def expand_resources_in_group(group):
            group = group.sort_values(by='Start Timestamp')
            #if the resource is the same for two consecutive activities, and the Start Timestamp of the second activity is before the Complete Timestamp of the first activity, then the resource is considered as two different resources
            for i in range(len(group)-1):
                if group.iloc[i]['Resource'] == group.iloc[i+1]['Resource'] and group.iloc[i+1]['Start Timestamp'] < group.iloc[i]['Complete Timestamp']:
                    group.at[group.index[i+1], 'Resource'] = group.iloc[i+1]['Resource'] + '_REPLICA'
                    print(f"Changed resource to {group.iloc[i+1]['Resource']}")
            return group

        def erase_concurrent_activities(group):
            group = group.sort_values(by='Start Timestamp')
            #if the resource is the same for two consecutive activities, and the Start Timestamp of the second activity is before the Complete Timestamp of the first activity, then remove the second activity
            i = 0
            while i < len(group)-1:
                if group.iloc[i]['Resource'] == group.iloc[i+1]['Resource'] and group.iloc[i+1]['Start Timestamp'] < group.iloc[i]['Complete Timestamp']:
                    group = group.drop(group.index[i+1])
                    group = group.reset_index(drop=True)
                else:
                    i += 1
            return group

        # Apply the filter_group function to each group
        df = df.groupby('Case ID').apply(erase_concurrent_activities).reset_index(drop=True)

        # Erase cases that were assigned to the same resources whose final activity Complete Time is prior to the Start Time of the first activity of the second case
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

        #Save the reduced dataset
        df.to_csv(whandle, index=False, line_terminator='\n')


        whandle.flush()
        whandle.close()
        rhandle.close()

