import pandas as pd

# We need to clean the dataset. There are several challenges:
# - task complete events are actually single events with a start time and a complete time, but they may be suspended and resumed in between
# - there are tasks complete without explicitly starting: remove cases that have that
# - there are tasks that start twice: remove cases that have that
# After doing all that, we have a dataset that has about 95% of the cases left.
# All tasks now start and complete, so we have their duration.


if __name__ == '__main__':

    args = {
        'generate_reduced_dataset': False,
        'clean_dataset': True
    }


    if args['generate_reduced_dataset']:

        rhandle = open("bpi2018/2018.csv", "r")
        whandle = open("bpi2018/2018_reduced.csv", "w", newline='')

        pandas_data = pd.read_csv(rhandle, header=0)

        pandas_data_reduced = pandas_data[['Case ID', 'Activity', 'Resource', 'Complete Timestamp', 'Variant', 'penalty_amount0']]

        # Drop rows where the 'Resource' column contains the string '0;n/a'
        pandas_data_cleaned = pandas_data_reduced[pandas_data_reduced['Resource'] != '0;n/a']
        pandas_data_cleaned = pandas_data_cleaned[pandas_data_cleaned['Resource'] != 'Remote inspection export']

        pandas_data_cleaned.to_csv(whandle, index=False,  line_terminator='\n')

        whandle.flush()
        whandle.close()
        rhandle.close()

    if args['clean_dataset']:
        #rhandle = open("bpi2018/2018_reduced.csv", "r")
        #whandle = open("bpi2018/2018_clean.csv", "w", newline='')

        pandas_data = pd.read_csv("bpi2018/2018_reduced.csv", header=0)

        # Filter out rows whose resource is "Processing automaton" or "Document processing automaton" (they are not interesting for us)
        pandas_data = pandas_data[~pandas_data['Resource'].str.contains("automaton", case=False)]
        # Filter out rows whose resource contains "Remote inspection" (they are not interesting for us)
        pandas_data = pandas_data[~pandas_data['Resource'].str.contains("Remote inspection", case=False)]


        print(pandas_data.head())

        print(pandas_data.columns)

        for column in pandas_data.columns:
            print(f'Column: {column} with {len(pandas_data[column].unique())} unique values')
            print(pandas_data[column].unique())

        #Many activity labels (167), and a large portion indicate initializations or terminations
        #We need to merge them to extract a new column called "Start Timestamp"

        #Check the values in column "Activity" that contain the term 'begin'
        start_events = pandas_data[pandas_data['Activity'].str.contains('begin', case=False)]['Activity'].unique()
        print(f"There are {len(start_events)} unique start events: {start_events}")

        #Edit every occurrence of the term 'begin' in the start_events list to 'finish'
        end_events_calculated = start_events.copy()
        for i in range(len(end_events_calculated)):
            end_events_calculated[i] = end_events_calculated[i].replace('begin', 'finish')

        #Check the values in column "Activity" that contain the term 'finish'
        end_events = pandas_data[pandas_data['Activity'].str.contains('finish', case=False)]['Activity'].unique()

        #Check overlap between end_events_calculated and end_events
        print(f"Length of end_events_calculated: {len(end_events_calculated)}")
        print(f"Length of end_events: {len(end_events)}")
        print(f"Number of overlappping elements between end_events_calculated and end_events: {len(set(end_events_calculated).intersection(set(end_events)))}")

        #Create a dictionary that maps start events to end events
        start_to_end = dict(zip(start_events, end_events_calculated))

        #Create a new column called "Start Timestamp" that contains the timestamp of the start event
        #pandas_data['Start Timestamp'] = pandas_data['Complete Timestamp']

        #Drop all rows in pandas_data that are not start events or end events (calculated)
        pandas_data = pandas_data[pandas_data['Activity'].isin(start_events) | pandas_data['Activity'].isin(end_events_calculated)]



        #Create a new empty dataset
        #pandas_data_cleaned = pd.DataFrame(columns=pandas_data.columns)

        # Filter the DataFrame to get only the rows with 'begin' in the 'Activity' column
        start_events_df = pandas_data[pandas_data['Activity'].str.contains('begin', case=False)]
        end_events_df = pandas_data[pandas_data['Activity'].str.contains('finish', case=False)]

        # Merge the start events DataFrame with the original DataFrame to find corresponding end events
        merged_df = start_events_df.merge(
            end_events_df,
            left_on=['Case ID', 'Resource', start_events_df['Activity'].map(start_to_end)],
            right_on=['Case ID', 'Resource', 'Activity'],
            suffixes=('_start', '_end')
        )

        # Create a new DataFrame with the required columns and values
        pandas_data_cleaned = merged_df[
            ['Case ID', 'Resource', 'Activity_end', 'Complete Timestamp_start', 'Complete Timestamp_end', 'Variant_start', 'penalty_amount0_start']].copy()
        pandas_data_cleaned.rename(columns={
            'Activity_end': 'Activity',
            'Complete Timestamp_start': 'Start Timestamp',
            'Complete Timestamp_end': 'Complete Timestamp',
            'Variant_start': 'Variant',
            'penalty_amount0_start': 'penalty_amount0'
        }, inplace=True)


        #Filter out the rows where Start Timestamp is bigger or equal than Complete Timestamp
        pandas_data_cleaned = pandas_data_cleaned[pandas_data_cleaned['Start Timestamp'] < pandas_data_cleaned['Complete Timestamp']]

        #Filter out the rows containing null values in any field
        pandas_data_cleaned = pandas_data_cleaned.dropna()

        #Filter out the rows contining None values in any field
        pandas_data_cleaned = pandas_data_cleaned[pandas_data_cleaned['Case ID'].notnull()]
        pandas_data_cleaned = pandas_data_cleaned[pandas_data_cleaned['Resource'].notnull()]
        pandas_data_cleaned = pandas_data_cleaned[pandas_data_cleaned['Activity'].notnull()]
        pandas_data_cleaned = pandas_data_cleaned[pandas_data_cleaned['Start Timestamp'].notnull()]
        pandas_data_cleaned = pandas_data_cleaned[pandas_data_cleaned['Complete Timestamp'].notnull()]

        #Filter out the rows contining '' values in any field
        pandas_data_cleaned = pandas_data_cleaned[pandas_data_cleaned['Case ID'] != '']
        pandas_data_cleaned = pandas_data_cleaned[pandas_data_cleaned['Resource'] != '']
        pandas_data_cleaned = pandas_data_cleaned[pandas_data_cleaned['Activity'] != '']
        pandas_data_cleaned = pandas_data_cleaned[pandas_data_cleaned['Start Timestamp'] != '']
        pandas_data_cleaned = pandas_data_cleaned[pandas_data_cleaned['Complete Timestamp'] != '']


        #Save the cleaned dataset
        pandas_data_cleaned.to_csv("bpi2018/2018_clean.csv", index=False, line_terminator='\n')
