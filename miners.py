import numpy as np
import pandas
import datetime
import pickle
import matplotlib.pyplot as plt
import os
from statistics import mean

import pandas as pd

from simulator import MinedProblem

##PARAMS##
# possible values: '2012', '2017', 'consulta', 'production', 'microsoft'
problem_name = 'microsoft'
include_parallel_activities = False
threshold = 0  # 3000 for 2018 and 2017, 5000 for fines, 500 for toloka
only_plot = False
#########################################################################


def adjust_resource_pool_size(df, min_pool_size, max_pool_size):
    resource_counts = df['Resource'].value_counts()
    adjusted_pool = resource_counts[(resource_counts >= min_pool_size) & (resource_counts <= max_pool_size)].index
    return df[df['Resource'].isin(adjusted_pool)]


def mine_problem(log, problem_name, mean_interarrival_time_adjustment_factor=1, task_type_filter=None, datetime_format="%Y/%m/%d %H:%M:%S", min_tasks_per_case=1, max_tasks_per_case=None, min_resource_count=2, min_resource_pool_size=2, resource_schedule_timeunit=datetime.timedelta(hours=1), resource_schedule_repeat=168):
    """
    Mines a problem and returns it as a :class:`.problems.Problem` that can be simulated.
    The log from which the model is mined must at least have the columns
    Case ID, Activity, Resource, Start Timestamp, Complete Timestamp,
    which identify the corresponding event log information. Activity labels
    are the same as Task Types for the purposes of the problem definition.
    The timing distributions associated with the problem are all in hours.

    :param log: a pandas dataframe from which the problem must be mined.
    :param task_type_filter: a function that takes the name of a task type/ activity
                             and returns if it should be included, or None to include all task types.
    :param datetime_format: the datetime format the Start Timestamp and Complete Timestamp columns use.
    :param min_tasks_per_case: the minimum number of tasks a case must have for it to be included in the problem.
    :param max_tasks_per_case: the maximum number of tasks a case must have for it to be included in the problem.
    :param min_resource_count: the minimum number of times a resource must have executed a task
                               of a particular type, for it to be considered in the pool of resources for
                               the task type. This must be greater than 1, otherwise the standard deviation
                               of the processing time cannot be computed.
    :param min_resource_pool_size: the minimum number of resources a task type must have in its pool for it to be included in the problem.
    :param resource_schedule_timeunit: the timeunit in which resource schedules should be represented. Default is 1 hour.
    :param resource_schedule_repeat: the number of times after which the resource schedule is expected to repeat itself. Default is 168 repeats (of 1 hour is a week).
    :return: a :class:`.problems.Problem`.
    """

    # MINE THE BASICS
    # Mine the task types
    # Mine the resources
    # Mine the initial task type distribution
    # Mine the next task type distribution per task type
    # Mine the interarrival time
    # Mine the resource pool per task type
    # Mine the processing time distribution per task_type/resource combination
    # TODO: Data distribution is empty for now, future work


    df = log.copy()
    df['Resource'] = df['Resource'].astype(str)
    df['Start Timestamp'] = pandas.to_datetime(df['Start Timestamp'], format=datetime_format)
    df['Complete Timestamp'] = pandas.to_datetime(df['Complete Timestamp'], format=datetime_format)
    df['Duration'] = df[['Start Timestamp', 'Complete Timestamp']].apply(lambda tss: (tss[1]-tss[0]).total_seconds()/3600, axis=1)


    df = df.drop_duplicates()

    task_types = df['Activity'].unique()
    if task_type_filter is not None:
        task_types = [tt for tt in task_types if task_type_filter(tt)]
        df = df[df['Activity'].isin(task_types)] #ADDED (don't know how it was done before...)

    if min_resource_pool_size > 1:
        df = df.groupby('Activity').filter(lambda x: len(x['Resource'].unique()) >= min_resource_pool_size)
    task_types = df['Activity'].unique()


    resources = df['Resource'].unique()
    df_cases = df.groupby('Case ID').agg({'Start Timestamp': 'min', 'Activity': lambda tss: list(tss)})
    df_cases = df_cases.rename(columns={'Activity': 'Trace'})
    df_cases = df_cases.sort_values(by='Start Timestamp')
    if min_tasks_per_case > 1:
        #check the number of tasks per case
        df_cases['Task Count'] = df_cases['Trace'].apply(lambda x: len(x))
        df_cases = df_cases[df_cases['Task Count'] >= min_tasks_per_case]

        #remove the cases that have only one task from the dataframe
        df = df[df['Case ID'].isin(df_cases.index)]
    elif min_tasks_per_case < 1:
        raise ValueError("min_tasks_per_case must be greater than 1")

    average_tasks_per_case = df_cases['Trace'].apply(len).mean()
    print(f"Average amount of tasks per case after filter: {average_tasks_per_case}")
    std_tasks_per_case = df_cases['Trace'].apply(len).std()
    print(f"Std amount of tasks per case after filter: {std_tasks_per_case}")
    #print(f"Average amount of tasks per case after filter: {df.groupby('Case ID')['Activity'].count().mean()}")
    #print(f"Standard dev tasks per case after filter: {df.groupby('Case ID')['Activity'].count().std()}")

    initial_tasks = dict()
    following_task = dict()
    interarrival_times = []
    last_arrival_time = None
    for index, row in df_cases.iterrows():
        if last_arrival_time is not None:
            interarrival_times.append((row['Start Timestamp'] - last_arrival_time).total_seconds()/3600)
        last_arrival_time = row['Start Timestamp']
        if not row['Trace'][0] in initial_tasks.keys():
            initial_tasks[row['Trace'][0]] = 0
        initial_tasks[row['Trace'][0]] += 1
        for i in range(len(row['Trace'])):
            predecessor = row['Trace'][i]
            if i+1 >= len(row['Trace']):
                successor = None
            else:
                successor = row['Trace'][i+1]
            if not (predecessor, successor) in following_task:
                following_task[(predecessor, successor)] = 0
            following_task[(predecessor, successor)] += 1
    mean_interarrival_time = (sum(interarrival_times)/len(interarrival_times))*mean_interarrival_time_adjustment_factor  # Assuming exponential distribution, so we only need the mean
    initial_task_distribution = []
    for it in initial_tasks:
        initial_task_distribution.append((initial_tasks[it]/len(df_cases), it))
    next_task_distribution = dict()
    task_occurrences = dict()
    for (predecessor, successor) in following_task:
        if predecessor not in next_task_distribution.keys():
            next_task_distribution[predecessor] = dict()
            task_occurrences[predecessor] = 0
        next_task_distribution[predecessor][successor] = following_task[(predecessor, successor)]
        task_occurrences[predecessor] += following_task[(predecessor, successor)]
    for predecessor in next_task_distribution:
        successors = []
        for successor in next_task_distribution[predecessor]:
            successors.append((next_task_distribution[predecessor][successor]/task_occurrences[predecessor], successor))
        next_task_distribution[predecessor] = successors

    df_resources = df.groupby(['Activity', 'Resource'], as_index=False).agg(Duration_mean=('Duration', 'mean'), Duration_std=('Duration', 'std'), Resource_count=('Resource', 'count'))
    resource_pools = dict()
    processing_time_distribution = dict()
    for tt in task_types:
        resource_pools[tt] = []
    for index, row in df_resources.iterrows():
        if row["Resource_count"] > min_resource_count:
            resource_pools[row['Activity']].append(row['Resource'])
            processing_time_distribution[(row['Activity'], row['Resource'])] = (row['Duration_mean'], row['Duration_std'])

    # MINE THE RESOURCE SCHEDULE
    begin = min(df['Start Timestamp'])
    end = max(df['Complete Timestamp'])
    hr = (begin, begin + resource_schedule_timeunit)
    schedule = [[] for i in range(resource_schedule_repeat)]
    resource_presence = dict()  # nr of hours during which a resource was present
    for r in resources:
        resource_presence[r] = 0
    x = 0
    while hr[1] <= end:
        # Tasks are within the hour hr (or other timeunit if that is chosen), if the hour ends or begins between the start and end of the task
        tasks_in_hour = df[((df['Start Timestamp'] <= hr[0]) & (df['Complete Timestamp'] >= hr[0])) | (
                (df['Start Timestamp'] <= hr[1]) & (df['Complete Timestamp'] >= hr[1]))]
        resources_in_hour = tasks_in_hour['Resource'].unique()
        for r in resources_in_hour:
            resource_presence[r] += 1
        schedule[x % resource_schedule_repeat].append(len(resources_in_hour))
        x += 1
        hr = (hr[0] + datetime.timedelta(hours=1), hr[1] + datetime.timedelta(hours=1))
    for x in range(resource_schedule_repeat):
        schedule[x] = round(mean(schedule[x]))
    resource_weights = []
    for r in resources:
        resource_weights.append(resource_presence[r])

    #Adjust resource schedule based on the problem
    if problem_name == 'production' or problem_name == 'consulta' or problem_name == 'microsoft':
        schedule = [1+int(s + s) if 1+int(s+s) < len(list(resources)) else len(list(resources)) for s in schedule]


    # CREATE THE PROBLEM
    result = MinedProblem()
    result.schedule = schedule
    result.resource_weights = resource_weights
    result.task_types = list(task_types)  # The task types
    result.resources = list(resources)  # The resources
    result.initial_task_distribution = initial_task_distribution  # The initial task type distribution
    result.next_task_distribution = next_task_distribution  # The next task type distribution per task type
    result.mean_interarrival_time = mean_interarrival_time  # The interarrival time
    result.resource_pools = resource_pools  # The resource pool per task type
    result.processing_time_distribution = processing_time_distribution  # The processing time distribution per task_type/resource combination

    inspect_dataset(df)

    return result, df

def inspect_dataset(df, threshold=None):
    """
    Inspects the dataset and prints some statistics.

    :param df: a pandas dataframe.
    :param threshold: an integer threshold for the minimum occurrence count of an activity to be included in the df.
    """
    print("Dataset statistics:")
    print("Number of rows:", len(df))
    print("Number of cases:", len(df['Case ID'].unique()))
    print("Number of activities:", len(df['Activity'].unique()))
    print("Number of resources:", len(df['Resource'].unique()))
    print("Number of unique start timestamps:", len(df['Start Timestamp'].unique()))
    avg_completion_time = mean(
        (pd.to_datetime(df['Complete Timestamp']) - pd.to_datetime(df['Start Timestamp'])).dt.total_seconds())
    print("Average completion time (seconds):", avg_completion_time)
    print("Average completion time (hours):", avg_completion_time / 3600)
    print("Average completion time (days):", avg_completion_time / 3600 / 24)


    print("Start time:", min(df['Start Timestamp']))
    print("End time:", max(df['Complete Timestamp']))

    #For each unique activity, print its occurrence count
    print("Activity occurrence count:")
    excluded_activities = {}
    for activity in df['Activity'].unique():
        occ_count = len(df[df['Activity'] == activity])
        print(f'Number of occurrences of {activity}: {occ_count}')
        if threshold is not None and occ_count < threshold:
            print(f"Excluding activity {activity} with occurrence count {occ_count}")
            excluded_activities[activity] = False
        else:
            excluded_activities[activity] = True

    df['Start Timestamp'] = pd.to_datetime(df['Start Timestamp'])
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])

    #for each week in the dataset, print the number of unique users
    print("Number of unique users per week:")
    #for week in range(1, 53):
    #    print(f"Week {week} (dates from {df[df['Start Timestamp'].dt.week == week]['Start Timestamp'].min()} to {df[df['Start Timestamp'].dt.week == week]['Start Timestamp'].max()}: {len(df[df['Start Timestamp'].dt.week == week]['Resource'].unique())}")



    return excluded_activities

def plot_problem(problem, problem_name):
    """
    Plots several statistics of the problem.
    :param problem: the MinedProblem to plot.
    :param problem_name: the suffix to save the plots with.
    :return: the problem
    """


    #create plots folder if it does not exist

    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.figure()
    plt.bar(range(len(problem.schedule)), problem.schedule)
    plt.title("Resource schedule")
    plt.xlabel("Hour")
    plt.ylabel("Number of resources")
    plt.savefig(f"plots/{problem_name}_resource_schedule.png")

    plt.figure()
    plt.bar(range(len(problem.resource_weights)), problem.resource_weights)
    plt.title("Resource weights")
    plt.xlabel("Resource")
    plt.ylabel("Number of hours present")
    plt.savefig(f"plots/{problem_name}_resource_weights.png")

    plt.figure()
    plt.bar(range(len(problem.initial_task_distribution)), [x[0] for x in problem.initial_task_distribution])
    plt.title("Initial task distribution")
    plt.xlabel("Task type")
    plt.ylabel("Probability")
    plt.savefig(f"plots/{problem_name}_initial_task_distribution.png")

    plt.figure()
    for i, task in enumerate(problem.next_task_distribution):
        plt.bar([i]*len(problem.next_task_distribution[task]), [x[0] for x in problem.next_task_distribution[task]])
    plt.title("Next task distribution")
    plt.xlabel("Task type")
    plt.ylabel("Probability")
    plt.savefig(f"plots/{problem_name}_next_task_distribution.png")

    plt.figure()
    plt.bar(range(len(problem.resource_pools)), [len(problem.resource_pools[tt]) for tt in problem.resource_pools])
    plt.title("Resource pools")
    plt.xlabel("Task type")
    plt.ylabel("Number of resources")
    plt.savefig(f"plots/{problem_name}_resource_pools.png")

    plt.figure()
    for i, task in enumerate(problem.resource_pools):
        plt.bar([i]*len(problem.resource_pools[task]), [1]*len(problem.resource_pools[task]))
    plt.title("Resource pools")
    plt.xlabel("Task type")
    plt.ylabel("Resource")
    plt.savefig(f"plots/{problem_name}_resource_pools_2.png")

    plt.figure()
    for i, task in enumerate(problem.processing_time_distribution):
        plt.bar([i]*len(problem.processing_time_distribution[task]), [x for x in problem.processing_time_distribution[task]])
    plt.title("Processing time distribution")
    plt.xlabel("Task type")
    plt.ylabel("Mean processing time")
    plt.savefig(f"plots/{problem_name}_processing_time_distribution.png")

    return problem



if __name__ == "__main__":


    if not only_plot:
        if problem_name == '2012':
            log = pandas.read_csv('data/bpi2012/BPI_Challenge_2012 - clean.csv')
            excluded_activities = inspect_dataset(log, threshold=threshold)

            problem, df = mine_problem(log, problem_name='bpi2012', mean_interarrival_time_adjustment_factor=0.6, min_resource_pool_size=2, datetime_format="%Y-%m-%d %H:%M:%S")
            print(problem)

            with open(f'data/bpi{problem_name}_problem.pkl', 'wb') as f:
                pickle.dump(problem, f, protocol=pickle.HIGHEST_PROTOCOL)

            plot_problem(problem, problem_name)
        if problem_name == '2018':
            log = pandas.read_csv('data/bpi2018/2018_clean.csv')
            excluded_activities = inspect_dataset(log, threshold=threshold)

            problem, df = mine_problem(log, problem_name='bpi2018', min_resource_pool_size=3,
                                       task_type_filter=lambda x: excluded_activities[x],
                                       datetime_format="%Y-%m-%d %H:%M:%S")

            print(problem)

            with open(f'data/bpi{problem_name}_problem.pkl', 'wb') as f:
                pickle.dump(problem, f, protocol=pickle.HIGHEST_PROTOCOL)

            plot_problem(problem, problem_name)
        if problem_name == '2017':
            log = pandas.read_csv('data/bpi2017/BPI Challenge 2017 - clean.csv')
            excluded_activities = inspect_dataset(log, threshold=threshold)

            problem, df = mine_problem(log, problem_name='bpi2017', mean_interarrival_time_adjustment_factor=0.7, task_type_filter=lambda x: x in ['W_Complete application', 'W_Call after offers', 'W_Validate application', 'W_Call incomplete files', 'W_Handle leads', 'W_Assess potential fraud', 'W_Shortened completion'])

            with open(f'data/bpi{problem_name}_problem.pkl', 'wb') as f:
                pickle.dump(problem, f, protocol=pickle.HIGHEST_PROTOCOL)

            plot_problem(problem, problem_name)

        elif problem_name == 'fines':
            log = pandas.read_csv('data/fines/fines_cleaned.csv')
            excluded_activities = inspect_dataset(log, threshold=threshold)

            problem, df = mine_problem(log, problem_name='fines', datetime_format="%Y-%m-%d %H:%M:%S", max_tasks_per_case=20) #task_type_filter=lambda x: excluded_activities[x], min_tasks_per_case=4, min_resource_count=2, datetime_format="%Y-%m-%d %H:%M:%S")
            print(problem)

            inspect_dataset(df, threshold=None)

            with open(f'data/fines_problem.pkl', 'wb') as f:
                pickle.dump(problem, f, protocol=pickle.HIGHEST_PROTOCOL)

            plot_problem(problem, problem_name)



        elif problem_name == 'toloka':
            log = pandas.read_csv('data/toloka/assignments_cleaned.csv')

            log = log[log['Start Timestamp'] >= '2018-10-01']
            log = log[log['Complete Timestamp'] <= '2018-11-29']

            excluded_activities = inspect_dataset(log, threshold=threshold)

            problem, df = mine_problem(log, problem_name='toloka', min_resource_count=2, datetime_format="%Y-%m-%d %H:%M:%S", resource_schedule_repeat=168)


            with open(f'data/toloka_problem.pkl', 'wb') as f:
                pickle.dump(problem, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(problem.resource_pools)
            plot_problem(problem, problem_name)
            inspect_dataset(df, threshold=None)

        elif problem_name == 'consulta':
            log = pandas.read_csv('data/consulta2018/consulta_cleaned.csv')

            #log = log[log['Start Timestamp'] >= '2016-03-07']
            #log = log[log['Complete Timestamp'] <= '2016-04-25']
            #log = log[log['Activity'] != 'Avanzar recepcion documentos']

            excluded_activities = inspect_dataset(log, threshold=threshold)

            problem, df = mine_problem(log, problem_name='consulta', min_tasks_per_case=2, mean_interarrival_time_adjustment_factor=2, min_resource_pool_size=2, datetime_format="%Y-%m-%d %H:%M:%S", resource_schedule_timeunit=datetime.timedelta(hours=1), resource_schedule_repeat=168)


            with open(f'data/consulta.pkl', 'wb') as f:
                pickle.dump(problem, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(problem.resource_pools)
            plot_problem(problem, problem_name)
            inspect_dataset(df, threshold=None)

        elif problem_name == 'production':
            log = pandas.read_csv('data/production/productions_cleaned.csv')

            excluded_activities = inspect_dataset(log, threshold=threshold)

            included_activities = ['Turning & Milling', 'Turning & Milling Q.C.', 'Laser Marking', 'Lapping', 'Round Grinding', 'Final Inspection Q.C.', 'Packing', 'Turning Q.C.', 'Flat Grinding', 'Grinding Rework', 'Turning', 'Milling', 'Turn & Mill. & Screw Assem']

            problem, df = mine_problem(log, problem_name='production', min_tasks_per_case=2, min_resource_count=2, min_resource_pool_size=2,
                                       datetime_format="%Y-%m-%d %H:%M:%S", mean_interarrival_time_adjustment_factor=1,
                                       resource_schedule_timeunit=datetime.timedelta(hours=1), resource_schedule_repeat=168, task_type_filter=lambda x: x in included_activities)

            with open(f'data/{problem_name}.pkl', 'wb') as f:
                pickle.dump(problem, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(problem.resource_pools)
            plot_problem(problem, problem_name)
            inspect_dataset(df, threshold=None)

        elif problem_name == 'microsoft':
            log = pandas.read_csv('data/microsoft/microsoft_cleaned.csv')

            excluded_activities = inspect_dataset(log, threshold=threshold)

            problem, df = mine_problem(log, problem_name='microsoft', min_resource_count=2,
                                       datetime_format="%Y-%m-%d %H:%M:%S", mean_interarrival_time_adjustment_factor=1,
                                       resource_schedule_timeunit=datetime.timedelta(hours=1), resource_schedule_repeat=168)

            with open(f'data/{problem_name}.pkl', 'wb') as f:
                pickle.dump(problem, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(problem.resource_pools)
            plot_problem(problem, problem_name)
            inspect_dataset(df, threshold=None)

    else:
        if problem_name == '2012' or problem_name == '2017' or problem_name == '2018':
            problem_name = f"bpi{problem_name}"
        with open(f'data/{problem_name}_problem.pkl', 'rb') as f:
            problem = pickle.load(f)
        plot_problem(problem, problem_name)

    #Print the problem's task types number
    print("Number of task types:", len(problem.task_types))

    #Calculate and print the average case lenght in terms of activities, considering rhe initial task distribution and the next task distribution
    #The average case length is the average amount of tasks encountered before reaching the task None, given the initial task distribution and the next task distribution
    def calculate_expected_tasks(task, next_task_distribution):
        memo = {}
        stack = [(task, 1)]
        while stack:
            current_task, current_prob = stack.pop()
            if current_task in memo:
                continue
            if current_task not in next_task_distribution:
                memo[current_task] = 0
                continue
            expected_tasks = 0
            for prob, next_task in next_task_distribution[current_task]:
                if next_task is None:
                    expected_tasks += prob
                else:
                    expected_tasks += prob * (1 + memo.get(next_task, 0))
                    stack.append((next_task, prob))
            memo[current_task] = expected_tasks
        return memo[task]


    #Print the problem's resources number
    print("Number of resources:", len(problem.resources))

    #Print the problem's resource pools average size
    print("Average resource pool size:")
    for tt in problem.resource_pools:
        print(f"{tt}: {len(problem.resource_pools[tt])}")
    avg = np.mean([len(problem.resource_pools[tt]) for tt in problem.resource_pools])
    print("Average resource pool size:", avg)
    print("Standard deviation resource pool size:", np.sqrt(np.var([len(problem.resource_pools[tt]) for tt in problem.resource_pools])))

    #Print the resources schedule average size
    print("Average resource schedule size:", np.mean(problem.schedule))
    print("Standard deviation resource schedule size:", np.sqrt(np.var(problem.schedule)))

    #Print the resource weights average size
    print("Average resource weights size:", np.mean(problem.resource_weights))
    print("Standard deviation resource weights size:", np.sqrt(np.var(problem.resource_weights)))

    #Print the arrival rate
    print("Arrival rate:", 1/problem.mean_interarrival_time)