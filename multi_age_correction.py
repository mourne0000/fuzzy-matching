import pandas as pd
from datetime import datetime
from multiprocessing import Process, Queue, cpu_count
import time
import numpy as np

def correct_age_gender(df):
    # correct gender first
    df["gender"] = df.groupby("ID")["gender"].transform(
        lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x
    )

    # prepare date and year
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    # allocate batches by ID groups
    unique_ids = df['ID'].unique()
    batch_size = 100
    id_batches = [unique_ids[i:i+batch_size] for i in range(0, len(unique_ids), batch_size)]

    # create groups
    batched_groups = []
    for id_batch in id_batches:
        batch_df = df[df['ID'].isin(id_batch)]
        batched_groups.append(batch_df)

    # setup multiprocessing queues
    task_queue = Queue()
    result_queue = Queue()

    # fill task queue with batched groups
    for batch in batched_groups:
        task_queue.put(batch)

    # number of workers
    num_workers = max(1, cpu_count() - 2)

    # start worker processes
    processes = []
    for _ in range(num_workers):
        p = Process(target=batch_worker, args=(task_queue, result_queue))
        p.start()
        processes.append(p)

    # end the queue
    for _ in range(num_workers):
        task_queue.put(None)

    # collect processed batches
    processed_batches = []
    for _ in range(len(batched_groups)):
        processed_batch = result_queue.get()
        processed_batches.append(processed_batch)

    # concatenate and clean
    final_df = pd.concat(processed_batches)
    final_df = final_df.drop(columns=['year']).sort_index()

    # wait for all workers
    for p in processes:
        p.join()

    return final_df

# process every batch which includes a couple of IDs
def batch_worker(task_queue, result_queue):
    while True:
        batch = task_queue.get()
        if batch is None:
            break
        processed_groups = []
        for _, group in batch.groupby('ID'):
            processed_group = process_single_id(group)
            processed_groups.append(processed_group)
        
        result_queue.put(pd.concat(processed_groups))

def process_single_id(group):

    group = group.sort_values('year').reset_index(drop=True)
    ages = group['age'].copy()
    years = group['year']

    for i in range(len(group)):
        if pd.isna(ages[i]):
            prev_age, prev_year = find_previous_value(ages, years, i)
            next_age, next_year = find_next_value(ages, years, i)

            if prev_age is not None and next_age is not None:
                year_diff = next_year - prev_year
                if year_diff == 0:
                    if prev_age == next_age:
                        ages[i] = prev_age
                    else:
                        ages[i] = (prev_age + next_age) // 2
                else:
                    exact_age = prev_age + ((years[i] - prev_year) / year_diff) * (next_age - prev_age)
                    ages[i] = int(round(exact_age))
            elif prev_age is not None:
                ages[i] = prev_age + (years[i] - prev_year)
            elif next_age is not None:
                ages[i] = next_age - (next_year - years[i])

# cope with the first and last id corner case
    first_valid_idx = ages.first_valid_index()
    if first_valid_idx is not None and first_valid_idx > 0:
        first_valid_age = ages[first_valid_idx]
        first_valid_year = years[first_valid_idx]
        for i in range(first_valid_idx-1, -1, -1):
            ages[i] = first_valid_age - (first_valid_year - years[i])

    last_valid_idx = ages.last_valid_index()
    if last_valid_idx is not None and last_valid_idx < len(ages)-1:
        last_valid_age = ages[last_valid_idx]
        last_valid_year = years[last_valid_idx]
        for i in range(last_valid_idx+1, len(ages)):
            ages[i] = last_valid_age + (years[i] - last_valid_year)

    group['age'] = ages.astype('Int64')
    return group

def find_previous_value(ages, years, current_idx):
    for i in range(current_idx-1, -1, -1):
        if not pd.isna(ages[i]):
            return ages[i], years[i]
    return None, None

def find_next_value(ages, years, current_idx):
    for i in range(current_idx+1, len(ages)):
        if not pd.isna(ages[i]):
            return ages[i], years[i]
    return None, None

if __name__ == '__main__':
    total_start = time.perf_counter()
    df = pd.read_csv("1health_data.csv")
    processed_df = correct_age_gender(df)
    processed_df.to_csv("multi_processed_health_data.csv", index=False)
    total_end = time.perf_counter()
    total_run_time = total_end - total_start
    print(f"Total run time: {total_run_time:.5f} seconds")
    print(processed_df.head(20))