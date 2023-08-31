import concurrent.futures
import os
import multiprocessing
from getStats import main

def determine_workers():
    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    # Set a max limit to avoid excessive parallelism
    max_workers = min(num_cores * 2, 16)  # You can adjust the maximum as needed

    return max_workers

def process_batch(urls, temp_file, max_workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(main, url, temp_file) for url in urls]
        concurrent.futures.wait(futures)

# Determine the optimal number of workers
max_workers = determine_workers()

# Call your process function with the determined number of workers
process_batch(urls_to_process, temp_file, max_workers)
