Write a Python script to reindex item IDs across multiple domains (all amazon domains) into one unified global ID space.

Requirements:
1. The script should process domains in a fixed user-specified order.
2. For each domain, assign new item IDs starting from the current global offset.
3. IDs must be continuous across domains:
   - Domain 1: from 0 to N_1 - 1
   - Domain 2: from N_1 to N_1 + N_2 - 1
   - Domain 3: from N_1 + N_2 to N_1 + N_2 + N_3 - 1
   - and so on
4. For each domain:
   - load the original item ID file from its embedding directory
   - verify that the number of items matches the expected count
   - generate a new `np.array` of reindexed global item IDs with dtype `np.int64`
   - save it as `shard_00000_ids_incremental.npy` under that domain’s embed path
5. The script should use hard-coded paths and metadata in the code, not command-line arguments.
6. Print a summary for each domain, including:
   - domain name
   - original file path
   - output file path
   - item count
   - assigned ID range
   - first 10 and last 10 new IDs
7. Use safe checks:
   - raise an error if the original `.npy` file does not exist
   - raise an error if the loaded array is not 1D
   - raise an error if the loaded item count does not match the expected count

Input format:
A hard-coded list of domain configs, where each config contains:
- `name`: domain name
- `expected_num_items`: expected number of items
- `domain_path`: directory containing `shard_00000_ids.npy`

Output:
For each domain, save:
`<domain_path>/shard_00000_ids_incremental.npy`

Please write clean, runnable Python code using only `os` and `numpy`.