import os 
import sys 
import time 
import psutil 
import ray 
start_time = time.time()

# print("Hello World!")
# print("Python version: ", sys.version)
# print("Python path: ", sys.executable)
# print("Current working directory: ", os.getcwd())
print(f"CPU Affinity: {psutil.Process().cpu_affinity()}")
ray.init(
    include_dashboard=False,
)
# print(f"\nAvailable resources: {ray.available_resources()}")
# print(f"\nCluster resources: {ray.cluster_resources()}")

@ray.remote(num_cpus=1)
def hello_world(i):
    print(f"Task {i} started!")
    start_t = time.time()
    result = 0
    for _ in range(5000000):
        result = (result + 1) * 2  # Some dummy compu
        result = result / 2
        result = round(result, 5)
        if result == 0:
            print("Something went wrong!")
    return f"Task {i} done in {time.time()-start_t:.3f} seconds!"

# # Synchronus - BAD!
# # print(ray.get(hello_world.remote()))
# # print(ray.get(hello_world.remote()))
# # print(ray.get(hello_world.remote()))

# Asynchronus - GOOD!
result_ids = [hello_world.remote(i) for i in range(10)]
for result_id in result_ids:
    print(ray.get(result_id))

print(f"\nCompleted all tasks in {time.time() - start_time:.2f} seconds.")
