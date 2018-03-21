import time, random

random.seed()

t0 = time.perf_counter()
time.sleep(random.expovariate(0.5))
print(str(time.perf_counter() - t0))
