import time, sys

for i in range(10):
    sys.stdout.write("\rDoing thing %i" % i)
    sys.stdout.flush()
    time.sleep(1)
