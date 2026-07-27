import time
import threading
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)


class PowerMonitor(threading.Thread):
    def __init__(self, handle, interval=0.05):
        super().__init__()
        self.handle = handle
        self.interval = interval
        self.samples = []
        self.running = True
        
    def run(self):
        while self.running:
            mW = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            self.samples.append(mW / 1000.0) # convert to Watts
            time.sleep(self.interval)
            
    def stop(self):
        self.running = False

# --- Your Code Region ---
monitor = PowerMonitor(handle, interval=0.01)
monitor.start()

# Simulate a GPU workload or code region
time.sleep(2.0) 

monitor.stop()
monitor.join()
pynvml.nvmlShutdown()

if monitor.samples:
    avg_power = sum(monitor.samples) / len(monitor.samples)
    print(f"Average Power during region: {avg_power:.2f} W")