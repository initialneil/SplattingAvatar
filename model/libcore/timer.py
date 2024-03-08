# Timer helper.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import torch
import time

CUDA_TIMERS = {}
CPU_TIMERS = {}

class CudaTimerStatus:
    def __init__(self) -> None:
        self.start_event = None
        self.end_event = None
        self.elapsed_time = 0
        self.elapsed_count = 0

    @property
    def elapsed_time_ms(self):
        return self.elapsed_time / (self.elapsed_count + 1e-5)

    def start(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def stop(self):
        self.end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_ms = self.start_event.elapsed_time(self.end_event)
        self.elapsed_time += elapsed_ms
        self.elapsed_count += 1

    def reset(self):
        self.start_event = None
        self.end_event = None
        self.elapsed_time = 0
        self.elapsed_count = 0

def startCudaTimer(key):
    # torch.cuda.synchronize()
    if not key in CUDA_TIMERS:
        CUDA_TIMERS[key] = CudaTimerStatus()
    
    CUDA_TIMERS[key].start()

def stopCudaTimer(key, print_count = 1):
    # torch.cuda.synchronize()
    if not key in CUDA_TIMERS:
        return
    
    CUDA_TIMERS[key].stop()
    
    if print_count <= 0:
        return

    if CUDA_TIMERS[key].elapsed_count >= print_count:
        ave_ms = CUDA_TIMERS[key].elapsed_time_ms
        print('[CUDA Timer] %s takes %.4f ms' % (key, ave_ms))

        CUDA_TIMERS[key].reset()



class CpuTimerStatus:
    def __init__(self) -> None:
        self.start_time = None
        self.end_time = None
        self.elapsed_time = 0
        self.elapsed_count = 0

    @property
    def elapsed_time_ms(self):
        return self.elapsed_time / (self.elapsed_count + 1e-5)

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        elapsed_ms = (self.end_time - self.start_time) * 1000.
        self.elapsed_time += elapsed_ms
        self.elapsed_count += 1

    def reset(self):
        # self.start_time = None
        # self.end_time = None
        self.elapsed_time = 0
        self.elapsed_count = 0

def startCpuTimer(key):
    if not key in CPU_TIMERS:
        CPU_TIMERS[key] = CpuTimerStatus()
    
    CPU_TIMERS[key].start()

def stopCpuTimer(key, print_count = 1):
    if not key in CPU_TIMERS:
        return
    
    CPU_TIMERS[key].stop()
    
    if print_count <= 0:
        return

    if CPU_TIMERS[key].elapsed_count >= print_count:
        ave_ms = CPU_TIMERS[key].elapsed_time_ms
        print('[CPU Timer] %s takes %.4f ms' % (key, ave_ms))

        CPU_TIMERS[key].reset()
