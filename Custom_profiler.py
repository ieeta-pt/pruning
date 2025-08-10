from pytorch_lightning.profilers import Profiler

import matplotlib.pyplot as plt
import time

plt.style.use('ggplot')

class Custom_Profiler(Profiler):
    def __init__(self, functions_to_profile: list = []):
        super().__init__()

        self.functions_to_profile = functions_to_profile

        self.records = {func: {"total_time": 0.0, "num_calls": 0} for func in self.functions_to_profile}
        self.current_timers = {}

    def start(self, action_name: str):
        if action_name in self.functions_to_profile:
            self.current_timers[action_name] = time.time()

    def stop(self, action_name: str):
        if action_name in self.functions_to_profile and action_name in self.current_timers:
            duration = time.time() - self.current_timers[action_name]
            self.records[action_name]['total_time'] += duration
            self.records[action_name]['num_calls'] += 1
            del self.current_timers[action_name]

    def summary(self):
        """Returns a formatted summary of profiling results."""
        summary_str = "\n🔥 Custom Profiler Report 🔥\n"
        summary_str += f"{'Function':<20}{'Calls':<10}{'Total Time (s)':<15}{'Avg Time per Call (s)':<20}\n"
        summary_str += "-" * 65 + "\n"

        for func, stats in self.records.items():
            num_calls = stats["num_calls"]
            total_time = stats["total_time"]
            avg_time = total_time / num_calls if num_calls > 0 else 0.0
            summary_str += f"{func:<20}{num_calls:<10}{total_time:<15.6f}{avg_time:<20.6f}\n"
        
        return summary_str

    def end(self):
        print(self.summary())

    def profile(self, func):
        def wrapper(*args, **kwargs):
            self.start(func.__name__)
            result = func(*args, **kwargs)
            self.stop(func.__name__)
            return result
        return wrapper

    # def __enter__(self):
    #     """Required for `Trainer(profiler=...)` to work without errors."""
    #     return self  # Simply return the profiler instance

    # def __exit__(self, exc_type, exc_value, traceback):
    #     """Called when exiting the 'with' block."""
    #     self.end()  # Automatically print the profiler summary at the end
