from aether.schedule import make_learning_rate_schedule


schedule = make_learning_rate_schedule(
    peak_learning_rate=1e-3,
    min_learning_rate=1e-4,
    warmup_steps=100,
    total_steps=1000,
)


def learning_rate_at(step):
    return float(schedule(step))


# starts at zero, reaches the peak exactly when warmup ends
assert learning_rate_at(0) == 0.0
assert abs(learning_rate_at(50) - 5e-4) < 1e-9
assert abs(learning_rate_at(100) - 1e-3) < 1e-9

# after the peak it only goes down, and ends at the minimum
rates = [learning_rate_at(step) for step in range(100, 1001, 50)]
assert all(earlier >= later for earlier, later in zip(rates, rates[1:]))
assert abs(learning_rate_at(1000) - 1e-4) < 1e-9

print("schedule works")
