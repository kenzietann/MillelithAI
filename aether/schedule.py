import mlx.optimizers as optim


def make_learning_rate_schedule(peak_learning_rate, min_learning_rate, warmup_steps, total_steps):
    """Climb linearly to the peak, then glide down along a cosine curve.

    Warmup because the weights start out random and the early gradients are wild.
    Decay because small steps near the end help the loss settle into the valley.
    """
    warmup = optim.linear_schedule(0.0, peak_learning_rate, warmup_steps)
    decay = optim.cosine_decay(peak_learning_rate, total_steps - warmup_steps, min_learning_rate)
    return optim.join_schedules([warmup, decay], [warmup_steps])
