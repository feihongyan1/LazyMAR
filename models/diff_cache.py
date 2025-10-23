def init_cache(diffloss_d, num_sample_step, cache_type):
    return {
        -1: {index: {} for index in range(diffloss_d)}
    }