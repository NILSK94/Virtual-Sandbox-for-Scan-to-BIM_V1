"""Read ALL selected points in bounded blocks, even for a 20-GB source.

points is only the interactive working sample. iter_point_blocks() reads the
full normalized data, respects the selected region, and returns global IDs.
Choose a focused region if the complete scan exceeds the experiment time limit.
"""
import numpy as np
count=0
center_sum=np.zeros(3)
low=np.full(3,np.inf)
high=np.full(3,-np.inf)
for block,block_colors,global_ids in iter_point_blocks():
    count+=len(block)
    center_sum+=block.sum(axis=0)
    low=np.minimum(low,block.min(axis=0))
    high=np.maximum(high,block.max(axis=0))
if not count:raise ValueError("Die gewählte Region ist leer.")
result={"method":"full selected source, blockwise numerical reduction", "point_count":count,
        "working_sample_count":len(points),"center_m":center_sum/count,
        "min_m":low,"max_m":high,"extent_m":high-low}
