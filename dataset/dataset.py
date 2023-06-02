# Add image index to each image boxes (for multiprocessing.)
# How do I construct the dataloader and set the batch (size)?


# 1. Randomly crop two areas of the image.
# 2. If they have at least K object regions in the overlapping region T return the scenes s1 and s2 (they are our targets)
# 3. We return instances directly since we dont know which instances to match directly?

# Calculate each loss based on the method (Lscene -> BYOL, Ls-i, Li-i)