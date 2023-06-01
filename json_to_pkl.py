import os
import ijson
import pickle
import tqdm
from pathlib import Path

file_path = '/home/kuartis-dgx1/utku/UniVIP/COCO/train2017_selective_search_proposal.json'
file_size = os.path.getsize(file_path)

data = {"bbox": []}

with tqdm.tqdm(total=file_size, unit='B', unit_scale=True, desc='Loading file') as pbar:
    with open(file_path, 'rb') as file:
        parser = ijson.parse(file)
        for prefix, event, value in parser:
            if prefix == 'bbox.item':
                data["bbox"].append(value)
            current_position = file.tell()
            pbar.update(current_position - pbar.n)

# Save data as pickle
with open(f'{Path(file_path).stem}.pkl', 'wb') as f:
    pickle.dump(data, f)

