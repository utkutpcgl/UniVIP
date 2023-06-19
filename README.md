# UniVIP
Unofficial implementation of UniVIP

## 1. Generating bbox proposals

### (RECOMMENDED) Using already generated proposals to create filtered bbox proposals 

From the repo `https://github.com/Jiahao000/ORL/tree/master`, download the non-filtered object proposals generated with selective search.

*Repeat the steps below both for train2017 and unlabelled2017 COCO dataset:*
1. To spead up loading the json file (~5GB) convert it to pickle file with `UniVIP/generate_proposals/json_to_pkl.py` (specifiy the path with `file_path`).
2. Filter out small bounding boxes with `UniVIP/generate_proposals/post_filter_boxes.py` (specifiy the path with `TARGET_PKL`).
3. Enumerate the pickle file per image to sort them accordingly with `/home/kuartis-dgx1/utku/UniVIP/dataset/enumerate_bbox.py`. (specifiy the path with `PICKLE_FILE`).
4. Match boxes with image names with `/home/kuartis-dgx1/utku/UniVIP/dataset/name_pkl.py` (specifiy the path with `TARGET_PKL`). Note that you have to modify `ANNOTATION_INFO` with `instances_train2017.json` or `image_info_unlabeled2017.json` paths from the COCO dataset.

At the end you will have pkl files with bounding box proposals (filtered for 64 and 96 min_size according to UniVIP) and image names matched.

### (NOT RECOMMENDED) Re-generating bounding box proposal generation with selective search

* Replace the code in `selective_search_iou.py` and `/home/utku/Documents/ODTU/CENG502/project/ORL/openselfsup/datasets/selective_search.py` to apply iou_tresh together with selective search for any given min_size (96 and 64 for this paper.).

* Then generate the proposal boxes with selective search for min_size 96 and 64 (64 is used if there are not sufficient)

*NOTE*: If you first filter boxes ith 64 size and apply iou_thresh simultenously, some >=96 might be lost. Hence, using only 64 min_size to generate proposals can produce fewer 96 pixel boxes. If you apply 64 size filter and apply iou_thresh seperately for 96 and 64 pixel sizes, it will be equal to generating different proposal files with 64+iou_trehs and 96+iou_thresh. 64-filter is the fallback.

* Copy the python scripts in `ORL_files` to the repo ORL to run the command `bash dist_selective_search_single_gpu.sh configs/selfsup/orl/coco/stage2/selective_search_train2017.py univip_instances_train2017.json` in ORL (get from `https://github.com/Jiahao000/ORL/tree/2ad64f7389d20cb1d955792aabbe806a7097e6fb` and install runtime.txt dependendencies and install this for selective search `pip3 install opencv-contrib-python --upgrade`).

* You have to re-run the script above to generate proposals again for 64 and 96 min_size.


## GENERAL NOTES REGARDING THE PAPER AND IMPLEMENTATION

- The paper seems to be very similar to a previous paper (ORL) which is mentioned in the related works part of the UniVIP paper. In ORL, box with widht or height smaller than 96 are filtered, look here: https://github.com/Jiahao000/ORL/blob/2ad64f7389d20cb1d955792aabbe806a7097e6fb/openselfsup/datasets/correspondence.py#L52 . 
- This is the default selective search setting of UniVIP, but if there are not sufficient common objects (K), only the min_size is dropped from 96 to 64. The max ratio is 3 and the max_iou_threshold is 0.5 (there are the same both for ORL and UniVIP: https://github.com/Jiahao000/ORL/blob/2ad64f7389d20cb1d955792aabbe806a7097e6fb/configs/selfsup/orl/coco/stage2/r50_bs512_ep800_generate_all_correspondence.py#L61)
- The bounding box format of the proposals are (x,y,w,h), I am not sure what x and y are yet (center or upper left corner?)


- I had to change the OS mmap limitation x4 with `sudo sysctl -w vm.max_map_count=<new_value>` (`sudo sysctl -w vm.max_map_count=262120`)

# Maybe check these
- Check sinkhorn imp
- Maybe final BYOL params check
- Check max_trials for naive box (no need?)

# Findings

Instance proposals are not very different sometimes. The paper says that optimal transport applies contrastive learning (pull closer similar images and apart dissimilar ones). But my observations says it only closens similarr representations.

# My interpretation
sinkhorn
naive box generation
overlapping box generation (minimum 64)


# Difficulties
Number of images with edge smaller than 64: 2
Number of images with edge smaller than 70: 2
smaller_files are [PosixPath('/raid/utku/datasets/COCO_dataset/train2017/000000187714.jpg'), PosixPath('/raid/utku/datasets/COCO_dataset/train2017/000000363747.jpg')]

## Training the system
NaN values, clamping bicubic output
DDP setting.
Using GPU for dataset (speed up)
Very long train time


## 