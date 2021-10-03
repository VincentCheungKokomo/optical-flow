python ./tool/video_completion_modified.py \
       --mode object_removal \
       --path ./data/tennis \
       --path_mask ./data/tennis_mask \
       --outroot ./result/tennis_removal_modified_mixed_precision \
       --mixed_precision \
       --seamless
