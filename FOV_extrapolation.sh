cd tool
python video_completion.py \
       --mode video_extrapolation \
       --path ../data/tennis \
       --outroot ../result/tennis_extrapolation \
       --H_scale 2 \
       --W_scale 2 \
       --seamless


python ./tool/video_completion_modified.py \
--mode object_removal \
--path ../data/test_a/video_0000/frames_corr \
--path_mask ../data/test_a/video_0000/masks \
--outroot ../data/result_test_a/video_0000 \
--seamless \
--edge_guide


cd tool 
python video_completion_modified.py \
       --mode object_removal \
       --path ../data/tennis \
       --path_mask ../data/tennis_mask \
       --outroot ../result/tennis_removal_modified \
       --seamless \
       --edge_guide