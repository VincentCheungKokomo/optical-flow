python main.py \
--inference \
--model FlowNet2 \
--save_flow \
--inference_dataset ImagesFromFolder \
--inference_dataset_root /home/yons/code/data/tennis \
--resume /home/yons/code/weight/FlowNet2_checkpoint.pth.tar \
--save ./test/output/  
