
ffmpeg -f image2 -i /home/yons/code/data/tennis/%5d.png  -pix_fmt yuv420p  -t 5 test.mp4

ffmpeg  -r 12 -i /home/yons/code/data/tennis/%5d.png  -pix_fmt yuv420p  test.mp4