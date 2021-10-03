ffmpeg -i final_gt.mp4  -i final_k=1.mp4  -lavfi psnr="stats_file=psnr.log" -f null -
ffmpeg -i final_gt.mp4  -i final_k=2.mp4  -lavfi psnr="stats_file=psnr.log" -f null -
ffmpeg -i final_gt.mp4  -i final_k=4.mp4  -lavfi psnr="stats_file=psnr.log" -f null -
ffmpeg -i final_gt.mp4  -i final_k=8.mp4  -lavfi psnr="stats_file=psnr.log" -f null -
ffmpeg -i final_gt.mp4  -i final_k=16.mp4  -lavfi psnr="stats_file=psnr.log" -f null -
ffmpeg -i final_gt.mp4  -i final_k=32.mp4  -lavfi psnr="stats_file=psnr.log" -f null -

ffmpeg -i final_gt.mp4  -i final_k=1.mp4  -lavfi ssim="stats_file=ssim.log" -f null -
ffmpeg -i final_gt.mp4  -i final_k=2.mp4  -lavfi ssim="stats_file=ssim.log" -f null -
ffmpeg -i final_gt.mp4  -i final_k=4.mp4  -lavfi ssim="stats_file=ssim.log" -f null -
ffmpeg -i final_gt.mp4  -i final_k=8.mp4  -lavfi ssim="stats_file=ssim.log" -f null -
ffmpeg -i final_gt.mp4  -i final_k=16.mp4  -lavfi ssim="stats_file=ssim.log" -f null -
ffmpeg -i final_gt.mp4  -i final_k=32.mp4  -lavfi ssim="stats_file=ssim.log" -f null -