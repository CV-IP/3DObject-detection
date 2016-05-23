export LD_LIBRARY_PATH=/home/jikai/ganyk/usr/lib:/home/jikai/ganyk/usr/libs:/home/jikai/ganyk/anaconda2/lib/:$LD_LIBRARY_PATH
LOG=../log/train-`date +%Y-%m-%d-%H-%M-%S`.log

./build/tools/caffe train \
	--solver ../models/solver.prototxt \
	--weights /home/jikai/ganyk/vgg/VGG_ILSVRC_16_layers.caffemodel \
	-gpu 0 2>&1 | tee $LOG
	
