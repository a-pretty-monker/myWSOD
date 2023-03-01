conda activate torch17
cd /data/lijiaxin/myWSOD
python tools/train_net_step.py --dataset voc2007     --cfg configs/vgg16_voc2007_ours.yaml --nw 4 --iter_size 4 --use_tfboard
nohup python tools/train_net_step.py --dataset voc2007     --cfg configs/vgg16_voc2007_ours.yaml --nw 4 --iter_size 4 --use_tfboard > my2.log 2>&1 &