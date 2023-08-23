import argparse
parser = argparse.ArgumentParser()
# change dir according to your own dataset dirs
# train/val
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=6, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')#0.1
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--lr_train_root', type=str, default='', help='the train images root')
parser.add_argument('--lr_val_root', type=str, default='', help='the val images root')
parser.add_argument('--save_path', type=str, default='', help='the path to save models and logs')
# test(predict)
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--test_path',type=str,default='/media/hjk/shuju/轻量级/data set/RGBT_test/',help='test dataset path')
opt = parser.parse_args()
