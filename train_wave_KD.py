import os
import torch
import torch.nn.functional as F

from datetime import datetime
from torchvision.utils import make_grid

from networks.Wavenet import WaveNet
from rgbt_dataset_KD import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from config import opt
from torch.cuda import amp
import pytorch_iou

# set the device for training
cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

# build the model


model = WaveNet()
model.load_pre()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path
train_dataset_path = opt.lr_train_root
image_root = train_dataset_path + '/RGB/'
depth_root = train_dataset_path + '/T/'
gt_root = train_dataset_path + '/GT/'
bound_root = train_dataset_path + '/bound/'
gt2_root = train_dataset_path + '/GT_SwinNet/'
val_dataset_path = opt.lr_val_root
val_image_root = val_dataset_path + '/RGB/'
val_depth_root = val_dataset_path + '/T/'
val_gt_root = val_dataset_path + '/GT/'
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')

train_loader = get_loader(image_root, gt_root,depth_root,bound_root,gt2_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
# print(len(train_loader))
test_loader = test_dataset(val_image_root, val_gt_root,val_depth_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info(save_path + "Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed



def joint_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)



# 超参数
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0
Sacler = amp.GradScaler()

# train function
length = 821

def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    # model2.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts,depths,bound,gts2) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            # ima = images
            # dep = depths
            images = images.cuda()
            # print(images.shape)
            depths = depths.cuda()
            gts = gts.cuda()
            bound = bound.cuda()
            gts2 = gts2.cuda()
            _,_,w_t,h_t = gts.size()

            s1, s2, s3, s4, s1_sig, s2_sig, s3_sig, s4_sig, edge, latent_loss =model(images, depths)
            '''
            We directly use the predicts from the teacher model, but you can also use this type to load teacher model.
            
            t1,t2, t3, t4, t1_sig, t2_sig, t3_sig, t4_sig, tedge, tlatent_loss = model_T(images, depths)
            t1, t2 = model_T(images, depths)
            s1, s2, s3,s4,s1_sig, s2_sig, s3_sig,s4_sig= model(images, depths)  # , self.sigmoid(s5)
            target,_ = model2(ima, dep)
            t1 = t1.cuda()
            m = nn.Sigmoid()
            t1 = m(t1)
            t1 = t1.detach()

            t2 = t2.cuda()
            m = nn.Sigmoid()
            t2 = m(t2)
            t2 = t2.detach()
            print('latent_loss',latent_loss)
            '''
            loss1 = CE(s1, gts) + IOU(s1_sig, gts)
            loss2 = CE(s2, gts) + IOU(s2_sig, gts)
            loss3 = CE(s3, gts) + IOU(s3_sig, gts)
            loss4 = CE(s4, gts) + IOU(s4_sig, gts)
            loss5 = CE(edge, bound)
            anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
            loss6 = 0.1 * anneal_reg * latent_loss
            loss7 = CE(s1_sig, gts2) + IOU(s1_sig, gts2)

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
            loss.backward()


            clip_gradient(optimizer, opt.clip)

            optimizer.step()
            step += 1
            epoch_step = epoch_step +1
            loss_all =loss_all + loss.item()
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}],W*H [{:03d}*{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch+1, opt.epoch, w_t, h_t, i, total_step, loss.item()))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                             format(epoch+1, opt.epoch, i, total_step, loss.item()))
                writer.add_scalar('Loss/total_loss', loss, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data,1,normalize=True)
                writer.add_image('train/RGB',grid_image, step)
                grid_image = make_grid(depths[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('train/Ti', grid_image, step)

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch+1, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch+1) % 10 == 0 or (epoch+1) == opt.epoch:
            torch.save(model.state_dict(), save_path + 'RES34_1_epoch_{}_test.pth'.format(epoch+1))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'RES34_1_epoch_{}_test.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

#

# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name = test_loader.load_data()
            gt = gt.cuda()
            image = image.cuda()
            depth = depth.cuda()

            res = model(image, depth)
            res = torch.sigmoid(res[0])

            res = (res-res.min())/(res.max()-res.min()+1e-8)
            mae_train =torch.sum(torch.abs(res-gt))*1.0/(torch.numel(gt))
            mae_sum = mae_train.item()+mae_sum

        mae = mae_sum / test_loader.size # mae / length:.8f

        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'RES34_1_best_mae_test.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))



if __name__ == '__main__':
    print("Start train...")
    start_time = datetime.now()
    for epoch in range(opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
    finish_time = datetime.now()
    h, remainder = divmod((finish_time - start_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(time)