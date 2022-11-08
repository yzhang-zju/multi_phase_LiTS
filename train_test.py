import warnings
import datetime
import torch
import torch.nn as nn
import os
import numpy as np
from skimage.io import imsave
from data_loader import FLLs_train, FLLs_val
from progress_bar import ProgressBar
import cv2

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from networks.network import Network
import math

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_GPU else "cpu")
num_workers = 8

def save_checkpoint(state, fold, filename, SAVE=False):
    if SAVE:
        torch.save(state, os.path.join(fold, filename))
        print('save model in specific epoch ', filename)

def soft_dice(input_, target):
    smooth = 1.

    input_flat = input_.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = input_flat * target_flat

    dice = (2 * intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)

    return dice

def dice_coef(pred_label, gt_label):
    # list of classes
    c_list = np.unique(gt_label)

    dice_c = []
    for c in range(1,len(c_list)): # dice not for bg
        # intersection
        ints = np.sum(((pred_label == c_list[c]) * 1) * ((gt_label == c_list[c]) * 1))
        # sum
        sums = np.sum(((pred_label == c_list[c]) * 1) + ((gt_label == c_list[c]) * 1)) + 0.0001
        dice_c.append((2.0 * ints) / sums)

    return dice_c

def train(param_set, model):
    folder = param_set['model']+datetime.datetime.now().strftime('%Y-%m-%d-%H')
    save_dir = param_set['result_dir'] + folder
    ckpt_dir = save_dir + '/checkpoint'
    log_dir = save_dir + '/log'
    test_result_dir = save_dir + '/testResult'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.mkdir(ckpt_dir)
        os.mkdir(test_result_dir)
    for file in os.listdir(log_dir):
        print('removing ' + os.path.join(log_dir, file))
        os.remove(os.path.join(log_dir, file))

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p:p.requires_grad,model.parameters()),lr = 5e-4)  # define the optimizer

    epoch_save = param_set['epoch_save']
    num_epochs = param_set['epoch']
    train_batch = ['batch2', 'batch3', 'batch4', 'batch5']

    iter_count = 0
    train_loader = DataLoader(FLLs_train(param_set['imgdir'],train_batch),num_workers=num_workers, batch_size=param_set['batch_size'], shuffle=True)
    print('steps per epoch:', len(train_loader))
    #####train#####
    model.train()
    for epoch in range(num_epochs+1):
        train_progressor = ProgressBar(mode='Train',epoch=epoch,
                                       total_epoch=num_epochs,
                                       model_name=param_set['model'],
                                       total = len(train_loader))

        iter_loss = 0
        iter_dice = 0
        for step, (images_pv, images_art,labels) in enumerate(train_loader):
            train_progressor.current = step
            model.train()
            if USE_GPU:
                images_pv = images_pv.cuda(non_blocking=True)  # inputs to GPU
                images_art = images_art.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            outputs, init_outputs = model(images_pv, images_art)

            tumor_map = F.softmax(outputs, dim=1).data.cpu()
            out_dice = soft_dice(tumor_map[:, 1].cpu(), labels.cpu().float())
            iter_dice += out_dice.cpu()
            loss = criterion(outputs, labels) + criterion(init_outputs, labels)
            iter_loss += float(loss.item())

            train_progressor.current_loss = loss.item()
            train_progressor.current_dice = out_dice.cpu()

            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backward
            optimizer.step()  # optimize
            train_progressor()

            iter_count += 1
            # clear cache
            del images_pv, images_art, labels, loss, outputs, tumor_map, out_dice
            torch.cuda.empty_cache()

        train_progressor.done()

        ####eval####
        test_batch = ['batch1']
        val_loader = DataLoader(FLLs_val(param_set['imgdir'], test_batch), num_workers=num_workers,
                                batch_size=param_set['batch_size'], shuffle=False)

        valid_result = evaluate(val_loader, model, criterion, epoch)
        print("epoch {epoch}, validation dice {val_dice}".format(epoch=epoch, val_dice=valid_result))  #for debug
        with open(save_dir + "/validation_log.txt", "a") as f:
            print("epoch {epoch}, validation dice {val_dice}".format(epoch=epoch, val_dice=valid_result), file=f)

        ####save checkpoint###
        filename = "{model}-{epoch:03}-{step:04}-{dice}.pth".format(model=param_set['model'], epoch=epoch,
                                                                 step=step,dice=valid_result)
        save_checkpoint(model.state_dict(),
                            ckpt_dir,
                            filename,
                            SAVE=(epoch>0 and epoch % epoch_save == 0 or epoch==num_epochs))


def evaluate(val_loader, model, criterion, epoch):
    val_progressor = ProgressBar(mode='Val',epoch=epoch,
                                 total_epoch=param_set['epoch'],
                                 model_name=param_set['model'],
                                 total=len(val_loader))

    model.cuda()
    model.eval()
    with torch.no_grad():
        val_iter_loss = 0
        val_iter_dice = 0
        for step, (images_pv,images_art, labels) in enumerate(val_loader):
            val_progressor.current = step
            if USE_GPU:
                images_pv = images_pv.cuda()
                images_art = images_art.cuda()
                labels = labels.cuda()
            #compute the output
            outputs, _ = model(images_pv,images_art)
            val_loss = criterion(outputs, labels)
            val_out_soft = F.softmax(outputs,dim=1).data.cpu()
            val_dice = soft_dice(val_out_soft[:,1],labels.cpu().float())
            val_iter_loss += float(val_loss.item())
            val_iter_dice += val_dice.cpu()

            val_progressor.current_loss = val_loss.item()
            val_progressor.current_dice = val_dice.cpu()
            val_progressor()
            # clear cache
            del images_pv, images_art, labels, outputs, val_loss, val_dice
            torch.cuda.empty_cache()
        val_progressor.done()

        val_epoch_dice = val_iter_dice / len(val_loader)
        return val_epoch_dice

def predict(param_set, model):
    ckpt_dir = param_set['model_loc']
    folder = param_set['folder']
    save_dir = param_set['result_dir'] + folder

    dice_dir = save_dir + '/dice'
    test_result_dir = save_dir + '/testResult'
    test_batch = ['batch1']
    pv_test_dir = param_set['testdir']
    art_test_dir = pv_test_dir.replace('PV', 'ART')

    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)
    if not os.path.exists(dice_dir):
        os.mkdir(dice_dir)

    print(ckpt_dir)
    model_name = ckpt_dir.split('\\')[-1]
    model.load_state_dict(torch.load(ckpt_dir))
    model.eval()
    print('Current model: ',model_name)

    model_test_dir = os.path.join(test_result_dir,model_name[:-4])
    # save
    print(test_result_dir)
    print(model_name)
    print(model_test_dir)
    if not os.path.exists(model_test_dir):
        os.mkdir(model_test_dir)

    print('Applying on: ' + test_batch[0])

    files = os.listdir(os.path.join(pv_test_dir,'img'))
    files.sort()
    for file in files:
        if file.endswith('.png'):
            with open(os.path.join(pv_test_dir, 'img', file), 'rb') as f:
                image_pv = cv2.imread(os.path.join(pv_test_dir, 'img', file))
            with open(os.path.join(art_test_dir, 'img', file), 'rb') as f:
                image_art = cv2.imread(os.path.join(art_test_dir, 'img', file))

            image_pv = np.asarray(image_pv, np.float32) / 255.0
            image_art = np.asarray(image_art, np.float32) / 255.0

            # outputs = model_apply_full(model, image_pv, image_art)  #fast test on full image
            outputs = model_apply(model, image_pv, image_art, param_set['ins'], file.split('.')[0]) #test on cropped patches and stitch the results
            out_str = model_test_dir + '/' + file
            #output.save(out_str)
            imsave(out_str, np.asarray(outputs * 255., np.uint8))
    return


def model_apply_full(model,image_pv, image_art):
    tI_pv = np.transpose(image_pv,[2,0,1])
    tI_pv = torch.Tensor(tI_pv.copy()).cuda().unsqueeze(0)
    tI_art = np.transpose(image_art, [2, 0, 1])
    tI_art = torch.Tensor(tI_art.copy()).cuda().unsqueeze(0)

    pred, _ = model(tI_pv, tI_art)
    prob = F.softmax(pred,dim=1).squeeze(0).data.cpu().numpy()
    pmap_img = prob[1]
    return pmap_img

def model_apply(model,image_pv,image_art,ins,file):
    caseNum = file[3:6]

    avk = 4
    nrotate = 1
    wI = np.zeros([ins, ins])
    pmap = np.zeros([image_pv.shape[0], image_pv.shape[1]])
    avI = np.zeros([image_pv.shape[0], image_pv.shape[1]])
    for i in range(ins):
        for j in range(ins):
            dx = min(i, ins - 1 - i)
            dy = min(j, ins - 1 - j)
            d = min(dx, dy) + 1
            wI[i, j] = d
    wI = wI / wI.max()

    # get centroid tumor z
    centroid_path = '/data/zy/5type_data/newdata/xyy/myData/Multi/Multi_liverROI/centroid_tumor/'
    cen_slices = np.loadtxt(os.path.join(centroid_path, caseNum + '.txt'), delimiter='\n')
    cen_slices = np.array(cen_slices, dtype='int')
    cen_pv_x = cen_slices[0]  # pv center x
    cen_pv_y = cen_slices[1]  # pv center y
    cen_art_x = cen_slices[3]  # art center
    cen_art_y = cen_slices[4]  # art center

    for i1 in range(math.ceil(float(avk) * (float(image_pv.shape[0]) - float(ins)) / float(ins)) + 1):
        for j1 in range(math.ceil(float(avk) * (float(image_pv.shape[1]) - float(ins)) / float(ins)) + 1):

            # pv start and end index
            insti = math.floor(float(i1) * float(ins) / float(avk))
            instj = math.floor(float(j1) * float(ins) / float(avk))
            inedi = insti + ins
            inedj = instj + ins

            # art start and end index
            insti_art = max(insti + cen_art_x - cen_pv_x, 0)
            instj_art = max(instj + cen_art_y - cen_pv_y, 0)
            inedi_art = insti_art + ins
            inedj_art = instj_art + ins

            if inedi > image_pv.shape[0]:
                inedi = image_pv.shape[0]
                insti = inedi - ins
            if inedj > image_pv.shape[1]:
                inedj = image_pv.shape[1]
                instj = inedj - ins
            if inedi_art > image_art.shape[0]:
                inedi_art = image_art.shape[0]
                insti_art = inedi_art - ins
            if inedj_art > image_art.shape[1]:
                inedj_art = image_art.shape[1]
                instj_art = inedj_art - ins

            small_pmap = np.zeros([ins, ins])

            for i in range(nrotate):
                small_in_pv = image_pv[insti:inedi, instj:inedj]
                small_in_art = image_art[insti_art:inedi_art, instj_art:inedj_art]
                small_in_pv = np.rot90(small_in_pv,i)
                small_in_art = np.rot90(small_in_art,i)

                tI_pv = np.transpose(small_in_pv,[2,0,1])
                tI_art = np.transpose(small_in_art,[2,0,1])
                tI_pv = torch.Tensor(tI_pv.copy()).cuda()
                tI_art = torch.Tensor(tI_art.copy()).cuda()
                pred= model(tI_pv.unsqueeze(0), tI_art.unsqueeze(0))

                prob = F.softmax(pred,dim=1).squeeze(0).data.cpu().numpy()
                small_out = prob[1]
                small_out = np.rot90(small_out,-i)

                small_pmap = small_pmap + np.array(small_out)

            small_pmap = small_pmap / nrotate

            pmap[insti:inedi, instj:inedj] += np.multiply(small_pmap, wI)
            avI[insti:inedi, instj:inedj] += wI
    pmap_img = np.divide(pmap, avI)
    return pmap_img


def main(param_set):

    warnings.filterwarnings('ignore')
    print('====== Phase >>> %s <<< ======' % param_set['mode'])
    NUM_CLASSES = param_set['nclass']  # number of class
    model = Network(NUM_CLASSES)
    print('model: ',param_set['model'])
    model.to(DEVICE)  # send to GPU
    print(model)

    pre_train = False
    if param_set['mode']=='train':
        if pre_train:
            ckpt_dir = param_set['model_loc']
            model.load_state_dict(torch.load(ckpt_dir))
            model_name = ckpt_dir.split('/')[-1]
            print('load pretrained model ', model_name)
        train(param_set, model)
    elif param_set['mode'] == 'test':
        predict(param_set, model)

if __name__ == '__main__':
    #training parameter #
    param_set = dict(numChannels=3,
                      mode='train', # 'train' or 'test'
                      ins=224,      #input resolution
                      ous=224,      #output resolution
                      nclass=2,     #class number
                      batch_size=8,
                      result_dir='./Result/FLLs/',  #save results to this dir
                      folder='2021-01-16-22',
                      imgdir='/data/PV/5fold',      #data path
                      testdir='/data/PV/5fold/batch1',  #test path
                      model_loc='ultimate-500.pth', #checkpoint path
                      epoch=500,
                      epoch_save=10,
                      model='SANet')
    main(param_set)
