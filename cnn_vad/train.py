import os
import math
import argparse
import torch
import utils
import net
import numpy as np
import torch.nn as nn
from model import VADnet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training VAD net")

    parser.add_argument('--gpu', '-g', type=int, help="GPU ID (negative value indicates CPU)", default=-1)
    parser.add_argument('--dataset', '-i', type=str, help="training dataset", choices=[" "], default=" ")
    parser.add_argument('--save_root', '-o', type=str, help="path for saving model", default="./ /")

    parser.add_argument('--epoch', '-e', type=int, help="# of epochs for training", default=1000)
    parser.add_argument('--snapshot', '-s', type=int, help="interval of snapshot", default=100)
    parser.add_argument('--iteration', '-it', type=int, help="number of iterations", default=9)
    parser.add_argument('--lrate', '-lr', type=float, help="learning rate", default=0.0001)

    config = parser.parse_args()

    # Constant values
    N_EPOCH = config.epoch
    N_ITER = config.iteration
    SEGLEN = 128

    # =============== Directories and data ===============
    # Make directories and create log file
    save_path = os.path.join(config.save_root, "VAD_net")
    logprint = utils.set_log(save_path, add=False)[1]

    # Set input directories and data paths
    if config.dataset == "wsj_noisy":
        data_root = "./data/wsj_noisy/"
        label_root = "./data/label/"
    src_folders = sorted(os.listdir(data_root))
    src_folders_label = sorted(os.listdir(label_root))
    data_paths = ["{}/cspec/".format(data_root)]
    file_paths = sorted(os.listdir(data_paths))
    #data_paths = ./data/wsj_noisy/cspec/
    stat_paths = ["{}/train_cspecstat.npy".format(data_root)]
    label_paths = ["{}/label.npy".format(label_root)]
    n_src = len(src_folders)
    # get the num of training data
    n_file = len(file_paths)

    # Define data, lable, and loss function
    src_data = [sorted(os.listdir(p)) for p in data_paths]
    n_src_data = [len(d) for d in src_data]
    src_batch_size = [math.floor(n) // N_ITER for n in n_src_data]
    labels = [np.load(p) for p in label_paths]
    loss_function = nn.CrossEntropyLoss()

    # =============== Set model ==============
    # Set up model and optimizer
    x_tmp = np.load(data_paths[0] + src_data[0][0])
    n_freq = x_tmp.shape[0] - 1
    del x_tmp

    model = VADnet()
    n_para_vad = sum(p.numel() for p in model.parameters())

    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
        cvae.cuda(device)
    else:
        device = torch.device("cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lrate)

    # load pretrained model
    if config.model_path is not None:
        checkpoint = torch.load(config.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = 1 if config.model_path is None else checkpoint['epoch'] + 1

    # set cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # =============== Define functions ===============
    def print_msg(epoch, iter, src, batch_size, loss):
        logprint("epoch {}, iter {}, src {}, batch size={}: KLD={}, rec_loss={}".format(
            epoch, iter, src, batch_size, float(loss.data)))

    def snapshot(epoch):
        print('save the model at {} epoch'.format(epoch))
        torch.save({'epoch': epoch, 'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(save_path, '{}.model'.format(epoch)))

    # =============== Write log file ===============
    logprint("Save folder: {}".format(save_path))
    logprint("\nTraining data:")
    logprint("\tDataset: {}".format(config.dataset))
    logprint("\tNumber of sources: {}".format(n_src))
    logprint("\nNetwork:")
    logprint("\tEncoder architecture:\n\t\t{}".format(model))
    logprint("\tParameter # of VAD_net: {}".format(n_para_vad))
    if config.model_path is not None:
        logprint("\tPretrained model: {}".format(config.model_path))
    logprint("\nTraining conditions:")
    print("\tGPU #: {}".format(config.gpu))
    logprint("\tEpoch #: {}".format(N_EPOCH))
    logprint("\tIteration #: {}".format(N_ITER))
    logprint("\tFile # used in a batch: {}".format(src_batch_size))
    logprint("\tOptimizer: Adam")
    logprint("\tLearning rate: {}".format(config.lrate))
    logprint("\tSnapshot: every {} iteration(s)".format(config.snapshot))

    # =============== Train model ===============
    try:
        for epoch in range(start_epoch, N_EPOCH + 1):
            perms = [np.random.permutation(n) for n in n_src_data]
            perms_data = []

            for i in range(n_src):
                perms_data.append([src_data[i][j] for j in perms[i]])

            for i in range(N_ITER):
                # data pre-processing
                for j in range(n_file):
                    x = utils.dat_load_trunc(file_paths[j], data_paths, SEGLEN, 16)[0]
                    # Normalization
                    x = utils.prenorm(stat_paths[j], x)[0]
                    # Turn X into magnitude spectrograms
                    mag_x = np.linalg.norm(x, axis=1, keepdims=True)
                    # to GPU
                    x = torch.from_numpy(np.asarray(mag_x, dtype="float32")).to(device)
                    l = torch.from_numpy(np.asarray(labels[j], dtype="float32")).to(device)

                    # update trainable parameters
                    optimizer.zero_grad()
                    l_pred = model(x.to(device, torch.float))
                    loss = loss_function(l_pred, l.to(device, torch.long))
                    loss.backward()
                    optimizer.step()

                    print_msg(epoch, i + 1, j + 1, x.size(0), loss)

            if epoch % config.snapshot == 0:
                snapshot(epoch)


    except KeyboardInterrupt:
        logprint("\nKeyboard interrupt, exit.")

    else:
        logprint("Training done!")

    finally:
        print("Output: {}".format(save_path))
        logprint.close()