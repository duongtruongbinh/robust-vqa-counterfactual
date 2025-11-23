from __future__ import print_function

import errno
import os
import numpy as np
# from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-7


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def assert_array_eq(real, expected):
    assert (np.abs(real-expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def weights_init(m):
    """custom weights initialization."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' % cname)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def save_model(path, model, epoch, best_eval_score, optimizer=None):
    model_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'best_eval_score': best_eval_score
        }
    if optimizer is not None:
        model_dict['optimizer_state'] = optimizer.state_dict()

    torch.save(model_dict, path)
    
    print("Model saved in ", path)


def set_lr(optimizer, value):
    for p in optimizer.param_groups:
        p['lr'] = value


def decay_lr(optimizer, rate):
    for p in optimizer.param_groups:
        p['lr'] *= rate


def batch_accuracy(logits, labels):
    """
    follow Bilinear Attention Networks https://github.com/jnhwkim/ban-vqa.git
    """
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    
    return scores.sum(1), one_hots


def calculate_loss(answer, pred, method):
    """
    answer = [batch, 3129]
    pred = [batch, 3129]
    """
    max_answers = 2274
    
    if method == 'binary_cross_entropy_with_logits':
        loss = F.binary_cross_entropy_with_logits(pred, answer) * max_answers
    elif method == 'soft_cross_entropy':
        nll = -F.log_softmax(pred, dim=1)
        loss = (nll * answer).sum(dim=1).mean()   # this is worse than binary_cross_entropy_with_logits
    elif method == 'KL_divergence':
        pred = F.softmax(pred, dim=1)
        kl = ((answer / (pred + 1e-12)) + 1e-12).log()
        loss = (kl * answer).sum(1).mean()
    elif method == 'multi_label_soft_margin':
        loss = F.multilabel_soft_margin_loss(pred, answer)
    else:
        print('Error, pls define loss function')
    return loss


def batch_accuracy(logits, labels):
    """
    follow Bilinear Attention Networks https://github.com/jnhwkim/ban-vqa.git
    """
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    
    return scores.sum(1), one_hots

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)