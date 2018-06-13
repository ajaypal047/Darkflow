import os
import time
import numpy as np
import tensorflow as tf
import pickle
from copy import deepcopy
from numpy.random import permutation as perm
from utils.udacity_voc_csv import udacity_voc_csv

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op, self.summary_op] 
        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        self.writer.add_summary(fetched[2], i)

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % 10 #(self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)
       
        if i%2==0:
            print("====================================")
            boxes = val(self)
            eval_list(self.FLAGS.val,boxes)

            #self.val_writer.add_summary(fetched[1], loss_)

    #if ckpt: _save_ckpt(self, *args)

def eval_list(boxes):
    inp_path = self.FLAGS.val
    all_inp_ = os.listdir(inp_path)
    all_inp_ = [i for i in all_inp_ if self.framework.is_inp(i)]

    conf_thresh = 0.25
    nms_thresh = 0.4
    iou_thresh = 0.5
    min_box_scale = 8. / 448

    with open(self.FLAGS.valANN) as fp:
        lines = fp.readlines()

    total = 0.0
    proposals = 0.0
    correct = 0.0
    lineId = 0
    avg_iou = 0.0
    imageName = ''

    for line in lines:
        img_path = line.rstrip()
        imageName = image_path[0]
        break;

    for line in lines:
        img_path = line.rstrip()
        if img_path[0] == imageName:
            truth[image][]



        truths = read_truths_args(lab_path, min_box_scale)
        total = total + truths.shape[0]


    for i in range(len(boxes)):
            proposals = proposals+1

    for i in range(truths.shape[0]):
        box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0]
        best_iou = 0
        for j in range(len(boxes)):
            iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
            best_iou = max(iou, best_iou)
        if best_iou > iou_thresh:
            avg_iou += best_iou
            correct = correct+1

    precision = 1.0*correct/proposals
    recall = 1.0*correct/total
    fscore = 2.0*precision*recall/(precision+recall)
    print("%d IOU: %f, Recal: %f, Precision: %f, Fscore: %f\n" % (lineId-1, avg_iou/correct, recall, precision, fscore))

def read_truths(lab_path):
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size/5, 5) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)

def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = np.min(boxes1[0], boxes2[0])
        Mx = np.max(boxes1[2], boxes2[2])
        my = np.min(boxes1[1], boxes2[1])
        My = np.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = np.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0)
        Mx = np.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0)
        my = np.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0)
        My = np.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea

def predict(self):
    inp_path = self.FLAGS.test
    all_inp_ = os.listdir(inp_path)
    all_inp_ = [i for i in all_inp_ if self.framework.is_inp(i)]
    if not all_inp_:
        msg = 'Failed to find any test files in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inp_))

    boxes = []

    for j in range(len(all_inp_) // batch):
        inp_feed = list(); new_all = list()
        all_inp = all_inp_[j*batch: (j*batch+batch)]
        for inp in all_inp:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.framework.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        all_inp = new_all

        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}
    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        for i, prediction in enumerate(out):
            self.framework.postprocess(prediction,
                    os.path.join(inp_path, all_inp[i]), False, False)

        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))


def val(self):
    inp_path=self.FLAGS.val
    all_inp_ = os.listdir(inp_path)
    all_inp_ = [i for i in all_inp_ if self.framework.is_inp(i)]
    if not all_inp_:
        msg = 'Failed to find any test files in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inp_))

    boxes = []

    for j in range(len(all_inp_) // batch):
        inp_feed = list(); new_all = list()
        all_inp = all_inp_[j*batch: (j*batch+batch)]
        for inp in all_inp:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.framework.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        all_inp = new_all

        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}
    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        for i, prediction in enumerate(out):
            boxes.append(self.framework.postprocess(prediction, os.path.join(inp_path, all_inp[i]),False, True))

        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        return boxes