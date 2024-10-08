import os
from os.path import join as ospj
import sys
import time
import shutil
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn as nn


# TODO get the experiment directory
def get_exp_dir(just_local=False):
    os.makedirs("exps_stl", exist_ok=True)
    return "./exps_stl"

def find_path(path):
    return "exps_stl/%s"%(path)

class EtaEstimator():
    def __init__(self, start_iter, end_iter, check_freq, num_workers=1):
        self.start_iter = start_iter
        num_workers = 1 if num_workers is None else num_workers
        self.end_iter = end_iter//num_workers
        self.check_freq = check_freq
        self.curr_iter = start_iter
        self.start_timer = None
        self.interval = 0
        self.eta_t = 0
        self.num_workers = num_workers

    def update(self):
        if self.start_timer is None:
            self.start_timer = time.time()
        self.curr_iter += 1
        if self.curr_iter % (max(1,self.check_freq//self.num_workers)) == 0:
            self.interval = self.elapsed() / (self.curr_iter - self.start_iter)        
            self.eta_t = self.interval * (self.end_iter - self.curr_iter)
    
    def elapsed(self):
        return time.time() - self.start_timer
    
    def eta(self):
        return self.eta_t
    
    def elapsed_str(self):
        return time_format(self.elapsed())
    
    def interval_str(self):
        return time_format(self.interval)

    def eta_str(self):
        return time_format(self.eta_t)

def time_format(secs):
    _s = secs % 60 
    _m = secs % 3600 // 60
    _h = secs % 86400 // 3600
    _d = secs // 86400
    if _d != 0:
        return "%02dD%02dh%02dm%02ds"%(_d, _h, _m, _s)
    else:
        if _h != 0:
            return "%02dH%02dm%02ds"%(_h, _m, _s)
        else:
            if _m != 0:
                return "%02dm%02ds"%(_m, _s)
            else:
                return "%05.2fs"%(_s)


# TODO create the exp directory
def setup_exp_and_logger(args, set_gpus=True, just_local=False, test=False):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sys.stdout = logger = Logger()
    EXP_ROOT_DIR = get_exp_dir(True)
    args.exp_dir_full = os.path.join(EXP_ROOT_DIR, "g%s_%s" % (logger._timestr, args.exp_name))
    args.viz_dir = os.path.join(args.exp_dir_full, "viz")
    args.src_dir = os.path.join(args.exp_dir_full, "src")
    args.model_dir = os.path.join(args.exp_dir_full, "models")
    os.makedirs(args.viz_dir, exist_ok=True)
    os.makedirs(args.src_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    for fname in os.listdir('./'):
        if fname.endswith('.py'):
            shutil.copy(fname, os.path.join(args.src_dir, fname))

    logger.create_log(args.exp_dir_full)
    with open(ospj(args.exp_dir_full, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(sys.argv))
    np.savez(os.path.join(args.exp_dir_full, 'args'), args=args)

    if set_gpus and hasattr(args, "gpus") and args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    return args

class Logger(object):
    def __init__(self):
        self._terminal = sys.stdout
        self._timestr = datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")
        self.log = None

    def create_log(self, log_path):
        self.log = open(log_path + "/log-%s.txt" % self._timestr, "a", 1)

    def write(self, message):
        self._terminal.write(message)
        if self.log is not None:
            self.log.write(message)

    def flush(self):
        pass

def to_np(x):
    return x.detach().cpu().numpy()

def to_torch(x):
    return torch.from_numpy(x).float().cuda()

def uniform_tensor(amin, amax, size):
    return torch.rand(size) * (amax - amin) + amin

def rand_choice_tensor(choices, size):
    return torch.from_numpy(np.random.choice(choices, size)).float()

def build_relu_nn(input_dim, output_dim, hiddens, activation_fn, last_fn=None):
    n_neurons = [input_dim] + hiddens + [output_dim]
    layers = []
    for i in range(len(n_neurons)-1):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
        layers.append(activation_fn())
    if last_fn is not None:
        layers[-1] = last_fn()
    else:
        del layers[-1]
    return nn.Sequential(*layers)

def build_relu_nn1(input_output_dim, hiddens, activation_fn, last_fn=None):
    return build_relu_nn(input_output_dim[0], input_output_dim[1], hiddens, activation_fn, last_fn=last_fn)

def soft_step_hard(x):
    hard = (x>=0).float()
    soft = (torch.tanh(500 * x) + 1)/2
    return soft + (hard - soft).detach()

def xxyy_2_Ab(x_input):
    xmin, xmax, ymin, ymax = x_input
    A = np.array([
            [-1, 1, 0, 0],
            [0, 0, -1, 1]
        ]).T
    b = np.array([-xmin, xmax, -ymin, ymax])
    return A, b