import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import time 

from lib_stl_core import *
plt.rcParams.update({'font.size': 20})

import utils
from utils import to_np, uniform_tensor, rand_choice_tensor, build_relu_nn, soft_step_hard

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        self.net = build_relu_nn(7, args.nt, args.hiddens, activation_fn=nn.ReLU)
    
    def forward(self, x):
        u = self.net(x)
        if args.no_tanh:
            u = torch.clip(u, -10.0, 10.0)
        else:
            u = torch.tanh(u) * 10.0
        return u

def soft_step(x):
    if args.hard_soft_step:
        return soft_step_hard(args.tanh_ratio * x)
    else:
        return (torch.tanh(500 * x) + 1)/2


def soft_step_ft(x):
    hard = (x>=0).float()
    soft = (torch.tanh(args.tanh_ratio * x) + 1)/2
    return soft

def dynamics(x0, u, include_first=False, finetune=False):
    # input:  x, (n, 6)  # xe, ve, i, t, x_dist, vo, trigger
    # input:  u, (n, T, 1)
    # return: s, (n, T, 6)
    t = u.shape[1]
    x = x0.clone()
    segs = []
    if include_first:
        segs.append(x)
    for ti in range(t):
        new_x = dynamics_per_step(x, u[:, ti:ti+1], finetune=finetune)
        segs.append(new_x)
        x = new_x
    return torch.stack(segs, dim=1)


def dynamics_per_step(x, u, finetune=False):
    new_x = torch.zeros_like(x)
    new_x[:, 0] = x[:, 0] + x[:, 1] * args.dt
    mask = (torch.logical_and(x[:, 0]<args.stop_x, torch.logical_and(x[:, 2]==0, x[:, 4]<0))).float() # stop sign, before the stop region
    if args.test:
        if args.finetune:
            new_x[:, 1] = torch.clip(x[:, 1] + (u[:, 0]) * args.dt, -0.01, 10) * (1-mask) + torch.clip(x[:, 1] + (u[:, 0]) * args.dt, 0.1, 10) * mask
        else:
            new_x[:, 1] = torch.clip(x[:, 1] + (u[:, 0]) * args.dt, -0.01, 10) * (1-mask) + torch.clip(x[:, 1] + (u[:, 0]) * args.dt, 0.1, 10) * mask
    else:
        new_x[:, 1] = torch.clip(x[:, 1] + (u[:, 0]) * args.dt, -0.01, 10)
    new_x[:, 2] = x[:, 2]
    if args.finetune and finetune:
        stop_timer = (x[:, 3] + args.dt * soft_step_ft((x[:,0]-args.stop_x))) * soft_step_ft(-x[:,0])
    else:
        stop_timer = (x[:, 3] + args.dt * soft_step(x[:,0]-args.stop_x)) * soft_step(-x[:,0])
    light_timer = (x[:, 3] + args.dt) % args.phase_t
    new_x[:, 3] = (1-x[:, 2]) * stop_timer + x[:, 2] * light_timer
    new_x[:, 4] = x[:, 4] + (x[:, 5] - x[:, 1]) * args.dt * (x[:, 4]>=0).float()
    new_x[:, 5] = x[:, 5]
    new_x[:, 6] = x[:, 6]
    return new_x


def heading_base_sampling(set0_ve):
    relu = nn.ReLU()
    n = set0_ve.shape[0]
    set00_xo = uniform_tensor(-1, -1, (n//2, 1))
    set00_vo = uniform_tensor(0, 0, (n//2, 1))
    set01_xo = uniform_tensor(args.safe_thres, args.xo_max, (n//2, 1))
    lower = torch.sqrt(relu((args.safe_thres - set01_xo)*args.amax*2 + set0_ve[n//2:]**2))
    set01_vo = uniform_tensor(0, 1, (n//2, 1)) * (args.vmax-lower) + lower

    return torch.cat([set00_xo, set01_xo], dim=0), torch.cat([set00_vo, set01_vo], dim=0)


def initialize_x(N):
    # generate initial points
    # set-0
    # x ~ [-10, 0]
    # v ~ [] make sure v^2/2a < |x|
    # t ~ 0
    #####################################################################
    n = N // 4
    set0_xe = uniform_tensor(-10, args.stop_x, (n, 1))
    bound = torch.clip(torch.sqrt(2*args.amax*(-set0_xe+args.stop_x)), 0, args.vmax)
    set0_ve = uniform_tensor(0, 1, (n, 1)) * bound  # v<\sqrt{2a|x|}
    set0_id = uniform_tensor(0, 0, (n, 1))
    set0_t = uniform_tensor(0, 0, (n, 1))
    if args.heading:
        if args.mock:
            set0_xo = uniform_tensor(-1, -1, (n, 1))
            set0_vo = uniform_tensor(0, 0, (n, 1))
        else:
            set0_xo, set0_vo = heading_base_sampling(set0_ve)
    else:
        set0_xo = uniform_tensor(0, 0, (n, 1))
        set0_vo = uniform_tensor(0, 0, (n, 1))
    if args.triggered:
        if args.mock and not args.no_tri_mock:
            set0_tri = rand_choice_tensor([0, 0], (n, 1))
        else:
            set0_tri = rand_choice_tensor([0, 1], (n, 1))
    else:
        set0_tri = uniform_tensor(0, 0, (n, 1))

    set1_xe = uniform_tensor(args.stop_x, 0, (n, 1))
    set1_ve = uniform_tensor(0, 1.0, (n, 1))
    set1_id = uniform_tensor(0, 0, (n, 1))
    set1_t = uniform_tensor(0, args.stop_t+0.1, (n, 1))
    if args.heading:
        if args.mock:
            set1_xo = uniform_tensor(-1, -1, (n, 1))
            set1_vo = uniform_tensor(0, 0, (n, 1))
        else:
            set1_xo = uniform_tensor(-1, -1, (n, 1))
            set1_vo = uniform_tensor(0, 0, (n, 1))
    else:
        set1_xo = rand_choice_tensor([0, 0], (n, 1))
        set1_vo = uniform_tensor(0, 0, (n, 1))
    set1_tri = uniform_tensor(0, 0, (n, 1))

    n2 = 2*n
    set2_xe = uniform_tensor(-10, args.traffic_x, (n2, 1))
    bound = torch.clip(torch.sqrt(2*args.amax*(-set2_xe + args.traffic_x)), 0, args.vmax)
    set2_ve = uniform_tensor(0, 1, (n2, 1)) * bound  # v<\sqrt{2a|x|}
    set2_id = uniform_tensor(1.0, 1.0, (n2, 1))
    set2_t = uniform_tensor(0, args.phase_t, (n2, 1))
    if args.heading:
        if args.mock:
            set2_xo = uniform_tensor(-1, -1, (n2, 1))
            set2_vo = uniform_tensor(0, 0, (n2, 1))
        else:
            set2_xo, set2_vo = heading_base_sampling(set2_ve)
    else:
        set2_xo = uniform_tensor(0, 0, (n2, 1))
        set2_vo = uniform_tensor(0, 0, (n2, 1))
    set2_tri = uniform_tensor(0, 0, (n2, 1))

    set0 = torch.cat([set0_xe, set0_ve, set0_id, set0_t, set0_xo, set0_vo, set0_tri], dim=-1)
    set1 = torch.cat([set1_xe, set1_ve, set1_id, set1_t, set1_xo, set1_vo, set1_tri], dim=-1)
    set2 = torch.cat([set2_xe, set2_ve, set2_id, set2_t, set2_xo, set2_vo, set2_tri], dim=-1)
    
    x_init = torch.cat([set0, set1, set2], dim=0).float().cuda()

    return x_init


def main():
    utils.setup_exp_and_logger(args, test=args.test)
    eta = utils.EtaEstimator(0, args.epochs, args.print_freq)
    net = Policy(args).cuda()
    if args.net_pretrained_path is not None:
        state_dict = torch.load(utils.find_path(args.net_pretrained_path))
        net.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    cond1 = Eventually(0, args.nt, AP(lambda x: x[..., 3] - args.stop_t, comment="t_stop>=%.1fs"%(args.stop_t)))   
    if args.triggered:
        cond_sig = Imply(AP(lambda x: x[..., 6]-0.5, "Triggered"), Always(0,args.nt, AP(lambda x:args.stop_x*0.5 - x[..., 0], "stop")))
        cond1 = And(cond1, cond_sig)
    cond2 = Always(0, args.nt, 
                Not(And(AP(lambda x: args.phase_red - x[...,3], comment="t=red"),
                        AP(lambda x: -x[..., 0] * (x[..., 0]-args.traffic_x), comment="inside intersection")
                )))
    
    cond3 = Always(0, args.nt, AP(lambda x:x[..., 4]-args.safe_thres,comment="heading>0"))

    if args.heading:
        stl = ListAnd([
            Imply(AP(lambda x: 0.5-x[..., 2], comment="I=stop"), cond1),  # stop signal condition
            Imply(AP(lambda x: x[...,2]-0.5, comment="I=light"), cond2),  # light signal condition
            Imply(AP(lambda x: x[..., 4]+0.5, comment="heading"), cond3)  # heading condition
            ])
    else:
        stl = ListAnd([
            Imply(AP(lambda x: 0.5-x[..., 2], comment="I=stop"), cond1), 
            Imply(AP(lambda x: x[...,2]-0.5, comment="I=light"), cond2)
            ])

    print(stl)
    stl.update_format("word")
    print(stl)

    relu = nn.ReLU()
    x_init = initialize_x(args.num_samples)

    if args.add_val:
        x_init_val = initialize_x(args.num_samples//10)

    n = args.num_samples // 4
    n2 = 2 * n

    for epi in range(args.epochs):
        eta.update()
        if args.update_init_freq > 0 and epi % args.update_init_freq == 0 and epi != 0:
             x_init = initialize_x(args.num_samples)

        x0 = x_init.detach()
        u = net(x0)
        seg = dynamics(x0, u, include_first=args.include_first)
        
        score = stl(seg, args.smoothing_factor)[:, :1]
        acc = (stl(seg, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        acc_avg = torch.mean(acc)
        acc_stop_avg = torch.mean(acc[:n2])
        acc_light_avg = torch.mean(acc[n2:])

        pass_avg = torch.mean((seg[:args.num_samples//2, -1, 0]+seg[:args.num_samples//2, -1, -1] > 0).float())

        stl_loss = torch.mean(relu(0.5-score))
        green_mask = (torch.logical_or(seg[:, :, 2]==0, seg[:, :, 3] > args.phase_red)).float()
        v_loss = torch.mean(acc * torch.relu(5-seg[:,:,1]) * green_mask) * args.v_loss
        s_loss = torch.mean(acc * torch.relu(args.traffic_x-seg[:,:,0])) * args.s_loss
        loss = stl_loss + v_loss + s_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epi % args.print_freq == 0:
            acc_avg_val = torch.tensor(0.0)
            if args.add_val:
                u_val = net(x_init_val.detach())
                seg_val = dynamics(x_init_val, u_val, include_first=args.include_first)
                acc_val = (stl(seg_val, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                acc_avg_val = torch.mean(acc_val)

            print("%s| %03d  loss:%.3f  stl:%.3f  vloss:%.3f sloss:%.3f acc:%.3f (%.3f | %.3f) acc_val:%.3f pass:%.3f dT:%s T:%s ETA:%s" % (
                args.exp_dir_full.split("/")[-1], epi, loss.item(), stl_loss.item(), v_loss.item(), s_loss.item(), acc_avg.item(), acc_stop_avg.item(), acc_light_avg.item(), 
                acc_avg_val.item(),
                pass_avg.item(),
                eta.interval_str(), eta.elapsed_str(), eta.eta_str()
                ))
        
        # Save models
        if epi % args.save_freq == 0:
            torch.save(net.state_dict(), "%s/model_%05d.ckpt"%(args.model_dir, epi))
        
        if epi % args.viz_freq == 0:
            seg_np = to_np(seg)
            acc_np = to_np(acc)
            t_len = seg_np.shape[1]
            N = args.num_samples
            
            linestyle = lambda x: ("-" if x[0, 6] == 0 else "-.") if args.triggered else "-"
            cond_color = lambda x: "green" if x[0]>0 else "red"
            if args.heading:
                plt.figure(figsize=(12, 8))
                nv = 25
                # plot the stop sign curve
                # plot the non-heading case
                # plot the ego-x curve
                # plot the timer curve
                # plot the lead_x curve
                plt.subplot(3, 2, 1)
                for i in range(nv):
                    plt.plot(range(t_len), seg_np[i, :, 0], color=cond_color(acc_np[i]))
                for i in range(N//4, N//4+nv):
                    plt.plot(range(t_len), seg_np[i, :, 0], color=cond_color(acc_np[i]))
                plt.axhline(y=0, xmin=0, xmax=args.nt, color="gray")
                plt.axhline(y=args.stop_x, xmin=0, xmax=args.nt, color="gray")

                plt.subplot(3, 2, 3)
                for i in range(nv):
                    plt.plot(range(t_len), seg_np[i, :, 3], color=cond_color(acc_np[i]))
                for i in range(N//4, N//4+nv):
                    plt.plot(range(t_len), seg_np[i, :, 3], color=cond_color(acc_np[i]))

                plt.subplot(3, 2, 5)
                for i in range(nv):
                    plt.plot(range(t_len), seg_np[i, :, 4], color=cond_color(acc_np[i]))
                for i in range(N//4, N//4+nv):
                    plt.plot(range(t_len), seg_np[i, :, 4], color=cond_color(acc_np[i]))

                # plot the heading case
                # plot the ego-x curve
                # plot the timer curve
                # plot the lead_x curve
                ls = "-."
                plt.subplot(3, 2, 2)
                for i in range(N//8, N//8+nv):
                    plt.plot(range(t_len), seg_np[i, :, 0], linestyle=ls, color=cond_color(acc_np[i]))
                plt.axhline(y=0, xmin=0, xmax=args.nt, color="gray")
                plt.axhline(y=args.stop_x, xmin=0, xmax=args.nt, color="gray")

                plt.subplot(3, 2, 4)
                for i in range(N//8, N//8+nv):
                    plt.plot(range(t_len), seg_np[i, :, 3], linestyle=ls, color=cond_color(acc_np[i]))

                plt.subplot(3, 2, 6)
                for i in range(N//8, N//8+nv):
                    plt.plot(range(t_len), seg_np[i, :, 4], linestyle=ls, color=cond_color(acc_np[i]))

            else:
                plt.figure(figsize=(8, 8))
                # plot the stop sign curve
                # plot the ego-x curve
                plt.subplot(2, 1, 1)
                for i in range(10):
                    plt.plot(range(t_len), seg_np[i, :, 0], color=cond_color(acc_np[i]))
                for j in range(N//2-10, N//2):
                    plt.plot(range(t_len), seg_np[j, :, 0], color=cond_color(acc_np[j]))
                plt.axhline(y=0, xmin=0, xmax=args.nt, color="gray")
                plt.axhline(y=args.stop_x, xmin=0, xmax=args.nt, color="gray")

                # plot the timer curve
                plt.subplot(2, 1, 2)
                for i in range(50):
                    plt.plot(range(t_len), seg_np[i, :, 3], color=cond_color(acc_np[i]))
                for j in range(N//2-50, N//2):
                    plt.plot(range(t_len), seg_np[j, :, 3], color=cond_color(acc_np[j]))                

            plt.savefig("%s/stopsign_iter_%05d.png"%(args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
            plt.close()

            plt.figure(figsize=(8, 8))
            seg_np = seg_np[n2:]
            acc_np = acc_np[n2:]
            plt.subplot(2, 1, 1)
            for i in range(10):
                for j in range(args.nt-1):
                    plt.plot([j, j+1], [seg_np[i, j, 0], seg_np[i, j+1, 0]], 
                             color="red" if seg_np[i, j, 3] <= args.phase_red else "green")
            plt.axhline(y=args.traffic_x, xmin=0, xmax=args.nt, color="gray")
            plt.axhline(y=0, xmin=0, xmax=args.nt, color="gray")
            plt.axis("scaled")
            if args.heading:
                plt.subplot(2, 1, 2)
                for i in range(args.num_samples//2-50, args.num_samples//2):
                    plt.plot(range(seg_np.shape[1]), seg_np[i, :, 4], linestyle=linestyle(seg_np[i]), color="green" if acc_np[i,0]>0 else "red")

            plt.savefig("%s/light_iter_%05d.png"%(args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--exp_name", '-e', type=str, default="traffic")
    add("--gpus", type=str, default="0")
    add("--seed", type=int, default=1007)
    add("--num_samples", type=int, default=50000)
    add("--epochs", type=int, default=50000)
    add("--lr", type=float, default=3e-5)
    add("--nt", type=int, default=25)
    add("--dt", type=float, default=0.1)
    add("--print_freq", type=int, default=100)
    add("--viz_freq", type=int, default=1000)
    add("--save_freq", type=int, default=1000)
    add("--smoothing_factor", type=float, default=100.0)
    add("--sim", action='store_true', default=False)
    add("--net_pretrained_path", '-P', type=str, default=None)
    add("--amax", type=float, default=10)
    add("--stop_x", type=float, default=-1.0)
    add("--v_loss", type=float, default=0.1)
    add("--phase_t", type=float, default=8.0)
    add("--phase_red", type=float, default=4.0)
    add("--traffic_x", type=float, default=-1.0)
    add("--sim_freq", type=int, default=5)
    add("--stop_t", type=float, default=1.0)
    add("--vmax", type=float, default=10.0)
    add("--s_loss", type=float, default=0.1)
    add("--inter_x", type=float, default=0.0)

    add("--test", action='store_true', default=False)
    add("--triggered", action='store_true', default=False)
    add('--heading', action='store_true', default=False)

    add("--safe_thres", type=float, default=1.0)
    add("--xo_max", type=float, default=10.0)

    add('--mock', action='store_true', default=False)
    add('--no_tri_mock', action='store_true', default=False)
    add('--hybrid', action='store_true', default=False)
    add('--bloat_dist', type=float, default=1.0)
    add('--no_viz', action='store_true', default=False)

    # new-tricks
    add("--hiddens", type=int, nargs="+", default=[64, 64, 64])
    add("--no_tanh", action='store_true', default=False)
    add("--hard_soft_step", action='store_true', default=False)
    add("--norm_ap", action='store_true', default=False)
    add("--tanh_ratio", type=float, default=1.0)
    add("--update_init_freq", type=int, default=-1)
    add("--add_val", action="store_true", default=False)
    add("--include_first", action="store_true", default=False)

    add("--mpc", action="store_true", default=False)
    add("--plan", action="store_true", default=False)
    add("--grad", action="store_true", default=False)
    add("--grad_lr", type=float, default=0.10)
    add("--grad_steps", type=int, default=200)
    add("--grad_print_freq", type=int, default=10)
    add("--rl", action="store_true", default=False)
    add("--rl_stl", action="store_true", default=False)
    add("--rl_acc", action="store_true", default=False)
    add("--rl_path", "-R", type=str, default=None)

    add("--pets", action="store_true", default=False)
    add("--mbpo", action="store_true", default=False)

    add("--eval_path", type=str, default="eval_result")

    add("--finetune", action="store_true", default=False)
    add("--backup", action='store_true', default=False)
    add("--cem", action='store_true', default=False)
    args = parser.parse_args()
    args.triggered=True
    args.heading=True

    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.4f seconds"%(t2 - t1))