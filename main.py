import gym
import gym_self_go
import argparse
import time
import Play_cuda
from Play_cuda import *
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable


def main():
    parser = argparse.ArgumentParser(description='AlphaGo-Zero Training')
    parser.add_argument('--size', default=3, type=int, choices=[3, 5, 9],
                        help='size of Go (default: 5x5)')
    parser.add_argument('--tau', '--t', default=1, type=int,
                        help='initial infinitesimal temperature (default: 1)')
    parser.add_argument('--search', default=30, type=int,
                        help='number of mcts minibatch search times (default: 10)')
    parser.add_argument('--mb', default=5, type=int,
                        help='minibatch size of mcts (default: 8)')
    parser.add_argument('--vl', default=1, type=int,
                        help='virtual loss(to ensure each thread evaluates different nodes) (default: 1)')
    parser.add_argument('--initial-play', '--iplay', default=30, type=int,
                        help='number of self play times at initial stage to get play datasets (default: 2000)')
    parser.add_argument('--eval', default=30, type=int,
                        help='number of play times to evaluate neural network (default: 100)')
    parser.add_argument('--epoch', default=2, type=int,
                        help='number of total epochs to run (default: 5)')
    parser.add_argument('--tb', default=10, type=int,
                        help='minibatch size of neural network training (default: 20)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', '--m', default=0.9, type=float,
                        help='initial momentum ')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--play', default=20, type=int,
                        help='number of self play times to get more datasets (default: 100)')
    parser.add_argument('--iter', default=5, type=int,
                        help='iterate times of self play and evaluate play (default: 100)')

    args = parser.parse_args()

    if args.size == 3:
        env = gym.make('SelfGo3x3-v0')
    elif args.size == 5:
        env = gym.make('SelfGo5x5-v0')
    else:
        env = gym.make('SelfGo9x9-v0')

    # make play_dataset
    for i in range(args.initial_play):
        env.reset()
        s_self_play = env.state
        play = SelfPlay(s_self_play)
        play.play(t=args.tau, iter=args.search, batch=args.mb, virtual_loss=args.vl)
    cat_dataset = torch.utils.data.ConcatDataset(Play_cuda.dataset)
    data_loader = torch.utils.data.DataLoader(cat_dataset, batch_size=args.tb)

    # loss and optimizer
    criterion_entropy = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    optimizer = torch.optim.SGD(net_for_train.parameters(), lr=args.lr, weight_decay=args.wd)

    # train - evaluate - self play and get data => how many times?
    for iter in range(args.iter):
        for epoch in range(args.epoch):
            for (i, (s, pi, z)) in enumerate(data_loader):
                s = Variable(s).cuda()
                pi = Variable(pi).cuda()
                z = Variable(z).cuda()
                optimizer.zero_grad()
                p_logits, v = net_for_train(s)
                loss = criterion_entropy(p_logits, pi) + criterion_mse(v, z)
                loss.backward()
                optimizer.step()
                print('[%d/%d]. Epoch [%d/%d] Iter [%d] Loss : %.4f' % (
                iter+1, args.iter, epoch + 1, args.epoch, i + 1, loss.data[0]))

        # evaluator
        mean_rewards = 0
        for i in range(args.eval):
            env.reset()
            s_evaluator = env.state
            evaluator = Evaluator(s_evaluator)
            mean_rewards += evaluator.play(t=args.tau, iter=args.search, batch=args.mb, virtual_loss=args.vl)

        print('[%d/%d]. mean rewards : %d' % (iter+1, args.iter, mean_rewards))
        if mean_rewards >= 0.1*(args.eval):
            net_for_self_play = net_for_train

        # get more dataset and drop former data
        for i in range(args.play):
            env.reset()
            s_self_play = env.state
            play = SelfPlay(s_self_play)
            play.play(t=args.tau, iter=args.search, batch=args.mb, virtual_loss=args.vl)
        Play_cuda.dataset = Play_cuda.dataset[args.play:]
        cat_dataset = torch.utils.data.ConcatDataset(Play_cuda.dataset)
        data_loader = torch.utils.data.DataLoader(cat_dataset, batch_size=args.tb)

        # play with pachi and get winning rate

start = time.time()
if __name__ == '__main__':
    main()
end = time.time()
print(end - start)

