import numpy as np
from abc import *
import mcts_minibatch_cuda as mcts
from Network import Net
from Network import ResidualBlock
import torch
import torchvision.transforms as transforms

net_for_self_play = Net(block=ResidualBlock, blocks=19, size=3).cuda()
net_for_train = Net(block=ResidualBlock, blocks=19, size=3).cuda()

dataset = []

class Play(metaclass=ABCMeta):
    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def next_state(self):
        pass

    @abstractmethod
    def play(self):
        pass


class SelfPlay(Play):
    def __init__(self, state, parent=None, net=net_for_self_play):
        self.state = state
        self.parent = parent
        self.net = net
        self.mcts_node = mcts.Node(state=state, parent=None)
        self.player = state.color - 1
        player_channel = np.ones((1, self.state.board.size, self.state.board.size)) * self.player
        self.mcts_node.channel = self.mcts_node.history() if self.parent == None \
            else np.concatenate((np.concatenate((self.state.board.encode()[:2],
                                                 self.parent.mcts_node.channel))[:16], player_channel))

    def select_action(self, t, iter, batch, virtual_loss):
        self.pi = self.mcts_node.pi(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=self.net)
        action_num = self.state.board.size ** 2 + 1
        action = np.random.choice(action_num, 1, p=self.pi)[0]
        return action

    def next_state(self, t, iter, batch, virtual_loss):
        child_action = self.select_action(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss)
        child_state = self.state.act(child_action)
        self.child = SelfPlay(state=child_state, parent=self)

    def play(self, t, iter, batch, virtual_loss):
        play_data = []
        while self.state.board.is_terminal != True:
            self.next_state(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss)
            self = self.child
        self.z = self.state.reward()
        self.pi = self.mcts_node.pi(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=self.net)
        play_data.append((torch.FloatTensor(self.mcts_node.channel),\
                         torch.FloatTensor(self.pi),torch.FloatTensor([self.z])))
        self = self.parent
        while self != None:
            self.z = self.child.z
            play_data.append((torch.FloatTensor(self.mcts_node.channel),\
                         torch.FloatTensor(self.pi),torch.FloatTensor([self.z])))
            self = self.parent
        dataset.append(play_data)


class Evaluator(Play):
    def __init__(self, state, parent=None, net_black=net_for_self_play, net_white=net_for_train):
        self.state = state
        self.parent = parent
        self.net_black = net_black
        self.net_white = net_white
        self.mcts_node = mcts.Node(state=state, parent=None)
        self.player = state.color - 1
        player_channel = np.ones((1, self.state.board.size, self.state.board.size)) * self.player
        self.mcts_node.channel = self.mcts_node.history() if self.parent == None \
            else np.concatenate((np.concatenate((self.state.board.encode()[:2],
                                                 self.parent.mcts_node.channel))[:16], player_channel))

    def select_action(self, t, iter, batch, virtual_loss, net):
        self.pi = self.mcts_node.pi(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=net)
        action_num = self.state.board.size ** 2 + 1
        action = np.random.choice(action_num, 1, p=self.pi)[0]
        return action

    def next_state(self, t, iter, batch, virtual_loss, net):
        child_action = self.select_action(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=net)
        child_state = self.state.act(child_action)
        self.child = Evaluator(state=child_state, parent=self)

    def play(self, t, iter, batch, virtual_loss):
        while self.state.board.is_terminal != True:
            self.next_state(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=self.net_black)
            self = self.child
            if self.state.board.is_terminal != True:
                self.next_state(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=self.net_white)
                self = self.child
            else:
                break
        return self.state.reward()



