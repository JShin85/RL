import gym
import gym_self_go
import numpy as np
from abc import *
import mcts_minibatch as mcts
from Network import Net
from Network import ResidualBlock
import time


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

#     @abstractmethod
#     def save_data(self):
#         pass

net = Net(block=ResidualBlock, blocks=19, size=3)

class SelfPlay(Play):
    def __init__(self, state, parent=None, net=net):
        self.state = state
        self.parent = parent
        self.mcts_node = mcts.Node(state=state, parent=None)
        self.player = state.color - 1
        player_channel = np.ones((1, self.state.board.size, self.state.board.size)) * self.player
        self.mcts_node.channel = self.mcts_node.history() if self.parent == None\
                                    else  np.concatenate((np.concatenate((self.state.board.encode()[:2],
                                                                          self.parent.mcts_node.channel))[:16], player_channel))
        self.net = net

    def select_action(self, t, iter, batch, virtual_loss, net):
        self.pi = self.mcts_node.pi(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=net)
        action_num = self.state.board.size**2 +1
        action = np.random.choice(action_num, 1, p=self.pi)[0]
        return action

    def next_state(self, t, iter, batch, virtual_loss, net):
        child_action = self.select_action(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=net)
        child_state = self.state.act(child_action)
        self.child = SelfPlay(state=child_state, parent=self)

    def play(self, t, iter, batch, virtual_loss, net):
        while self.state.board.is_terminal != True:
            self.next_state(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=net)
            self = self.child

        self.z = self.state.reward()
        self = self.parent
        while self != None:
            self.z = self.child.z
            self = self.parent

env = gym.make('SelfGo3x3-v0')
env.reset()
s = env.state
play = SelfPlay(s, None)
net = Net(block=ResidualBlock, blocks=19, size=3)

start = time.time()
play.play(1, 3, 3, 1, net=net)
end = time.time()
print(end - start)