import gym
import gym_self_go
import numpy as np
from math import sqrt
from collections import defaultdict


class Node:
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent
        self.children = defaultdict(lambda: None)
        action_num = state.board.size ** 2 + 1
        self.N = np.zeros([action_num], dtype=np.float32)
        self.W = np.zeros([action_num], dtype=np.float32)
        self.player = state.color - 1

    def select(self):
        c = 1
        self.U = c * (sqrt(sum(self.N))) / (1 + self.N)
        self.U = self.U * self.state.legal_actions()
        self.Q = np.divide(self.W, self.N, out=np.zeros_like(self.W), where=self.N != 0)

    def expand(self, child_state, action):
        child = Node(state=child_state, parent=self)
        self.children[action] = child
        self.children[action].action = action

    def history(self):
        player_channel = np.ones((1, self.state.board.size, self.state.board.size)) * self.player
        if self.parent == None:
            initial = np.concatenate([np.zeros((2, 3, 3)).astype(int) for _ in range(7)])
            history = np.concatenate((self.state.board.encode()[:2], initial))[:16]
            self.history = np.concatenate((history, player_channel))
        else:
            self.history = np.concatenate((self.state.board.encode()[:2], self.parent.history))[:16]
            self.history = np.concatenate((self.history, player_channel))
        return self.history


class Net():
    def __init__(self, size=3, channels=256, res_blocks=19):
        """
        Initialization method
        Args:
            size = the size of Go board
            channels = the number of channels in convolution
            res_blocks = the number of residual blocks
        """
        pass

    def forward(self, x):
        """
        Foward method
        Args:
            x = tensor of shape(batch, 17, size, size)
        Returns:
            p_logits, v
        """
        p_logits = np.array([0.1, 0.1, 0.3, 0.2, 0.1, 0.2, 0, 0, 0, 0])
        v = -0.2
        return p_logits, v


env = gym.make('SelfGo3x3-v0')
env.reset()
s = env.state
node = Node(s, None)
net = Net(3, 256, 19).forward


def search(iter, node, t, net):
    action_num = node.state.board.size ** 2 + 1
    pi = np.zeros([action_num], dtype=np.float32)

    for i in range(iter):
        # select
        # node.history = node.history()
        p_logits = net(node.history)[0]
        q = -1
        node.select()
        action = np.argmax(node.U * p_logits + node.Q * q)
        q = q * -1

        while node.children[action] != None:
            node = node.children[action]
            # node.history = node.history()
            p_logits = net(node.history)[0]
            node.select()
            action = np.argmax(node.U * p_logits + node.Q * q)
            q = q * -1

        # expand
        if node.state.board.is_terminal == False:
            new_state = node.state.act(action)
            node.expand(new_state, action)
            node = node.children[action]
            # node.history = node.history()
            v = net(node.history)[1]

        else:
            # backup
            node.N[[action]] += 1
            node.W[[action]] += v
            while node.parent != None:
                node.parent.N[[node.action]] += 1
                node.parent.W[[node.action]] += v
                node = node.parent

        # backup
        while node.parent != None:
            node.parent.N[[node.action]] += 1
            node.parent.W[[node.action]] += v
            node = node.parent

    for i in range(action_num):
        pi[[i]] = (node.N[[i]]) ** (1 / t) / (sum(node.N)) ** (1 / t)

    return pi


search(10, node, 1, net)

##iter 수 많아지면 illegal action 나타나는 문제랑 net input 어떻게 하는 건지