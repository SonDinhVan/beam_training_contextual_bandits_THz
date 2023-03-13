"""
This module implements the UCB and linUCB algorithms used for beam alignment
and tracking problems.
"""
from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass
class ARM:
    # i-th beamforming of BS
    i: int = None
    # j-th beamforming of UE
    j: int = None
    # number of times this arm is played
    n_play: int = 0


@dataclass
class ArmUCB(ARM):
    # mean reward
    mu: float = 0


class UCB:
    def __init__(self, arms: List[ArmUCB], c: float) -> None:
        # list of all arms
        self.arms = arms
        # number of arms
        self.n_arms = len(arms)
        # upper confidence bound parameter
        self.c = c

    def select_arm(self, t: int) -> int:
        """
        Select an arm to pull.

        Args:
            t (int): the time index.

        Returns:
            int: the index of the pulled arm.
        """
        if t < self.n_arms:
            return t
        else:
            # upper confidence bound for all arms
            ucb = [
                arm.mu + self.c * np.sqrt(np.log(t) / arm.n_play)
                for arm in self.arms
            ]
            return np.argmax(ucb)

    def pull_arm(self, k: int, r: float) -> None:
        """
        Pull a k-th arm and update the attributes of arms.

        Args:
            k (int): The index of the pulled arm.
            r (float): Reward received when k-th arm is pulled.
        """
        # update the estimated reward
        self.arms[k].mu = (self.arms[k].mu * self.arms[k].n_play + r) / (
            self.arms[k].n_play + 1
        )
        # update the number of arms that have been played
        self.arms[k].n_play += 1


@dataclass
class ArmLinUCB(ARM):
    # matrix A, this is used in LinUCB
    A: np.array = None
    # vector b, this is used in LinUCB
    b: np.array = None
    # the estimated vector phi, or coefficient vector
    # this is used in LinUCB
    phi: np.array = None


class LinUCB:
    # Implement the LinUCB algorithm
    # References: Li, Lihong, et al. "A contextual-bandit approach to
    # personalized news article recommendation." Proceedings of the 19th
    # international conference on World wide web. 2010.
    def __init__(
        self, arms: List[ArmLinUCB], delta: float = 0.05, d: int = 2
    ) -> None:
        # list of all arms
        self.arms = arms
        # Initialize phi and b as zero vectors
        # Initialize A as an Identity vector
        for arm in arms:
            arm.phi = np.zeros((d, 1))
            arm.A = np.eye(d)
            arm.b = np.zeros((d, 1))
        # number of arms
        self.n_arms = len(arms)
        # the LinUCB parameter delta, where 1 - delta is the probability of
        # guaranteeing the inequality.
        self.delta = delta
        # the dimension of the contextual vector is d x 1
        self.d = d

    def select_arm(self, t: int, x: np.array) -> int:
        """
        Select an arm to pull.

        Args:
            t (int): the time index.
            x (np.array): the contextual vector in the corresponding timeslot.

        Returns:
            int: the index of the pulled arm.
        """
        alpha = 1 + np.sqrt(0.5 * np.log(2 / self.delta))

        def calc_p(arm: ArmLinUCB) -> float:
            """
            Calculate the p_{t,a} for the ARM a.
            Note that all the values of A, D, c and phi were already updated
            in the attributes of the arm.
            """
            return np.mat(x).T * np.mat(arm.phi) + alpha * np.sqrt(
                np.mat(x).T * np.linalg.inv(arm.A) * np.mat(x)
            )

        return np.argmax([calc_p(arm) for arm in self.arms])

    def pull_arm(self, k: int, r: float, x: np.array) -> None:
        """
        Pull a k-th arm and update the attributes of arms.

        Args:
            k (int): The index of the pulled arm.
            r (float): Reward received when the k-th arm is pulled.
            x (np.array): The contextual vector in the corresponding timeslot.
        """
        # update matrix A and vector b
        self.arms[k].A = self.arms[k].A + np.mat(x) * np.mat(x).T
        self.arms[k].b = self.arms[k].b + r * np.mat(x)
        # update new estimation of phi
        self.arms[k].phi = np.linalg.inv(self.arms[k].A) * np.mat(
            self.arms[k].b
        )
        # update the number of times the arm is pulled
        self.arms[k].n_play += 1


@dataclass
class ArmExp3(ARM):
    # weight for each arm, initialized as 1
    w: float = 1
    # probability of selecting this arm
    p: float = None
    # estimated reward
    x_hat: float = None


class Exp3:
    # Implement the Exp3 algorithm
    # Paper: The non-stochastic multi-armed bandit problem.
    def __init__(self, arms: List[ArmExp3], gamma: float = 0.5) -> None:
        # the list of arms
        self.arms = arms
        # learning rate in the range [0, 1]
        self.gamma = gamma

    def select_arm(self, t: int) -> int:
        """
        Select an arm to pull.

        Args:
            t (int): the time index.

        Returns:
            int: the index of the pulled arm.
        """
        sum_weights = np.sum([arm.w for arm in self.arms])
        # Calculate the probability for each arm
        for arm in self.arms:
            arm.p = (1 - self.gamma) * arm.w / sum_weights + self.gamma / len(
                self.arms
            )
        return np.random.choice(
            range(len(self.arms)), p=[arm.p for arm in self.arms]
        )

    def pull_arm(self, k: int, r: float) -> None:
        """
        Pull a k-th arm and update the attributes of arms.

        Args:
            k (int): The index of the pulled arm.
            r (float): Reward received when k-th arm is pulled.
        """
        self.arms[k].n_play = 1
        for j in range(len(self.arms)):
            # the estimated reward for each arm
            self.arms[j].x_hat = r / self.arms[k].p if j == k else 0
            # update the weight for each arm
            self.arms[j].w *= np.exp(
                self.gamma * self.arms[j].x_hat / len(self.arms)
            )
