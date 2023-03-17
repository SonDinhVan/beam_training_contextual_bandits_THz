"""
This module provides calculations related to the opreation of the entire
network system.

Given a network setting, channel and location data, this module calculates
the spectral efficiency and its average values in different schemes.
"""
from configs import config
from modules import node
from modules import bandits

import numpy as np

noise_power = config.NOISE_POWER


class System:
    def __init__(
        self,
        BS: node.TRANSCEIVER,
        UE: node.TRANSCEIVER,
        H: np.array = None,
        loc: np.array = None,
    ) -> None:
        """
        Initialize the system.

        Args:
            BS (node.TRANSCEIVER): The BS.
            UE (node.TRANSCEIVER): The UE.
            H (np.array, optional): The channel data after processing.
                Defaults to None.
            loc (np.array, optional): Location data consisting of x and y
                coordinates. Defaults to None.
        """
        self.BS = BS
        self.UE = UE
        self.H = H
        self.loc = loc

    def calc_SE_at_time_t(self, arm: bandits.ARM, t: int) -> float:
        """
        Calculate the instantaneous spectral efficiency achieved in the
        timeslot t if a specific arm is played.

        Args:
            arm (bandits.ARM): an arm is played.
            t (int): The time slot.

        Returns:
            float: Achievable spectral efficiency (bps/s/Hz).
        """
        # The index of the beamforming used at BS and UE
        i = arm.i
        j = arm.j

        signal_mag = (
            self.BS.P
            * np.abs(
                np.mat(self.UE.W[:, j]).conjugate()
                * np.mat(self.H[t])
                * np.mat(self.BS.W[:, i]).transpose()
            )
            ** 2
        )
        noise = (
            1
            / np.sqrt(2)
            * np.sqrt(noise_power)
            * (
                np.random.normal(size=(self.UE.N, 1))
                + 1j * np.random.normal(size=(self.UE.N, 1))
            )
        )
        noise_mag = np.abs(np.mat(self.UE.W[:, j]).conjugate() * noise) ** 2

        return np.log2(1 + signal_mag.item() / noise_mag.item())

    def calc_average_SE(self, SE: np.array, w: int = 100) -> np.array:
        """
        Calculate the average spectral efficiency using sliding window.
        This is to smoothen the spectral efficiency used for the benchmark.

        Args:
            SE (np.array): The SE performance needs to be averaged.
            w (int, optional): The length of the sliding window.
                Defaults to 100.

        Returns:
            np.array: The average SE.
        """
        return np.convolve(SE, np.ones(w), 'valid') / w

    def calc_optimal_SE(self) -> np.array:
        """
        Calculate the optimal spectral efficiency by scanning all pairs of
        beamforming at BS and UE.

        Returns:
            np.array: Optimal SE obtained at each time slot.
        """
        optimal_SE = np.zeros(len(self.H))
        # Initialize the list of arms
        all_arms = []
        for i in range(self.BS.M):
            for j in range(self.UE.M):
                all_arms.append(
                    bandits.ArmUCB(
                        i=i, j=j
                    )
                )

        for t in range(len(self.H)):
            # scanning all the arms
            for arm in all_arms:
                # calculate the SE if this arm is pulled
                obtained_SE = self.calc_SE_at_time_t(arm=arm, t=t)
                # update the optimal SE if the obtained SE is better
                if obtained_SE > optimal_SE[t]:
                    optimal_SE[t] = obtained_SE
                    # update the optimal arm
                    optimal_arm = arm

            # update the attributes of the optimal arm in time slot t
            optimal_arm.n_play += 1
            optimal_arm.mu = (
                optimal_arm.n_play * optimal_arm.mu + optimal_SE[t])/(
                    optimal_arm.n_play + 1
                )

        return optimal_SE

    def calc_SE_using_UCB(self, delta: float = 0.05) -> np.array:
        """
        Calculate the performance using the UCB algorithm.

        Args:
            delta (float, optional): The uncertainty probability.
                Defaults to 0.05.

        Returns:
            np.array: The SE achieved by using UCB algorithm.
        """
        # Initialize the list of arms
        all_arms = []
        for i in range(self.BS.M):
            for j in range(self.UE.M):
                all_arms.append(
                    bandits.ArmUCB(
                        i=i, j=j
                    )
                )
        SE = np.zeros(len(self.H))
        ucb = bandits.UCB(arms=all_arms, delta=delta)

        for t in range(len(self.H)):
            # selected arm
            k = ucb.select_arm(t=t)
            # calculate the reward received in the time slot t
            SE[t] = self.calc_SE_at_time_t(arm=ucb.arms[k], t=t)
            # pull the arm and update attributes of arms
            ucb.pull_arm(k=k, r=SE[t])
            print('t = {}: pull arm {}, receive reward {}'.format(t, k, SE[t]))

        return SE

    def calc_SE_using_LinUCB(
        self, delta: float = 0.05, d: int = 2
    ) -> np.array:
        """
        Calculate the performance using the LinUCB algorithm.

        Args:

        Returns:
            np.array: The SE achieved by using UCB algorithm.
        """
        # Initialize the list of arms
        all_arms = []
        for i in range(self.BS.M):
            for j in range(self.UE.M):
                all_arms.append(
                    bandits.ArmLinUCB(
                        i=i, j=j
                    )
                )
        SE = np.zeros(len(self.H))
        LinUCB = bandits.LinUCB(arms=all_arms, delta=delta, d=d)

        for t in range(len(self.H)):
            # selected arm
            k = LinUCB.select_arm(x=self.loc[t].reshape(-1, 1))
            # calculate the reward received in the time slot t
            SE[t] = self.calc_SE_at_time_t(arm=LinUCB.arms[k], t=t)
            # pull the arm and update attributes of arms
            LinUCB.pull_arm(k=k, r=SE[t], x=self.loc[t].reshape(-1, 1))
            print('t = {}: pull arm {}, receive reward {}'.format(t, k, SE[t]))

        return SE

    def calc_SE_using_Exp3(
        self, gamma: float = 0.5
    ) -> np.array:
        """
        Calculate the performance using the Exp3 algorithm.

        Args:

        Returns:
            np.array: The SE achieved by using Exp3 algorithm.
        """
        # Initialize the list of arms
        all_arms = []
        for i in range(self.BS.M):
            for j in range(self.UE.M):
                all_arms.append(
                    bandits.ArmExp3(
                        i=i, j=j
                    )
                )
        SE = np.zeros(len(self.H))
        Exp3 = bandits.Exp3(arms=all_arms, gamma=gamma)

        for t in range(len(self.H)):
            # selected arm
            k = Exp3.select_arm()
            # calculate the reward received in the time slot t
            SE[t] = self.calc_SE_at_time_t(arm=Exp3.arms[k], t=t)
            # scale the SE to the range [0, 1]
            scaled_SE = SE[t] / 20
            # pull the arm and update attributes of arms
            Exp3.pull_arm(k=k, r=scaled_SE)
            print('t = {}: pull arm {}, receive reward {}'.format(t, k, SE[t]))

        return SE
