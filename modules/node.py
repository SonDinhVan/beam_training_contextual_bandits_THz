"""
This class provides the set-up for nodes (i.e. BS, Attacker, UE) in the
network.
"""
# from configs import config
from dataclasses import dataclass
import numpy as np


@dataclass
class Node:
    """
    A data class to contain nodes in the network system.

    Args:
        x (float, optional): x-coordinator position of node. Default = 0.
        y (float, optional): y-coordinator position of node. Default = 0.
    """

    x: float = 0.0
    y: float = 0.0

    def get_distance(self, node: "Node") -> float:
        """
        Get distance from itself to another node

        Args:
            node_2 (node): A node in the network

        Returns:
            float: the distance (in meters)
        """
        return np.sqrt((self.x - node.x) ** 2 + (self.y - node.y) ** 2)

    def get_cos_angle(self, node: "Node") -> float:
        """
        Get the cosin value of the angle of the signal between nodes

        Args:
            node_2 (node): A node in the network

        Returns:
            float: The cosin of angle of the signal respect to the Horizontal
        """
        return np.abs(self.x - node.x) / self.get_distance(node)

    def get_angle(self, node: "Node") -> float:
        """
        Get the angle of the direct signal between two nodes

        Args:
            node (Node): [A node in the network]

        Returns:
            float: [The angle of the signal respect to the Horizontal line]
        """
        return np.arccos(np.abs(self.x - node.x) / self.get_distance(node))


@dataclass
class TRANSCEIVER(Node):
    """
    TRANSCEIVER data class

    Args:
        N (int, optional): [the number of antenna]. Defaults to None.
        P (float, optional): [transmit power in Watt]. Defaults to None.
        M (int, optional): [codebook size]. Defaults to None.
        W (array, optional): [codebook with each column being
            a beamforming vector]. Defaults to None.
    """

    N: int = None
    P: float = None
    M: int = None
    W: np.array = None

    def construct_qbit_codebook(self, q: int) -> None:
        """
        Construct the q-bit codebook

        q: number of bits
        """
        self.W = np.array(
            [
                [
                    1
                    / np.sqrt(self.N)
                    * 1j ** ((4 * (n) * (m) - 2 * self.M) / 2**q)
                    for m in range(self.M)
                ]
                for n in range(self.N)
            ]
        )

    def construct_802_codebook(self, q: int) -> None:
        """
        Construct the 802 codebook

        q: number of bits
        """
        W = np.zeros(shape=(self.N, self.M), dtype="complex")
        set = np.arange(0, 2**q, 1) * 2 * np.pi / 2**q

        for m in range(self.M):
            for n in range(self.N):
                w = 1j ** np.floor(
                    4 * (n) * np.mod(m + self.M / 4, self.M) / self.M
                )
                angle = np.angle(w)
                angle = angle + 2 * np.pi if angle < 0 else angle
                index = np.argmin(np.abs(angle - set))
                W[n, m] = np.exp(1j * set[index])

        self.W = W

    def construct_DFT_codebook(self, q: int) -> None:
        """
        Construct the DFT codebook

        q: number of bits
        """
        W = np.zeros(shape=(self.N, self.M), dtype="complex")
        set = np.arange(0, 2**q, 1) * 2 * np.pi / 2**q

        for m in range(self.M):
            for n in range(self.N):
                w = np.exp(-1j * 2 * np.pi * n * m / self.M)
                angle = np.angle(w)
                angle = angle + 2 * np.pi if angle < 0 else angle
                index = np.argmin(np.abs(angle - set))
                W[n, m] = np.exp(1j * set[index])

        self.W = W

    def steering_vector(self, theta: float = None) -> np.array:
        """
        Calculate the steering vector.

        Args:
            theta (np.float): [The AoD angle]

        Returns:
            np.array: [Steering vector]
        """
        return np.array(
            list(
                map(
                    lambda m: 1
                    / np.sqrt(self.N)
                    * np.exp(1j * np.pi * m * np.sin(theta)),
                    range(self.N),
                )
            )
        ).reshape(self.N, 1)
