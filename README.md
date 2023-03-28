This Github repo provides source code for the techniques represented in the paper: "Rapid Beam Training at Terahertz Frequency with Contextual Multi-Armed Bandit Learning", NanoCom2023, Warwick, The UK.

Abstract:"Beam training is a promising technique for achieving terahertz (THz) multiple-input multiple-output (MIMO) communications without relying on explicit channel state information (CSI). Multi-arms bandit (MAB) has been adopted to enable online learning and decision making in beam training, without the need of offline training and data collection. In this paper, we introduce three algorithms to investigate the applications of MAB in beam training at THz frequency, namely UCB, Loc-LinUCB, and Probing-LinUCB. While UCB is built based on the well-known Upper Confidence Bound algorithm, Loc-LinUCB and Probing-LinUCB employ the location of the user equipment (UE) and probing information to enhance decision-making, respectively. The beam training protocol for each algorithm is also illustrated. Their performance are evaluated using data generated by the DeepMIMO framework, which represents abrupt changes and various stringent characteristics in wireless channels encountered in realistic scenarios when UE moves. The results demonstrate that Loc-LinUCB and Probing-LinUCB outperform UCB, which highlights the potential of utilizing contextual MAB in beam training for THz communications."

Installation via "conda env create"

For dependencies, refer to "environment.yaml"
