# MA-CJD Project Overview

This document provides a high-level overview of the Multi-Agent Cooperative Jamming Decision (MA-CJD) project.

## Goal

The primary goal is to implement the MA-CJD algorithm specified in `docs/implementation.md`, training multiple jammer agents to cooperatively decide which radars to target, with which jamming type (suppression/deception), and at what power level, to minimize radar detection capabilities while managing resource usage, according to the defined reward structure (`r = r_d + r_p + r_j`).

## Core Algorithm: MA-CJD (Based on QMix + MP-DQN)

The project implements the **MA-CJD** algorithm, which utilizes the **QMix** framework combined with an **MP-DQN** agent architecture, as detailed in `docs/implementation.md`. Key features include:

*   **QMix Value Function Factorization:** Learns individual agent utility functions and combines them into a global team value function (Q_tot) using a monotonic mixing network.
*   **Centralized Training, Decentralized Execution (CTDE):** Training uses global state information, but agents execute based on local observations (or the global state under current full observability assumption).
*   **Individual-Global Max (IGM) Property:** Inherited from QMix, ensuring coordination.
*   **MP-DQN for Parameterized Actions:** Handles the hybrid discrete-continuous action space (Jamming Target/Type + Power) using Actor and Q-networks within each agent.
*   **Double DQN:** Used during training target calculation to mitigate overestimation.

## Key Components

1.  **Simulation Environment (`simulation/environment.py`)**:
    *   Simulates the electromagnetic environment containing multiple radars and jammers based on `docs/implementation.md`.
    *   Manages the state (`s`), actions (`T_i`, `P_i`), and calculates the reward `r = r_d + r_p + r_j` using parameters loaded from `config/simulation_config.yaml`.

2.  **Agent Network (`core/networks.py:RNNAgent`)**:
    *   Shared network architecture implementing the verified MP-DQN structure (Actor + RNN base + Q-head).

3.  **Multi-Agent Controller (`core/mac.py:BasicMAC`)**:
    *   Manages the agent network.
    *   Handles observations and computes actions using the verified MP-DQN multi-pass logic.

4.  **Mixing Network (`core/networks.py:QMixer`)**:
    *   Implements the QMix mixing architecture.

5.  **QMix Learner (`core/qmix.py:QMixLearner`)**:
    *   Coordinates training based on QMix.
    *   Handles optimization and target network updates, incorporating the verified MP-DQN Q-values and Double DQN logic.

6.  **Episode Replay Buffer (`utils/replay_buffer.py:EpisodeReplayBuffer`)**:
    *   Stores and samples episode data (including hidden states for RNN).

7.  **Episode Runner (`runners/episode_runner.py:EpisodeRunner`)**:
    *   Orchestrates agent-environment interaction.
    *   Collects episode data (including reward components).

8.  **Main Script (`main.py`)**:
    *   Entry point, loads configuration, initializes components, runs training loop, handles logging.

## Data Flow Summary

`main.py` -> `EpisodeRunner` (interacts with `ElectromagneticEnvironment` & `BasicMAC`) -> `EpisodeReplayBuffer` -> `main.py` -> `QMixLearner` (uses `BasicMAC`, `RNNAgent`, `QMixer`) -> Update Networks. Logs generated throughout. 