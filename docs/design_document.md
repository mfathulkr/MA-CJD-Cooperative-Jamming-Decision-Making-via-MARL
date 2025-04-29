<!--- This document reflects the current design and implementation state --->
<!--- of the MA-CJD project as of the last update. --->
<!--- The primary and authoritative specification remains docs/implementation.md. --->
<!--- The codebase implements the design described herein. --->

# MA-CJD Project Design Document

## 1. Introduction

### 1.1 Purpose
This document provides the design specification for the Multi-Agent Cooperative Jamming Decision (MA-CJD) project. It details the algorithms, network architectures, environment simulation, and training procedures guided by the primary specification document: `docs/implementation.md`.

### 1.2 Scope
This project implements the MA-CJD algorithm, training multiple cooperative jammer agents within a simulated electromagnetic environment featuring multiple radars. The goal is to learn optimal jamming strategies (radar target, jamming type, power allocation) according to the specifications in `implementation.md`.

### 1.3 Definitions, Acronyms, and Abbreviations

*   **MA-CJD:** Multi-Agent Cooperative Jamming Decision
*   **MARL:** Multi-Agent Reinforcement Learning
*   **QMix:** A value-based MARL algorithm using monotonic value function factorization.
*   **MP-DQN:** Multi-Pass Deep Q-Network (for parameterized action spaces).
*   **DQN:** Deep Q-Network
*   **CTDE:** Centralized Training, Decentralized Execution
*   **IGM:** Individual-Global Max property
*   **MAC:** Multi-Agent Controller
*   **RNN:** Recurrent Neural Network
*   **GRU:** Gated Recurrent Unit
*   **Env:** Environment
*   **Obs:** Observation
*   **RL:** Reinforcement Learning
*   **TD:** Temporal Difference
*   **SNR:** Signal-to-Noise Ratio
*   **SINR:** Signal-to-Interference-plus-Noise Ratio
*   **Pd:** Probability of Detection
*   **YAML:** YAML Ain't Markup Language
*   **RCS:** Radar Cross Section
*   **P_t:** Radar Transmit Power
*   **P_j:** Jammer Transmit Power
*   **P_s:** Target Echo Power at Radar
*   **P_n:** Noise Power at Radar
*   **P_rj:** Received Jamming Power at Radar
*   **P_rjs:** Received Suppression Jamming Power
*   **P_rjd:** Received Deception Jamming Power
*   **G_a:** Radar Pulse Compression Gain
*   **D:** Radar Anti-Jamming Factor
*   **T_i:** Discrete Jammer Action (Target/Type)
*   **P_i:** Continuous Jammer Action (Normalized Power)

## 2. Problem Formulation

The cooperative jamming scenario is modeled as a Markov game <N, S, {A_i}, p, r, γ>, as defined in `docs/implementation.md` Section 2. The goal is for the set of jammer agents N to learn a joint policy that maximizes the expected discounted return `E[∑ γ^t * r_t]`, where the reward `r` balances radar tracking avoidance (`r_d`), resource efficiency (`r_p`), and jamming success (`r_j`).

## 3. Core Algorithm: MA-CJD

The system implements the MA-CJD algorithm specified in `docs/implementation.md` Section 4. This verified approach combines:

*   **QMix:** For cooperative value function factorization, utilizing its CTDE paradigm and IGM property.
*   **MP-DQN Agent Architecture:** To handle the parameterized action space `A_i = (T_i, P_i)`. Each agent uses an Actor network to generate continuous power parameters `P_i` for all discrete actions `T_i`, and a Q-network to evaluate state-action pairs `(s, T_i, P_i)`.
*   **Double DQN:** The mechanism is employed during TD target calculation within the QMix framework to mitigate Q-value overestimation.

## 4. Network Architectures

Based on `docs/implementation.md` Section 5.

### 4.1. Agent Network (`RNNAgent`)

*   **Purpose:** Implements the verified MP-DQN structure for each jammer agent.
*   **Architecture (MP-DQN Structure):** Composed of an Actor network (`actor_forward`) and a Q-network head (`get_q_value_for_action`) using a shared RNN base (`forward`).
    *   **Actor Network (`actor_forward`)**: 
        *   Input: State `s_t` (or observation `o_t`).
        *   Network: MLP (3-layer: Input -> ReLU -> ReLU -> Sigmoid).
        *   Output: Continuous power level `P_i` for *each* possible discrete action `T_i`.
    *   **RNN Base (`forward`)**: 
        *   Input: State `s_t` (or `o_t`), previous hidden state `h_{t-1}`.
        *   Network: FC -> ReLU -> GRUCell.
        *   Output: Updated hidden state `h_t`.
    *   **Q Network Head (`get_q_value_for_action`)**: 
        *   Input: Updated hidden state `h_t`, a specific discrete action `T_i` (one-hot), and the corresponding continuous power level `P_i` from the Actor.
        *   Network: MLP (Input -> ReLU -> Linear).
        *   Output: Single Q-value estimate `Q(s_t, T_i, P_i)`.
*   **Implementation Status:** The code in `core/networks.py:RNNAgent` implements this structure and has been verified.

### 4.2. Mixing Network (`QMixer`)

*   **Purpose:** Implements the QMix monotonic mixing network.
*   **Architecture:** As specified in `docs/implementation.md` Section 5.
    *   Input: Individual agent Q-values `Q_i(s, T_i, P_i)` (from the verified MP-DQN agent) and the global state `s`.
    *   Hypernetworks: Generate weights and biases for the mixing layers based on `s`. Hypernetworks use ReLU activation.
    *   Mixing Layers: Structure determined by hypernetwork outputs. Uses ELU activation in the first layer. Weights constrained via absolute value (`abs()`) for monotonicity.
    *   Output: Single value `Q_tot`.

## 5. Environment Simulation (`ElectromagneticEnvironment`)

Implemented in `simulation/environment.py`, `core/radar.py`, `core/jammer.py` based on `docs/implementation.md` Sections 2 & 3.

### 5.1. Entities

*   **Radars (`core/radar.py`):** Defined by parameters loaded from `config/simulation_config.yaml`. Includes state (`SEARCH`/`TRACK`) and methods for calculating echo power (`calculate_echo_power`) and detection probability (`detection_probability`).
*   **Jammers (`core/jammer.py`):** Defined by parameters loaded from `config/simulation_config.yaml`. Holds current `power` and calculates received power (`received_power`).
*   **Protected Target:** Defined in `config/simulation_config.yaml` by `position` and `rcs`. Used for calculating true target echo power `P_s`.

### 5.2. State Space (`s`)

*   The global state provided to the RL algorithm.
*   Implemented in `environment.get_state()` exactly as defined in `docs/implementation.md` Section 2:
    `s = [ (P_t, theta_m, T_s, Type_r_onehot, theta_a, pos_r_x, pos_r_y)_i , (pos_j_x, pos_j_y)_j ]`.
*   Dimension depends on `num_radars`, `num_jammers`, and `max_radar_types` (loaded from `config/simulation_config.yaml`).

### 5.3. Observation Space (`o_i`)

*   Information provided to each agent `i`.
*   Currently implemented assuming **Full Observability**: `o_i = s` for all `i`. `environment.get_obs()` returns the global state `s`.

### 5.4. Action Space (`u_i`)

*   Parameterized action space `u_i = (T_i, P_i)` implemented as defined in `docs/implementation.md` Section 2.
*   **Discrete `T_i`:** Integer `0` (idle) to `K = 2 * num_radars`. Encodes Target Radar ID (`floor((T_i+1)/2)-1`) and Jamming Type (`T_i % 2`: 1=Suppression, 0=Deception for `T_i>0`). Size `K+1`.
*   **Continuous `P_i`:** Normalized power level `[0, 1]`.
*   The environment's `step` method expects a list of these `(T_i, P_i)` tuples.

### 5.5. Reward Function (`r`)

*   Global reward signal `r = r_d + r_p + r_j` implemented and verified in `environment.step()` according to `docs/implementation.md` Section 2.
*   **`r_d` (Tracking Penalty):** Implemented. Negative reward applied when a radar enters `TRACK` state, magnitude based on `threat_level` (clamped to the `[rd_min, rd_max]` range specified in `config/simulation_config.yaml`).
*   **`r_p` (Resource Penalty):** Implemented. Negative reward calculated per jammer based on its actual power `P_j` using the specified linear formula and `rp_min`/`rp_max` values from `config/simulation_config.yaml`.
*   **`r_j` (Jamming Success): Implemented.** Calculations based on Pd reduction for suppression and 1-prod(1-Pd_f) for deception are verified.

### 5.6. Physics Simulation

*   Implemented in `environment.step()`, `Radar.calculate_echo_power`, `Radar.detection_probability`, `Jammer.received_power` based on formulas in `docs/implementation.md` Section 3.
*   Calculates `P_s`, `P_rj` (separated into `P_rjs`, `P_rjd`), `SNR_a = Ga*Ps / (D*Prjs + Pn)`, `SNR_f = D*Prjd / Pn`.
*   Uses Albersheim approximation for `Pd = f(SNR)`.
*   Simulates detection events via Monte Carlo.
*   Includes Anti-Jamming factor `D` and Pulse Compression Gain `Ga`.

## 6. Training Procedure

Standard QMix training loop, adapted and verified for MA-CJD (MP-DQN):

1.  **Initialization:** Initialize environment, agent networks (`RNNAgent`), mixing network (`QMixer`), target networks, replay buffer, Adam optimizer.
2.  **Episode Collection (`EpisodeRunner`):** Collect experience using `BasicMAC` and store episodes.
3.  **Training (`QMixLearner`):
    *   Sample batches.
    *   **MP-DQN Q-Value Calculation:**
        *   Use evaluation Actor network to get all `P_i`.
        *   Use evaluation Q-network head (with updated hidden state) to calculate `Q_i(eval)(o_t, T_it, P_it)` for all `T_it`.
        *   Get chosen action `u_t = (T_t, P_t)` Q-value `Q_i(eval)(o_t, T_t, P_t)`.
    *   Calculate `Q_tot(eval)` using evaluation mixer.
    *   **Target Calculation (Double DQN with MP-DQN):**
        *   Use evaluation Actor at `o_{t+1}` to get all `P'_{i,t+1}`.
        *   Use evaluation Q-network head at `o_{t+1}` to calculate all `Q_i(eval)(o_{t+1}, T'_{it+1}, P'_{it+1})`.
        *   Select best discrete actions `T'^*_{t+1}` based on `Q_i(eval)`.
        *   Get corresponding parameters `P'^*_{t+1}` from evaluation Actor.
        *   Calculate target `Q_i(target)(o_{t+1}, T'^*_{t+1}, P'^*_{t+1})` using *target* Q-network head.
        *   Calculate target `Q_tot(target)` using *target* mixer.
    *   Calculate TD target `y_t` and loss.
    *   Perform optimizer step.
    *   Update target networks.
4.  **Logging & Saving:** Log metrics; save checkpoints.

## 7. Configuration

*   **Environment (`config/simulation_config.yaml`):** Defines specific scenario parameters (radars, jammers, protected target) and environment-level parameters (`max_radar_types`, reward ranges `rd_min/max`, `rp_min/max`).
*   **Algorithm (`config/default.yaml`):** Defines RL hyperparameters (learning rate `alpha`, network dimensions, `gamma`, epsilon schedule, buffer/batch sizes, etc.).

## 8. Evaluation

*   Monitor training curves (return, loss, reward components) via TensorBoard.
*   Periodically run evaluation episodes with greedy action selection (`test_mode=True`).
*   Define and track scenario-specific metrics (e.g., radar tracking time, `Pd` reduction).
*   Refer to `docs/testing_guide.md` for detailed testing procedures.

## 9. System Architecture

### 9.1 Folder Structure

```
GPKA-RL-new/
├── config/
├── core/
├── docs/
├── logs/
├── models/
├── runners/
├── simulation/
├── utils/
├── main.py
└── requirements.txt
```

### 9.2 Component Descriptions

*   **`main.py`:** Orchestrates training.
*   **`config/`:** Holds YAML configuration files (`default.yaml`, `simulation_config.yaml`, etc.).
*   **`core/`:** Core algorithm components.
    *   `jammer.py`, `radar.py`: Entity definitions.
    *   `networks.py`: `RNNAgent`, `QMixer` implementations.
    *   `mac.py`: `BasicMAC` (Multi-Agent Controller for MP-DQN).
    *   `qmix.py`: `QMixLearner` (Handles training logic for QMix with MP-DQN).
*   **`docs/`:** Documentation.
*   **`logs/`, `models/`:** Output directories.
*   **`runners/`:** `EpisodeRunner`.
*   **`simulation/`:** `ElectromagneticEnvironment` implementation.
*   **`utils/`:** Helper classes/functions (`EpisodeReplayBuffer`, `EpsilonGreedyActionSelector`, `math_utils`, `state_utils`).

### 9.3 Data Flow Diagram (Simplified)

(The diagram illustrates the interaction between the main components, reflecting the verified MP-DQN data flow: `main.py` orchestrates the `EpisodeRunner`, which uses the `BasicMAC` and `ElectromagneticEnvironment`. Data goes to the `EpisodeReplayBuffer`, sampled by the `QMixLearner` which updates the networks (`RNNAgent`, `QMixer`).)

## 10. Interfaces

(Interfaces between Env, MAC, Learner, Buffer, Runner are established and verified for MP-DQN operation.)

## 11. Algorithms & Techniques

*   **MARL Algorithm:** QMix
*   **Agent Architecture:** MP-DQN (Actor/Q-Network structure)
*   **Policy:** Epsilon-Greedy based on MP-DQN Q-values
*   **Value Function:** Neural Networks (`RNNAgent`, `QMixer`).
*   **Optimization:** Adam.
*   **Stability:** Target Networks, Double DQN. 