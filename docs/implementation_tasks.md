# Implementation Tasks for MA-CJD Project

This document outlines remaining optional enhancements and potential tuning tasks for the MA-CJD project, based on the primary specification `docs/implementation.md`.

**Overall Status:** The core MA-CJD framework based on MP-DQN and QMix has been successfully implemented and verified according to `implementation.md`. This includes the agent network architecture, replay buffer integration, learner logic (including Double DQN), and the primary reward structure (`r=rd+rp+rj`). Focus can now shift towards longer training runs, potential environment fidelity enhancements (Phase 1), or hyperparameter tuning/evaluation (Phase 3).

## Phase 1: Environment & Scenario Fidelity (Optional Enhancements)

This phase focuses on potentially increasing the simulation's realism beyond the core `implementation.md` requirements, if desired.

### Task 1.1: Define Target Scenario & Gather Realistic Parameters

*   **Status:** **Optional** (Current parameters follow `implementation.md`)
*   **Description:** If higher fidelity is desired, define a specific operational scenario and populate configuration files (`config/simulation_config.yaml`) with corresponding realistic physical parameters. Otherwise, use parameters derived from `implementation.md`.
*   **Progress:** Structure for loading parameters exists.
*   **Remaining:** Research and populate values *only if* pursuing higher fidelity beyond the base paper.

### Task 1.2: Implement Realistic Radar Antenna Gain Model

*   **Status:** **Optional Enhancement**
*   **Description:** Replace the placeholder/simple `Radar.get_antenna_gain` with a more complex model accounting for directivity based on angle and beamwidth, if required for the desired level of realism.
*   **Motivation:** Constant gain might be a simplification in `implementation.md`.
*   **Implementation:** Modify `core/radar.py`.

### Task 1.5: Implement Radar Scanning Behavior

*   **Status:** **Optional Enhancement**
*   **Description:** Implement logic to update the radar's beam direction `theta_a` over time, if required for higher fidelity than assumed in `implementation.md`.
*   **Motivation:** Static beam direction might be a simplification.
*   **Implementation:** Add logic in `simulation/environment.py`.

## Phase 3: RL Framework Integration & Tuning

(Phase 2 tasks related to Agent Perception & Motivation are complete)

### Task 3.3: Review/Adjust Network Architectures

*   **Status:** **TODO (Optional Tuning)**
*   **Description:** Evaluate if current network sizes (`RNNAgent`, `QMixer`) specified in `config/default.yaml` (matching `implementation.md`) are optimal. Adjust if necessary during tuning.
*   **Motivation:** Ensure networks have capacity, potentially improve performance.
*   **Implementation:** Modify dimensions in config (`default.yaml`).

### Task 3.4: Systematic Hyperparameter Tuning

*   **Status:** **TODO (Optional Tuning)**
*   **Description:** Optimize RL hyperparameters (`lr`, `gamma`, epsilon annealing, etc.) beyond the values potentially specified or inferred from `implementation.md`.
*   **Motivation:** Improve learning speed and final performance.
*   **Implementation:** Systematic search (e.g., using tools like Optuna or Ray Tune).

### Task 3.5: Implement Rigorous Evaluation Protocol

*   **Status:** **TODO (Optional)**
*   **Description:** Add a dedicated evaluation phase (running episodes with greedy actions, no exploration/training) and potentially more specific performance metrics beyond average return.
*   **Motivation:** Assess true policy performance without exploration noise.
*   **Implementation:** Modify `main.py` loop; define metrics; log separately. 