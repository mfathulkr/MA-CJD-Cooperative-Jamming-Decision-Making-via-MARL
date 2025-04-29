# MA-CJD: Cooperative Jamming Decision-Making via MARL

## Overview

This project provides a Python implementation of the **Multi-Agent Cooperative Jamming Decision-making (MA-CJD)** method proposed in the following research paper:

*   **Paper:** Cai, B., Li, H., Zhang, N. *et al.* A cooperative jamming decision-making method based on multi-agent reinforcement learning. *Auton. Intell. Syst.* **5**, 3 (2025).
*   **Link:** [https://link.springer.com/article/10.1007/s43684-025-00090-4](https://link.springer.com/article/10.1007/s43684-025-00090-4)

The goal of the MA-CJD method, and this implementation, is to train multiple cooperative jammer agents to effectively counter networked radar detection in complex electromagnetic environments. The agents learn optimal strategies for:
1.  **Target Allocation:** Deciding which radar(s) to jam.
2.  **Jamming Mode Selection:** Choosing between suppression and deception jamming.
3.  **Power Control:** Determining the optimal power level for the chosen jamming action.

The objective is to minimize the detection probability of defended assets by hostile radars while efficiently managing jamming resources (power), as defined by the composite reward function `r = r_d + r_p + r_j`.

## Key Features & Algorithm

This implementation leverages the **QMix** multi-agent reinforcement learning algorithm combined with a **Parameterized Deep Q-Network (MP-DQN)** agent architecture, as detailed in the paper and summarized in `docs/overview.md`.

*   **QMix:** Enables effective coordination between jammer agents by factorizing the global Q-value (`Q_tot`) into individual agent utilities while maintaining the Individual-Global Max (IGM) property. It employs a centralized training and decentralized execution (CTDE) paradigm.
*   **MP-DQN:** Handles the hybrid action space where each agent selects a discrete action (target radar and jamming type) and a continuous parameter (jamming power).
*   **RNN Agents:** Recurrent Neural Networks (GRUs) are used within agents to handle partial observability and temporal dependencies, processing sequences of observations.
*   **Double DQN:** Incorporated into the QMix learner to mitigate Q-value overestimation during training.

## Results

Training runs conducted using this codebase have demonstrated successful learning by the agents. TensorBoard logs show:
*   Consistent improvement in average episode returns.
*   Trends indicating effective jamming (e.g., influence on components related to detection penalty `r_d`).
*   Learning of efficient power management (related to resource penalty `r_p`).
*   Convergence of Q-values and training loss.

These results align with the objectives outlined in the research paper, validating the implementation's ability to train cooperative jamming strategies.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```
3.  **Install dependencies:**
    *Ensure you have a `requirements.txt` file listing necessary packages (like PyTorch, NumPy, PyYAML, TensorBoard). If not, you'll need to create one or install manually.*
    ```bash
    pip install -r requirements.txt
    ```
    *(Key dependencies include: `torch`, `numpy`, `pyyaml`, `tensorboard`)*

## Running the Training

The main training script is `main.py`. You can run it using Python:

```bash
python main.py [OPTIONS]
```

**Common Options:**

*   `--config <name>`: Specifies the main RL configuration file from the `config/` directory (default: `default`). Example: `--config default`
*   `--env-config <name>`: Specifies the simulation environment configuration file from the `config/` directory (default: `simulation_config`). Example: `--env-config simulation_config`
*   `--device <cpu|cuda>`: Specifies the computation device (default: `cuda` if available, otherwise `cpu`). Example: `--device cuda`

**Example Command:**

```bash
python main.py --config default --env-config simulation_config --device cuda
```

Training logs and model checkpoints will be saved in the `logs/` and potentially `models/` directories, organized by test name and run timestamp.

## Viewing Results with TensorBoard

Training progress, performance metrics, and network parameters are logged using TensorBoard.

1.  **Ensure you are in the project's root directory.**
2.  **Run TensorBoard:**
    ```bash
    tensorboard --logdir logs/
    ```
    *This assumes your logs are stored in the default `logs/` directory structure created by `main.py`.*
3.  **Open your web browser** and navigate to the URL provided by TensorBoard (usually `http://localhost:6006/`).

You can monitor metrics such as:
*   Average Episode Return (`Perf/Avg_Return`)
*   Training Loss (`Loss/train_avg`)
*   Epsilon (`Params/Epsilon`)
*   Q-Values (`QValues/eval_qtot_avg`, `QValues/target_qtot_avg`)
*   Reward Components (`Rewards/r_d_avg`, `Rewards/r_p_avg`, `Rewards/r_j_avg`)
*   Action Statistics (`Perf/Avg_Power`, `ActionDist/*`)

## Project Structure

```
.
├── config/             # Configuration files (RL parameters, simulation scenario)
│   ├── default.yaml
│   └── simulation_config.yaml
├── core/               # Core MARL components (MAC, Learner, Networks, Entities)
│   ├── jammer.py
│   ├── mac.py
│   ├── networks.py
│   ├── qmix.py
│   └── radar.py
├── docs/               # Documentation
│   ├── implementation.md
│   └── overview.md
├── logs/               # TensorBoard logs and run outputs
├── runners/            # Episode runners
│   └── episode_runner.py
├── simulation/         # Environment simulation code
│   ├── environment.py
│   └── run_simulation.py # Standalone script for basic env testing
├── utils/              # Utility functions (replay buffer, math, etc.)
│   ├── math_utils.py
│   ├── replay_buffer.py
│   └── state_utils.py
├── main.py             # Main training script entry point
├── README.md           # This file
└── requirements.txt    # Project dependencies (ensure this exists)
```

## Citation

If you use this code or the underlying method in your research, please consider citing the original paper:

Cai, B., Li, H., Zhang, N. *et al.* A cooperative jamming decision-making method based on multi-agent reinforcement learning. *Auton. Intell. Syst.* **5**, 3 (2025). https://doi.org/10.1007/s43684-025-00090-4 