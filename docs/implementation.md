# Implementation Prompt: Multi-Agent Reinforcement Learning-Based Cooperative Jamming Decision-Making (MA-CJD)

**Objective:** Implement the MA-CJD algorithm for cooperative jamming decision-making in a multi-jammer, multi-radar electromagnetic confrontation scenario, based on the provided research paper.

## 1. Problem Description

The core problem is decision-making in a complex electromagnetic game involving multiple active jammers and radar systems. Defenders face cooperative detection from multiple adversarial radar systems. Deploying multiple active jammers for cooperative jamming is an effective countermeasure against networked radar detection. The goal is to develop fast and efficient cooperative jamming decision-making strategies. This involves achieving high-quality and efficient target allocation, jamming mode selection, and power control. Traditional methods often struggle in dynamic, complex electromagnetic environments due to reliance on prior knowledge and static decision-making. Artificial intelligence, particularly reinforcement learning (RL), offers a transformative approach to adversarial decision-making in these dynamic scenarios. Existing RL applications in jamming often focus on discrete action spaces or single-jammer scenarios, facing challenges like the curse of dimensionality and environmental instability in multi-jammer cooperation. The MA-CJD algorithm addresses these challenges.

## 2. Core Framework: Markov Game Model

The cooperative jamming scenario is modeled as a Markov game. A Markov game is defined by a six-tuple: <span class="math-inline">\\langle N ,S , \(A\_i\)\_\{i \\in N\} , p, r, \\gamma \\rangle</span>.

* **Agents (N):** The set of agents <span class="math-inline">N</span> refers to the set of all jammers. Each jammer acts as an agent that autonomously makes decisions.
* **Environment:** The radar model serves as the environment that implements state transitions. Let <span class="math-inline">J</span> be the set of jammers and <span class="math-inline">R</span> be the set of radar systems.
* **State Space (S):** The environment's state is the basis for jammer decision-making. The state space is divided into two parts: inherent attributes and dynamic features.
    * **Inherent Attributes:** These include the radar's detection capability and anti-jamming capability.
        * Detection capability: Represented by the radar's peak transmit power (<span class="math-inline">P\_t</span>), main beam width (<span class="math-inline">\\theta\_m</span>), and search scan period (<span class="math-inline">T\_s</span>).
        * Anti-jamming capability: Represented by the radar type (<span class="math-inline">Type\_r</span>). These attributes reflect the radar's threat level and are crucial for determining jamming targets.
    * **Dynamic Features:** These indicate the relative distance and orientation between the jammer and the radar, significantly impacting jamming effectiveness.
        * Radar's main beam direction (<span class="math-inline">\\theta\_a</span>).
        * Radar's position (<span class="math-inline">pos\_r</span>).
        * Jammer's position (<span class="math-inline">pos\_j</span>).
    * **State Vector:** The state vector <span class="math-inline">s</span> for the cooperative jamming decision-making process is represented as: <span class="math-inline">s \= \[ \(P\_\{it\} , \\theta\_\{im\}, T\_\{is\} , Type\_\{ir\} , \\theta\_\{ia\}, pos\_\{ir\} \)\_\{i \\in R\} , \( pos\_\{jj\} \)\_\{j \\in J\} \]</span>. Radar types are represented using one-hot encoding, and positions as 2D vectors. In a scenario with 4 jammers and 4 radars, the state vector dimension is 48.
* **Action Space (<span class="math-inline">A\_i</span>):** Each jammer agent <span class="math-inline">i</span> must decide its target, jamming type, and power level at each step. The jamming strategy is designed as a parameterized action space consisting of both discrete and continuous actions. The action space schematic is shown in Fig. 4 (conceptually: shows discrete actions with associated continuous parameters).
    * **Action Tuple:** The action <span class="math-inline">u\_i</span> for jammer <span class="math-inline">i</span> is denoted by the tuple <span class="math-inline">\(T\_i, P\_i\)</span>.
    * **Discrete Action (<span class="math-inline">T\_i</span>):** <span class="math-inline">T\_i</span> is a discrete integer value representing the jamming target and type.
        * <span class="math-inline">T\_i \= 0</span>: The jammer performs no jamming action.
        * <span class="math-inline">mod\(T\_i, 2\) \= 0</span>: The jamming type is deception jamming.
        * <span class="math-inline">mod\(T\_i, 2\) \= 1</span>: The type is suppression jamming.
        * <span class="math-inline">\\lfloor\(T\_i \+ 1\)/2\\rfloor</span>: Indicates the target ID of the jamming action.
        * The maximum value of <span class="math-inline">T\_i</span> is <span class="math-inline">K \= 2\|R\|</span>, where <span class="math-inline">\|R\|</span> is the number of radar systems. For 4 radars, <span class="math-inline">K\=8</span>, plus 1 for no action (<span class="math-inline">T\_i\=0</span>), making the discrete action space size 9.
    * **Continuous Action (<span class="math-inline">P\_i</span>):** <span class="math-inline">P\_i \\in \[0, 1\]</span> is a continuous value representing the normalized jamming power level. Let <span class="math-inline">P\_\{j,max\}</span> and <span class="math-inline">P\_\{j,min\}</span> denote the maximum and minimum transmission power of the jammer. The relationship between the actual jamming power <span class="math-inline">P\_j</span> and the jamming power level <span class="math-inline">P\_i</span> is defined as: <span class="math-inline">P\_j \= P\_i \\cdot \(P\_\{j,max\} – P\_\{j,min\}\) \+ P\_\{j,min\}</span>.
* **Reward Function (r):** The objective is to minimize radar detection capabilities with the least jamming resources, aiming to reduce the likelihood of radar successfully tracking real targets. The reward function has three components: tracking penalty (<span class="math-inline">r\_d</span>), resource consumption penalty (<span class="math-inline">r\_p</span>), and jamming success probability reward (<span class="math-inline">r\_j</span>).
    * **Total Reward:** <span class="math-inline">r \= r\_d \+ r\_p \+ r\_j</span>.
    * **Tracking Penalty (<span class="math-inline">r\_d</span>):** Applied when a radar detects and locks onto a defense unit. Its absolute value is larger for radars with higher threat levels. This encourages prioritizing jamming against higher threat radars. <span class="math-inline">r\_d</span> is a manually defined hyperparameter ranging from –1.2 to –0.8.
    * **Resource Consumption Penalty (<span class="math-inline">r\_p</span>):** A linear penalty related to jamming power, incentivizing resource conservation. Let <span class="math-inline">r\_\{p,min\}</span> and <span class="math-inline">r\_\{p,max\}</span> be minimum and maximum penalty values. The calculation is: <span class="math-inline">r\_p \= –r\_\{p,max\} – \(r\_\{p,max\} – r\_\{p,min\}\) \(P\_j – P\_\{j,min\}\) / \(P\_\{j,max\} – P\_\{j,min\}\)</span>. In this study, <span class="math-inline">r\_\{p,min\} \= –0\.1</span> and <span class="math-inline">r\_\{p,max\} \= –0\.01</span>. The magnitude of <span class="math-inline">r\_p</span> is smaller than <span class="math-inline">r\_d</span> to reflect its secondary importance.
    * **Jamming Success Probability Reward (<span class="math-inline">r\_j</span>):** Proposed to help agents identify nuanced state information and avoid sparse rewards. A higher <span class="math-inline">r\_j</span> indicates greater likelihood of successful jamming. <span class="math-inline">r\_j</span> is an expected probability value between 0 and 1.
        * **Deceptive Jamming:** Success probability corresponds to the radar tracking a false target. If any false target in set <span class="math-inline">F</span> (false targets with <span class="math-inline">SNR\_f \> SNR\_\{max\}</span> of real targets) passes detection, the radar is likely to track it. <span class="math-inline">r\_j \= 1 – \\prod\_\{i \\in F\} \(1 – p\_\{rd,i\}\)</span>, where <span class="math-inline">p\_\{rd,i\}</span> is the detection probability of false target <span class="math-inline">i \\in F</span> calculated using Eq. (3).
        * **Suppressive Jamming:** Effect is to reduce target echo SNR, lowering detection probability. <span class="math-inline">r\_j</span> is defined as the reduction in detection probability. <span class="math-inline">r\_j \= p\_\{rd\}\( \\frac\{G\_a P\_s\}\{D P\_\{rjs\} \+ P\_n\} \) – p\_\{rd\}\( \\frac\{G\_a P\_s\}\{P\_n\} \)</span>. (Where <span class="math-inline">P\_\{rjs\}</span> is total suppression jamming power, <span class="math-inline">P\_n</span> is noise power, <span class="math-inline">P\_s</span> is true target echo power, <span class="math-inline">G\_a</span> is pulse compression gain, <span class="math-inline">D</span> is anti-jamming factor. <span class="math-inline">p\_\{rd\}</span> is detection probability from Eq. 3).
        * <span class="math-inline">r\_j</span> dynamically adjusts to approach <span class="math-inline">r\_d</span> magnitude when potential jamming rewards are higher, encouraging interference actions. This design prioritizes reducing detection while balancing power efficiency and promoting proactive jamming.
* **Discount Factor (<span class="math-inline">\\gamma</span>):** Not explicitly defined in the source text, but is a standard component of the Markov Game tuple. Assume a common value like 0.99 or as specified in hyperparameter settings (which are not fully listed, only learning rates, <span class="math-inline">\\epsilon</span>, batch size are mentioned, so standard RL practice value can be inferred or a typical value chosen).

## 3. Mathematical Models

* **Radar Model:** Focuses on targets within the main lobe beam.
    * **Echo Power (<span class="math-inline">P\_s</span>):** True target echo power is calculated based on the radar equation.
        <span class="math-block">P\_s \= \\frac\{P\_t G\_t G\_r \\lambda^2 \\sigma\}\{\(4\\pi\)^3 R^4 L L\_\{Atm\}\}</span>
    * **Detection Probability (<span class="math-inline">p\_\{rd\}</span>):** Calculated based on Signal-to-Noise Ratio (SNR) using Albersheim's approximate formula.
        <span class="math-block">p\_\{rd\}\(SNR\) \= \\frac\{1\}\{1 \+ \\exp\(\-B\)\}</span>
        where <span class="math-inline">B \= \\frac\{10Z – A\}\{1\.7 \+ 0\.12A\}</span>, <span class="math-inline">Z \= SNR \+ \\frac\{5 \\times \\lg M\}\{6\.2 \+ 4\.54/\\sqrt\{M\} \+ 0\.44\}</span>, <span class="math-inline">A \= \\ln\(0\.62 p\_\{rfa\}\)</span>.
    * **SNR Calculation:**
        * For true target: <span class="math-inline">SNR \= \\frac\{\\text\{Echo power\}\}\{\\text\{Noise power \+ Suppression jamming power\}\}</span>
        * For false target: <span class="math-inline">SNR \= \\frac\{\\text\{Deception jamming power\}\}\{\\text\{Noise power\}\}</span>
        * Details of SNR calculation under jamming conditions are in Section 2.2 of the source.
    * **Detection Simulation:** Monte Carlo methods are used. Radar detects if a random number <span class="math-inline">d \\in \[0, 1\]</span> is <span class="math-inline">\\leq p\_\{rd\}</span>.
    * **Tracking:** Radar tracks the detected jammer or false target with the highest detection probability, transitioning to tracking state and adjusting trajectory.
    * **Radar States:** Search, Confirmation, Tracking. State transitions depend on detection success.
* **Jammer Model:** Can employ active suppression jamming or active deception jamming.
    * **Suppression Jamming:** Masks echo signals, making true targets hard to detect.
    * **Deception Jamming:** Generates false target echoes to mislead the radar.
    * **Received Jamming Signal Power (<span class="math-inline">P\_\{rj\}</span>):** Calculated at the radar.
        <span class="math-block">P\_\{rj\} \= \\frac\{P\_j G\_j G\_\{rj\} \\lambda^2 B\_r\}\{\(4\\pi R\_j\)^2 L\_j L\_r L\_\{Atm\} B\_j\}</span>
        (Where <span class="math-inline">P\_j</span> is jammer power, <span class="math-inline">G\_j</span> is jammer antenna gain, <span class="math-inline">G\_\{rj\}</span> is radar receiving gain towards jammer, <span class="math-inline">R\_j</span> is distance, <span class="math-inline">B\_r</span> is radar bandwidth, <span class="math-inline">B\_j</span> is jamming bandwidth, other terms are losses/wavelength).
    * **Received Jamming Power (<span class="math-inline">P\_\{rj\}</span>) vs. <span class="math-inline">P\_j</span>, <span class="math-inline">R\_j</span>, <span class="math-inline">G\_\{rj\}</span>:** <span class="math-inline">P\_\{rj\} \\propto \(P\_j G\_\{rj1\} / R\_j^2\)</span> in main lobe, and <span class="math-inline">P\_\{rj\} \\propto \(P\_j G\_\{rj2\} / R\_j^2\)</span> in sidelobe, assuming stable constants and different main/sidelobe gains (<span class="math-inline">G\_\{rj1\}, G\_\{rj2\}</span>). <span class="math-inline">P\_\{rjd\}</span> is deception power, <span class="math-inline">P\_\{rjs\}</span> is suppression power.
    * **Anti-Jamming Measures:** Radar is equipped with anti-jamming (factor <span class="math-inline">D</span>). <span class="math-inline">P\_\{rj\}</span> after measures <span class="math-inline">\= D \\cdot P\_\{rj\}</span>. <span class="math-inline">D</span> is related to signal direction (sidelobes easier to suppress) and radar system design.
    * **False Target Identification:** Radar identifies false targets based on jamming signal characteristics, probability (<span class="math-inline">p\_\{rij\}</span>) increases with JNR.
        <span class="math-block">p\_\{rij\} \= \\frac\{1\}\{1 \+ \\exp \(w \\cdot \(JNR^\{0\.5\} – JNR\_0\)\)\}</span>
        <span class="math-inline">JNR \= P\_\{rjd\} / P\_n</span>.
    * **SNR under Jamming:**
        * **False Target SNR (<span class="math-inline">SNR\_f</span>):** For signals not successfully identified, <span class="math-inline">SNR\_f \= D P\_\{rjd\} / P\_n</span>.
        * **Real Target SNR (<span class="math-inline">SNR\_a</span>):** Influenced by suppression jamming, <span class="math-inline">SNR\_a \= \\frac\{G\_a P\_s\}\{D P\_\{rjs\} \+ P\_n\}</span>.

## 4. MA-CJD Algorithm Architecture

The MA-CJD algorithm addresses multi-agent game challenges (instability, deceptive rewards) and the parameterized action space challenge. It innovatively applies the QMix multi-agent reinforcement learning algorithm and integrates the MP-DQN network structure. Double DQN mechanism is also employed to improve performance.

* **Centralized Training, Decentralized Execution:** This architecture is adopted.
    * **Training:** A Mixing hyper-network <span class="math-inline">Q\_\{total\}\(s\_t , Q\_1, \. \. \. , Q\_N ;\\omega\_\{mix\}\)</span> is used. It combines individual agent action values (<span class="math-inline">Q\_i</span>) into a global value estimate (<span class="math-inline">Q\_\{total\}</span>). The Mixing network encodes state info into its parameters. Input is agent action values, output is global action value. Non-negative parameters ensure global optimality implies individual optimality. QMix-based credibility allocation helps agents learn their contribution to the team.
    * **Execution:** Each agent independently computes its optimal action using its own DQN (or Q network in MP-DQN).
* **MP-DQN Network Structure:** Used to handle the parameterized action space (discrete actions with continuous parameters). It avoids discretizing continuous parameters, which causes precision loss.
    * **Components:** Actor network and multi-passed Q network.
    * **Functionality:** The Actor network generates the corresponding continuous power level (<span class="math-inline">P\_i</span>) for each possible discrete action (<span class="math-inline">T\_i</span>). Both the state vector and the generated power level (for a specific discrete action) are input into the Q network, which computes the action value estimate for that (discrete action, continuous power) pair. This allows the agent to compute a continuous power level while selecting a discrete action.
    * **Integration:** The MP-DQN structure is integrated into the QMix architecture (conceptually like Fig. 6). Each agent network consists of an Actor network and a Q network.
* **Double DQN Mechanism:** Introduced during training to improve decision-making performance and training speed. It mitigates the overestimation issue caused by maximization and bootstrapping when computing the TD target.
    * **TD Target Calculation:** The computation of the TD target (<span class="math-inline">y\_t</span>) is divided into two steps: selection and evaluation. The original network is used to compute action values and select the action with the maximum value (<span class="math-inline">T\_a^\*</span>). The target network is then used to compute the action value of <span class="math-inline">T\_a^\*</span> at the next time step, yielding the TD target. This differs from original QMix where selection and evaluation both use the target network.

## 5. Network Structure Details (Implementation)

* **Agent Network:** Composed of an Actor network (<span class="math-inline">\\mu\_a</span>) and a Q network (<span class="math-inline">Q\_a</span>).
    * **Actor Network (<span class="math-inline">\\mu\_a\(s\_t ; \\theta\)</span>):** A three-layer fully connected neural network.
        * Input Layer: 128 neurons.
        * Hidden Layer: 128 dimensions, uses ReLU activation function.
        * Output Vector: Mapped to the range <span class="math-inline">\[0, 1\]</span> using a Sigmoid function. It generates a power level <span class="math-inline">P\_i</span> for each possible discrete action <span class="math-inline">T\_i</span> based on the input state vector <span class="math-inline">s\_t</span>.
    * **Q Network (<span class="math-inline">Q\_a\(s\_t , T\_\{at\} , \\mu\_a; \\omega\)</span>):** Takes the state vector <span class="math-inline">s\_t</span> and the corresponding power level <span class="math-inline">P\_i</span> generated by the Actor network for a specific discrete action <span class="math-inline">T\_i</span> as input. It computes the action value estimate for this specific (discrete action, continuous power) pair.
        * Input Layer: 128 neurons, connected via a linear layer activated by ReLU.
        * Hidden Layer: 128-dimensional GRU units. Using GRU units enables the agent to consider past actions and states.
        * Output Layer: A linear layer with 9 neurons, matching the number of discrete actions (<span class="math-inline">K\+1</span>). This layer outputs value estimates for each discrete action, considering the associated continuous power output by the Actor.
* **Mixing Network (<span class="math-inline">Q\_\{mix\}\(s\_t , \\\{Q\_a\\\}\_\{a \\in J\} ; \\omega\_\{mix\}\)</span>):** Aggregates individual agent action-value estimates.
    * **Structure:** A two-layer hyper-network utilizing ELU activation functions.
    * **Layers:**
        * First Layer: 64 dimensions.
        * Output Layer: Single neuron.
    * **Weights & Biases:** Generated by linear layers with 128 dimensions. Weights are constrained to their absolute values to satisfy the Individual-Global-Max condition.
* **Target Networks:** Structures are the same as the original Q network and Mixing network, with parameters denoted as <span class="math-inline">\\omegã</span> and <span class="math-inline">\\omegã\_\{mix\}</span>, and <span class="math-inline">\\thetã</span> for the Actor network.

## 6. Training Process

The training system uses a Client-Server architecture. The server handles the simulation environment (C++, jammer/radar models, interaction logic), and the client handles training (Python, experience pool, agent training, decision-making). Multiple parallel environment instances improve sample collection efficiency.

* **Optimizer:** Adam optimizer is used for all networks.
* **Learning Rates:**
    * Q network and Mixing network (<span class="math-inline">\\alpha</span>): 0.0