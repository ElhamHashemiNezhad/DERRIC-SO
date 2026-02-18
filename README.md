# DERRIC-SO 
## Reinforced Fairness-Aware Multi-Agent Self-Organization for 6G Radio Access Network Orchestration

**DERRIC-SO** is a research framework for **self-organization orchestration** through Multi-Agent Reinforcement Learning (MARL) in heterogeneous 6G network environments which is the extension of DERRIC.

The framework addresses the organization of decentralized orchestration without requiring centralized coordination across heterogeneous infrastructure nodes while jointly optimizing:

- User throughput  
- Global fairness index  
- Local fairness index

DERRIC-SO models a fully decentralized multi-agent system composed of:

- **Orchestrator agents** that each agent organizes itself autonomously through Duplication, Termination and Relocation or Stay.


Learning is performed using **Proximal Policy Optimization (PPO)** implemented with Ray RLlib.

The environment is dynamic, scalable, and mobility-aware, simulating realistic 6G network behavior with heterogeneous compute resources and time-varying user associations.

---

## 🚀 Core Features

- Fully decentralized Multi-Agent Reinforcement Learning (MARL)
- Centralized Training and Decentralized Execution (CTDE)
- Parallel learning through independet agents
- Jain Fairness Index for user throughput
- Dynamic number of agents
- Proximal Policy Optimization (PPO) for policy learning
- Mobility-aware UE modeling
- SINR computation based on Free-Space Path Loss
- GEANT-v2 topology with 34 infrastructure nodes
- Implemented using:
  - Ray RLlib (MARL training)
  - NetworkX (topology modeling)

---

## 🧠 System Overview

DERRIC-SO models a distributed 6G RAN system consisting of:

- Infrastructure nodes with heterogeneous capacities
- Decentralized orchestration
- Distributed RAN controllers
- Base Stations
- User Equipments (UEs)


Orchestrator agents decide orchestrator placement over the network graph in order to increasing user throughput and fairness.

The system operates under dynamic user mobility and evolving network conditions.

---

## 📂 Repository Structure

```
DERRIC/
│
├── envs/
│   ├── MultiAgentEnvironment.py
│   ├── basestation.py
│   ├── networktopology.py
│   ├── usermobility.py
│   └── voronoidomains.py
│
├── agents/
│   ├── orchestrator.py
│   └── controller.py
│
├── pymobility/
│
├── results/
│
├── main.py
└── README.md
```


## 📦 Module Description

### envs/

**MultiAgentEnvironment.py**  
Custom multi-agent environment defining:
- Agents
- Observation and action spaces
- Reset and step logic

**basestation.py**  
Models 6G gNodeBs including:
- Maximum connected users
- Bandwidth and PRBs
- Maximum transmission power
- Noise parameters
- Coverage radius

**networktopology.py**  
Implements the GEANT-v2 infrastructure topology with:
- 34 compute nodes
- Node capacity modeling
- Link latency properties

**usermobility.py**  
Implements multiple mobility models:
- Random Walk  
- Truncated Lévy Walk  
- Random Direction  
- Random Waypoint  
- Gauss-Markov  

**voronoidomains.py**  
Voronoi-based domain partitioning for orchestrator regions based on spatial placement.

---

### agents/

**orchestrator.py**  
- Learns orchestrator placement policy to increase:
  - User throughput  
  - Jain Fairness Index  
- Learns controller placement policy to reduce:
  - User–controller latency  
  - Controller–orchestrator latency  

**controller.py**  
Learns transmission power allocation policy to improve:
- Packet Delivery Ratio    

---

## 📊 Outputs

After training, the `results/` directory contains:

- User-level performance metrics (CSV files)
- Best policy checkpoints for each agent type

---

## ▶ Running the Framework

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python main.py
```

---

## 📖 Publication

If you use this framework in your research, please cite:

**Reinforced Fairness-Aware Multi-Agent Self-Organization for 6G Radio Access Network Orchestration**  
IEEE 50th Conference on Local Computer Networks (LCN), 2025  
DOI: https://doi.org/10.1109/LCN65610.2025.11146320
