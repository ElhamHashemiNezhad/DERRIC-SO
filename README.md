# DERRIC-SO 
## Reinforced Fairness-Aware Multi-Agent Self-Organization for 6G Radio Access Network Orchestration

**DERRIC-SO** is a research framework for self-organization orchestration through Multi-Agent Reinforcement Learning (MARL) in heterogeneous 6G network environments which is the extension of DERRIC.

The framework addresses the placement of distributed RAN controllers across heterogeneous infrastructure nodes while jointly optimizing:

- End-to-end controller–user latency  
- User transmission power consumption  
- Packet Delivery Ratio (PDR)

DERRIC models a fully decentralized multi-agent system composed of:

- **Orchestrator agents** that learn optimal placement of distributed RAN controllers.
- **Controller agents** that regulate UE transmission power within their domains.

Learning is performed using **Proximal Policy Optimization (PPO)** implemented with Ray RLlib.

The environment is dynamic, scalable, and mobility-aware, simulating realistic 6G network behavior with heterogeneous compute resources and time-varying user associations.

---

## 🚀 Core Features

- Fully decentralized Multi-Agent Reinforcement Learning (MARL)
- Independent learning at two agent types:
  - Orchestrator agents (controller placement)
  - Controller agents (power control)
- Dynamic number of agents
- Proximal Policy Optimization (PPO) for policy learning
- Mobility-aware UE modeling
- SINR computation based on Free-Space Path Loss
- Transmission power control for UEs
- GEANT-v2 topology with 34 infrastructure nodes
- Implemented using:
  - Ray RLlib (MARL training)
  - NetworkX (topology modeling)

---

## 🧠 System Overview

DERRIC models a distributed 6G RAN system consisting of:

- Infrastructure nodes with heterogeneous capacities
- DEcentralized orchestration
- Distributed RAN controllers
- Base Stations
- User Equipments (UEs)


Orchestrator agents decide controller placement over the network graph, while controller agents manage transmission power of UEs connected to base stations.

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
Learns controller placement policy to reduce:
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

**DERRIC: Decentralized Reinforced RAN Intelligent Controller Orchestration for 6G Networks**  
IEEE Wireless Communications and Networking Conference (WCNC), 2025  
DOI: https://doi.org/10.1109/WCNC61545.2025.10978840
