import csv
import os
from collections import defaultdict
import networkx as nx
from gymnasium.spaces import Discrete, Box, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from agents.controlleragent import ControllerAgent
from agents.orchestratoragent import OrchestratorAgent
from .networktopology import create_geant_topology
from .usermobility import UserMobilityManager
from .voronoidomains import VoronoiDomainManager
from .basestation import BaseStation


class MultiAgentEnvironment(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ---- Core config ----
        self.num_hosts          = int(config.get("num_hosts"))
        self.num_orchestrators  = int(config.get("num_orchestrators"))
        self.num_controllers    = int(config.get("num_controllers"))
        self.num_users          = int(config.get("num_users"))
        self.num_base_stations  = int(config.get("num_base_stations"))
        self.seed               = config.get("seed")
        self.output_dir         = config.get("output_dir", "results")


        self.control_plane_latency = 20  # ms
        self.user_plane_latency = 1  # ms

        # Time/episode
        self.time_step_duration = float(config.get("time_step_duration"))
        self.episode_length     = int(config.get("episode_length"))
        self.current_step       = 0

        # ---- Assignments & managers ----
        self.user_power_allocation = {}
        self.controller_assignments = {}
        self.user_bs_assignments = {}
        self.bs_user_assignments = {}
        self.base_station_assignments = {}
        self.controller_to_bs_map = {}
        self.user_latency_map = {}

        # RNG
        self.rng = np.random.default_rng(self.seed)
        # ---- Mobility & positions ----
        self.mobility_manager = UserMobilityManager(self.num_users, seed=self.seed)
        self.mobility_manager.initialize_model()
        self.user_positions = self.mobility_manager.get_positions()
        self.dimensions = self.mobility_manager.dimensions

        self.host_positions = np.column_stack((
            np.linspace(0, self.dimensions[0], self.num_hosts, dtype=np.float64),
            np.linspace(0, self.dimensions[1], self.num_hosts, dtype=np.float64),
        ))

        # Orchestrator “high-capacity” host indices (validate against num_hosts)
        default_hi_cap = [0, 5, 6, 9, 10, 11, 25]
        self.orchestrator_host_indices = {
            f"orch_{i+1}": idx for i, idx in enumerate(default_hi_cap) if idx < self.num_hosts
        }
        if len(self.orchestrator_host_indices) < self.num_orchestrators:
            raise ValueError(
                f"Not enough high-capacity hosts for {self.num_orchestrators} orchestrators "
                f"(available: {len(self.orchestrator_host_indices)})."
            )
        self.orchestrator_hosts = self.host_positions[list(self.orchestrator_host_indices.values())].astype(np.float64)

        # ---- Agent IDs (use deterministic lists, not sets) ----
        self.orchestrator_agents = [f"orch_{i+1}" for i in range(self.num_orchestrators)]
        self.orchcont_agents = [f"orchcont_{i+1}" for i in range(self.num_orchestrators)]
        self.controller_agents   = [f"ctrl_{i+1}"     for i in range(self.num_controllers)]

        # After you define each agent group (lists), convert them to sets:
        self.orchestrator_agents = set(self.orchestrator_agents)
        self.orchcont_agents = set(self.orchcont_agents)
        self.controller_agents = set(self.controller_agents)

        self._agent_ids = set().union(
            self.orchestrator_agents,
            self.orchcont_agents,
            self.controller_agents,
        )

        # Orchestrator positions: sample and map to actual agent IDs
        selected = self.rng.choice(len(self.orchestrator_hosts), len(self.orchestrator_agents), replace=False)
        self.orchestrator_positions = {}
        for i, orch_id in enumerate(sorted(self.orchestrator_agents)):
            self.orchestrator_positions[orch_id] = np.array(self.orchestrator_hosts[selected[i]], dtype=np.float64)

        # Controller host positions 
        reserved = set(self.orchestrator_host_indices.values())
        available_ctrl_host_indices = [i for i in range(self.num_hosts) if i not in reserved]
        if not available_ctrl_host_indices:
            raise ValueError("No available controller host positions (all hosts reserved by orchestrators).")
        self.controller_hosts = self.host_positions[available_ctrl_host_indices].astype(np.float64)

        # Deterministic controller -> host mapping
        self.controller_host_indices = {}
        for idx, ctrl_id in enumerate(self.controller_agents):
            host_index = available_ctrl_host_indices[idx % len(available_ctrl_host_indices)]
            self.controller_host_indices[ctrl_id] = host_index

        self.controller_positions = {
            ctrl_id: self.host_positions[host_index].astype(np.float64)
            for ctrl_id, host_index in self.controller_host_indices.items()
        }
        self.orchestrator_controller_assignments()

        # Base station positions sampled from controller-available hosts
        if self.num_base_stations > len(available_ctrl_host_indices):
            raise ValueError("num_base_stations exceeds available controller host indices.")
        bs_indices = self.rng.choice(available_ctrl_host_indices, self.num_base_stations, replace=False)
        self.base_station_positions = self.host_positions[bs_indices]

        # ---- Base stations ----
        self.base_stations = {
            bs_id: BaseStation(self, station_id=bs_id, position=pos)
            for bs_id, pos in enumerate(self.base_station_positions)
        }

        self.domain_manager = VoronoiDomainManager(self)

        # Initial assignments
        self._update_user_bs_assignments()
        self._update_base_station_assignments()
        self._initialize_power_allocation()

        # ---- Action/Observation spaces ----
        ctrl_candidates = max(0, self.num_hosts - len(self.orchestrator_host_indices))

        # Orchestrator actions: move/terminate/duplicate 
        self.action_orchestrator_spaces = {
            agent: Discrete(2 * len(self.orchestrator_host_indices) + 1)
            for agent in self.orchestrator_agents
        }

        # OrchCont actions: binary decisions for controller locations
        self.action_orchcont_spaces = {
            agent: MultiDiscrete([2] * ctrl_candidates) for agent in self.orchcont_agents
        }

        # Controller power allocation
        self.action_controller_spaces = {
            agent: Box(low=0.0, high=1.0, shape=(self.num_users,), dtype=np.float32)
            for agent in self.controller_agents
        }

        # Observations 
        self.observation_orchestrator_spaces = {
            agent: Box(low=-1, high=1, shape=(28,), dtype=np.float32)
            for agent in self.orchestrator_agents
        }
       
        self.observation_orchcont_spaces = {
            agent: Box(low=-1e9, high=1e9, shape=(84,), dtype=np.float32)
            for agent in self.orchcont_agents
        }

        bs_map = self._get_controller_to_bs_mapping()

        self.MAX_USERS_PER_BS = 20
        self.MAX_BS_PER_CTRL = max(len(bs_list) for bs_list in bs_map.values())
        self.MAX_USERS_PER_CTRL = self.MAX_USERS_PER_BS * self.MAX_BS_PER_CTRL

        self.OBS_DIM_CONTROLLER = 4 * self.MAX_USERS_PER_CTRL
        self.observation_controller_spaces = {
            agent: Box(low=-1, high=1, shape=(self.OBS_DIM_CONTROLLER,), dtype=np.float32)
            for agent in self.controller_agents
        }

        # ---- Topology ----
        self.G, self.link_properties = create_geant_topology(self.num_hosts)

        # ---- Agent instances ----
        self.orchcont_policy = config.get("orchcont_policy")
        self.controller_policy = config.get("controller_policy")

        self.orchestrator_agent_instances = {
            agent_id: OrchestratorAgent(self, seed=self.seed)
            for agent_id in self.orchestrator_agents
        }

        self.orchcont_agent_instances = {
            agent_id: OrchestratorAgent(self, seed=self.seed)
            for agent_id in self.orchcont_agents
        }

        self.controller_agent_instances = {
            ctrl_id: ControllerAgent(self, controller_id=ctrl_id, controller_policy=self.controller_policy,)
            for ctrl_id in self.controller_agents
        }

        # ---------- Utility ----------

    @staticmethod
    def calculate_distance(pos1, pos2):
        return float(np.linalg.norm(pos1 - pos2))

    # ---------- Update Domains ----------
    def update_domains(self):
        """Update the Voronoi domains"""
        active_orchestrators = {
            oid: pos for oid, pos in self.orchestrator_positions.items()
            if not np.array_equal(pos, [-1, -1])
        }
        active_controllers = {cid: self.controller_positions[cid] for cid in self.controller_positions}

        self.domain_manager.update_domains(active_orchestrators)
        self.domain_manager.update_domains(active_controllers)

    # ---------- Add / Remove Controllers ----------
    def add_controller(self, ctrl_id, position, owner_id):
        self.controller_agents.add(ctrl_id)
        self.controller_positions[ctrl_id] = position
        self.controller_assignments[ctrl_id] = owner_id
        self.controller_agent_instances[ctrl_id] = ControllerAgent(self, controller_id=ctrl_id,
                                                                   controller_policy=self.controller_policy)
        self._agent_ids.add(ctrl_id)
        self.num_controllers = len(self.controller_agents)

    def remove_controller(self, ctrl_id):
        self.controller_agents.discard(ctrl_id)
        self.controller_positions.pop(ctrl_id, None)
        self.controller_assignments.pop(ctrl_id, None)
        self.controller_agent_instances.pop(ctrl_id, None)
        self.num_controllers = len(self.controller_agents)

    # ---------- Assignments / domains ----------
    def orchestrator_controller_assignments(self):
        orch_ids = [
            oid for oid, pos in self.orchestrator_positions.items()
            if not np.array_equal(pos, [-1, -1])
        ]
        if not orch_ids:
            return

        # ---------------------------------------------
        # 1) Start with a CLEAN slate of assignments
        # ---------------------------------------------
        rl_assignments = dict(self.controller_assignments)  # copy previous step
        self.controller_assignments = {}  # reset everything

        # Keep only RL assignments for PRESSENT orchestrators
        for ctrl_id, orch_id in rl_assignments.items():
            if orch_id in orch_ids:
                self.controller_assignments[ctrl_id] = orch_id

        # ---------------------------------------------
        # 2) Assign unassigned controllers by distance
        # ---------------------------------------------
        for ctrl_id, ctrl_pos in self.controller_positions.items():
            if ctrl_id in self.controller_assignments:
                continue  # RL action already assigned it this step

            # closest alive orchestrator
            dists = {
                oid: float(np.linalg.norm(ctrl_pos - self.orchestrator_positions[oid]))
                for oid in orch_ids
            }
            nearest = min(dists, key=dists.get)
            self.controller_assignments[ctrl_id] = nearest

        # ---------------------------------------------
        # 3) Ensure every orchestrator owns at least one controller
        # ---------------------------------------------
        controllers_by_orch = defaultdict(list)
        for ctrl, orch in self.controller_assignments.items():
            controllers_by_orch[orch].append(ctrl)

        for orch_id in orch_ids:
            if len(controllers_by_orch[orch_id]) == 0:
                # find donor with most controllers
                donor_id = max(controllers_by_orch, key=lambda k: len(controllers_by_orch[k]))
                moved = controllers_by_orch[donor_id].pop()
                self.controller_assignments[moved] = orch_id
                controllers_by_orch[orch_id].append(moved)

    def update_user_latency_map(self):
        """Recompute network latency from each user to its serving BS."""
        self.user_latency_map = {}

        G, _ = create_geant_topology(self.num_hosts)
        user_bs_map = dict(self.user_bs_assignments)
        for user, bs in user_bs_map.items():
            try:
                path = nx.shortest_path(
                    G,
                    source=user,
                    target=bs,
                    weight="latency_ms"
                )

                latency = sum(
                    G[path[i]][path[i + 1]]["latency_ms"]
                    for i in range(len(path) - 1)
                )

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                latency = float("inf")

            self.user_latency_map[user] = latency

    def _update_user_bs_assignments(self):
        """Assign users to base stations with basic fairness-aware heuristic."""
        old = dict(getattr(self, "user_bs_assignments", {}))
        self.user_bs_assignments = {}

        # Clear connected_users dict and reset RB accounting
        for bs in self.base_stations.values():
            bs.connected_users.clear()  # now a dict: {}
            bs.allocated_resource_blocks = 0  # safe to reset since we cleared users

        priority = []
        for u_id, u_pos in enumerate(self.user_positions):
            if u_id in old:
                old_bs_id = old[u_id]
                old_pos = self.base_stations[old_bs_id].position
                d = np.linalg.norm(u_pos - old_pos)
                priority.append((u_id, d, True))
            else:
                priority.append((u_id, 0.0, False))
        priority.sort(key=lambda x: (not x[2], x[1]))

        for user_id, _, had_prev in priority:
            u_pos = self.user_positions[user_id]
            scored = []
            for bs_id, bs in self.base_stations.items():
                dist = np.linalg.norm(u_pos - bs.position)
                if dist > bs.coverage_radius:
                    continue
                sinr = bs.calculate_sinr_dB(u_pos, user_id=user_id)
                current_users = len(bs.connected_users)
                if current_users >= bs.max_users:
                    continue
                load_factor = current_users / bs.max_users
                score = -sinr + 2.0 * load_factor
                scored.append((score, bs_id, dist))
            scored.sort(reverse=True)

            for _, bs_id, _ in scored:
                bs = self.base_stations[bs_id]
                velocity = self.mobility_manager.velocities[user_id]  # adjust to your attribute name

                # connect_user handles: RB allocation, SINR, capacity, latency, CSV logging
                success = bs.connect_user(user_id, u_pos, velocity)
                if success:
                    self.user_bs_assignments[user_id] = bs_id
                    break  # only assign to one BS

        # Reverse map
        self.bs_user_assignments = defaultdict(list)
        for u, b in self.user_bs_assignments.items():
            self.bs_user_assignments[b].append(u)

        return self.user_bs_assignments

    def _update_base_station_assignments(self):
        """Assign each base station to its nearest controller; ensure everyone has at least one."""
        self.base_station_assignments = {}
        controller_ids = list(self.controller_positions.keys())
        if not controller_ids:
            return {}

        controller_to_bs = defaultdict(list)
        for bs_id, bs_position in enumerate(self.base_station_positions):
            distances = [(cid, np.linalg.norm(bs_position - self.controller_positions[cid]))
                         for cid in controller_ids]
            nearest = min(distances, key=lambda x: x[1])[0]
            self.base_station_assignments[bs_id] = nearest
            controller_to_bs[nearest].append(bs_id)

        self._get_controller_to_bs_mapping()
        return self.base_station_assignments

    def _get_controller_to_bs_mapping(self):
        m = defaultdict(list)
        for bs_id, ctrl_id in self.base_station_assignments.items():
            m[ctrl_id].append(bs_id)
        self.controller_to_bs_map = dict(m)
        return self.controller_to_bs_map

    # ---------- Radio/metrics ----------
    def _update_base_station_metrics(self):
        capacity_metrics = {}
        for bs_id, bs in self.base_stations.items():
            if bs_id in self.base_station_assignments:
                ctrl_id = self.base_station_assignments[bs_id]
                ctrl_pos = self.controller_positions.get(ctrl_id)
                orch_id = self.controller_assignments.get(ctrl_id)
                if orch_id and orch_id.startswith("orchcont"):
                    orch_id = f"orch_{orch_id.split('_')[1]}"
                orch_pos = self.orchestrator_positions.get(orch_id) if orch_id else None

                if ctrl_pos is not None and orch_pos is not None and hasattr(bs, "set_orchestration_efficiency"):
                    eff = self._calculate_orchestration_efficiency(
                        np.linalg.norm(bs.position - ctrl_pos),
                        np.linalg.norm(bs.position - orch_pos)
                    )
                    bs.set_orchestration_efficiency(eff)

            total_cap = bs.calculate_total_capacity()  # (assumed in bps)
            capacity_metrics[bs_id] = {
                "total_capacity": total_cap / 1e6,
                "user_count": len(self.bs_user_assignments.get(bs_id, [])),
                "capacity_per_user": bs.calculate_per_user_capacity(),
                "controller_id": self.base_station_assignments.get(bs_id),
            }

        return capacity_metrics

    def _initialize_power_allocation(self):
        self.total_power_per_bs = {}
        mapping = self._get_controller_to_bs_mapping()
        for _, bs_list in mapping.items():
            for bs_id in bs_list:
                bs = self.base_stations[bs_id]
                self.user_power_allocation.setdefault(bs_id, {})
                users = self.bs_user_assignments.get(bs_id, [])
                if users:
                    usable_power_dBm = bs.transmit_power_dBm - 3.0
                    total_power_linear = 10 ** (usable_power_dBm / 10)
                    equal = total_power_linear / len(users)
                    for u in users:
                        self.user_power_allocation[bs_id][u] = 10 * np.log10(equal)
                    self.total_power_per_bs[bs_id] = usable_power_dBm
                else:
                    self.total_power_per_bs[bs_id] = 0.0
        return self.user_power_allocation

    def _calculate_orchestration_efficiency(self, controller_distance, orchestrator_distance):
        base_eff = 1.0
        max_d = float(np.linalg.norm(np.array(self.dimensions)))
        norm_c = min(controller_distance / (max_d / 2.0), 1.0)
        norm_o = min(orchestrator_distance / max_d, 1.0)

        inter_orch = self._calculate_inter_orchestrator_factor()
        fairness = self._calculate_network_fairness_factor()

        ctrl_factor = 1.5 - norm_c
        orch_factor = 1.5 - 0.8 * norm_o

        eff = base_eff * (0.5 * ctrl_factor + 0.15 * orch_factor + 0.15 * inter_orch + 0.2 * fairness)
        return 0.5 + 1.5 * (1 - np.exp(-2 * eff))

    def _calculate_inter_orchestrator_factor(self):
        active = {oid: pos for oid, pos in self.orchestrator_positions.items() if not np.array_equal(pos, [-1, -1])}
        if len(active) <= 1:
            return 1.0

        positions = list(active.values())
        mind = float("inf")
        dsum, cnt = 0.0, 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                mind = min(mind, d);
                dsum += d;
                cnt += 1
        avg = dsum / cnt if cnt else 0.0

        max_dim = float(max(self.dimensions[0], self.dimensions[1]))
        ideal = max_dim / (len(active) ** 0.5)
        norm_min = min(mind / ideal, 1.0) if ideal > 0 else 1.0

        if cnt > 1 and avg > 0:
            var = sum((float(np.linalg.norm(positions[i] - positions[j])) - avg) ** 2
                      for i in range(len(positions)) for j in range(i + 1, len(positions))) / cnt
            norm_var = min(1.0, var / (max_dim ** 2 / 4.0)) if max_dim > 0 else 0.0
            uniformity = 1.0 - norm_var
        else:
            uniformity = 1.0

        return 0.5 + (0.7 * norm_min + 0.3 * uniformity)

    def _calculate_network_fairness_factor(self):
        bs_throughputs = []
        for bs_id, bs in self.base_stations.items():
            users_on_bs = [u for u, b in self.user_bs_assignments.items() if b == bs_id]
            if not users_on_bs:
                continue
            bs_throughputs.append(bs.calculate_total_capacity() / 1e6)

        if len(bs_throughputs) <= 1:
            return 1.0

        s = sum(bs_throughputs)
        ss = sum(x * x for x in bs_throughputs)
        if ss == 0:
            j = 0.0
        else:
            j = (s * s) / (len(bs_throughputs) * ss)
        return 0.5 + j  # 0.5 .. 1.5

    def _update_interference_model(self):
        for bs in self.base_stations.values():
            bs.interference_sources = []

        for bs_id1, bs1 in self.base_stations.items():
            for bs_id2, bs2 in self.base_stations.items():
                if bs_id1 == bs_id2:
                    continue
                d = float(np.linalg.norm(bs1.position - bs2.position))
                radius = bs2.coverage_radius * 2.5
                if d < radius:
                    user_count = len([u for u, b in self.user_bs_assignments.items() if b == bs_id2])
                    factor = (1 - d / radius) * (1 + 0.1 * user_count)
                    effective_power = bs2.transmit_power_dBm - 10
                    bs1.add_interference_source(bs_id2, bs2.position, effective_power * factor)

    def _calculate_global_fairness_index(self):
        all_user_tput = []
        for bs_id, bs in self.base_stations.items():
            users = [u for u, b in self.user_bs_assignments.items() if b == bs_id]
            if users:
                per_user = bs.calculate_per_user_capacity() / 1e6
                all_user_tput.extend([per_user] * len(users))
        if len(all_user_tput) > 1:
            s = sum(all_user_tput);
            ss = sum(x * x for x in all_user_tput)
            return (s * s) / (len(all_user_tput) * ss) if ss != 0 else 0.0
        return 1.0

    # ---------- RLlib API ----------
    def reset(self, *, seed=None, options=None):
        """
        Reset environment state for a new episode.

        NOTE: We intentionally do NOT recreate agent lists or instances here.
        reset() only resets episode-local state (mobility, positions, metrics).
        Agent populations (self.orchestrator_agents, self.orchcont_agents,
        self.controller_agents) are persistent and may have been changed by the
        previous episode (or during runtime).
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Episode counters / logs
        self.current_step = 0
        self.step_logs = []

        # Reset mobility & user positions
        self.mobility_manager = UserMobilityManager(self.num_users, seed=seed)
        self.mobility_manager.initialize_model()
        self.user_positions = self.mobility_manager.get_positions()

        # Reset orchestrator positions for *current active orchestrators only*
        # (do NOT recreate orchestrator_agents here)
        active_orchs = sorted(list(self.orchestrator_agents))
        if active_orchs:
            # sample hosts for these active orchestrators
            indices = self.rng.choice(len(self.orchestrator_hosts), size=len(active_orchs), replace=False)
            self.orchestrator_positions = {
                orch_id: self.orchestrator_hosts[indices[i]].copy()
                for i, orch_id in enumerate(active_orchs)
            }
        else:
            self.orchestrator_positions = {}

        # Recompute assignments & metrics
        self.orchestrator_controller_assignments()
        self._update_user_bs_assignments()
        self._update_base_station_assignments()
        self._update_base_station_metrics()

        # Update domains (only pass active positions)
        self.domain_manager.update_domains(self.orchestrator_positions)
        self.domain_manager.update_domains(self.controller_positions)

        # We set it to current agent set so first call to step() behaves correctly.
        self._agent_ids = set(
            self.orchestrator_agents | self.orchcont_agents | self.controller_agents
        )
        # Build initial observations and infos only for active agents
        obs = {}
        infos = {}

        for aid in sorted(self._agent_ids):
            if aid.startswith("orch_"):
                obs[aid] = self.orchestrator_agent_instances[aid]._get_orchestrator_observation(aid)
            elif aid.startswith("orchcont_"):
                obs[aid] = self.orchcont_agent_instances[aid]._get_controller_observation(aid)
            elif aid.startswith("ctrl_"):
                obs[aid] = self.controller_agent_instances[aid]._get_power_observation(aid)

        # Infos — match obs exactly
        for aid in obs.keys():
            if aid.startswith("orch_"):
                infos[aid] = {
                    "position": self.orchestrator_positions.get(aid),
                    "num_controllers": sum(1 for c, o in self.controller_assignments.items() if o == aid),
                    "orchestrator_count": float(self.num_orchestrators),
                    "controller_count": float(self.num_controllers),
                    "user_count": float(self.num_users),
                }
            elif aid.startswith("orchcont_"):
                base_orch = f"orch_{aid.split('_')[1]}"
                infos[aid] = {
                    "position": self.orchestrator_positions.get(base_orch, None),
                    "num_controllers": sum(1 for c, o in self.controller_assignments.items() if o == base_orch),
                    "orchestrator_count": float(self.num_orchestrators),
                    "controller_count": float(self.num_controllers),
                    "user_count": float(self.num_users),
                    "selected_orchestrators": getattr(self, "selected_orchestrators", {}).get(aid, []),
                }
            elif aid.startswith("ctrl_"):
                infos[aid] = {
                    "position": self.controller_positions.get(aid),
                    "num_users": sum(1 for u, b in self.user_bs_assignments.items()
                                     if self.base_station_assignments.get(b) == aid),
                    "orchestrator_count": self.num_orchestrators,
                    "controller_count": self.num_controllers,
                }

        return obs, infos

    def step(self, action_dict):
        """
        Main step function. This version:
          - applies actions
          - updates dynamic agent lists BEFORE creating obs/infos
          - detects agents that died this step (previous -> current)
          - returns terminateds that include current agents and agents that died
        """

        self.current_step += 1


        # --- Communication accounting (bytes per round) ---
        self.comm = {
            "ctrl_uplink": {},
            "ctrl_downlink": {},
            "orch_uplink": {},
            "orch_downlink": {},
        }

        # Movement & state
        self.mobility_manager.update_positions()
        self.mobility_manager.initialize_model()
        self.user_positions = self.mobility_manager.get_positions()

        # --- Apply actions (only call methods for agents that exist right now) ---
        for aid in tuple(self.orchestrator_agents):
            if aid in action_dict and aid in self._agent_ids:
                # process_orchestrator_action may mutate orchestrator sets,
                # so be prepared for changes in self._agent_ids afterwards
                self.orchestrator_agent_instances[aid].process_orchestrator_action({aid: action_dict[aid]})

        for aid in tuple(self.orchcont_agents):
            if aid in action_dict and aid in self._agent_ids:
                self.orchcont_agent_instances[aid].process_controller_actions({aid: action_dict[aid]})

        for aid in tuple(self.controller_agents):
            if aid in action_dict and aid in self._agent_ids:
                self.controller_agent_instances[aid].power_allocation_action({aid: action_dict[aid]})

        # --- Recompute network state after actions ---
        self._agent_ids = set(
            self.orchestrator_agents | self.orchcont_agents | self.controller_agents
        )

        self.orchestrator_controller_assignments()
        self._update_base_station_assignments()
        self._update_user_bs_assignments()
        self.update_user_latency_map()
        self._update_interference_model()
        global_fairness = self._calculate_global_fairness_index()
        self._update_base_station_metrics()

        # --- Rewards for active agents only ---
        rewards = {}
        for aid in tuple(self._agent_ids):
            if aid.startswith("orch_"):
                rewards[aid] = self.orchestrator_agent_instances[aid]._calculate_orchestrator_reward(aid)
            elif aid.startswith("orchcont_"):
                rewards[aid] = self.orchcont_agent_instances[aid]._calculate_controller_reward(aid)
            elif aid.startswith("ctrl_"):
                rewards[aid] = self.controller_agent_instances[aid].calculate_reward(aid)

        done = self.current_step >= self.episode_length

        # --- Populate step logs only for active agents ---
        for aid in self._agent_ids:
            self.step_logs.append({
                "step": self.current_step,
                "agent_id": aid,
                "orch_uplink_bytes": self.comm["orch_uplink"].get(aid, 0),
                "orch_downlink_bytes": self.comm["orch_downlink"].get(aid, 0),
                "ctrl_uplink_bytes": self.comm["ctrl_uplink"].get(aid, 0),
                "ctrl_downlink_bytes": self.comm["ctrl_downlink"].get(aid, 0)
            })


        # --- terminateds: include current agents and dead_agents (dead agents get True) ---
        terminateds = {}
        for aid in self._agent_ids:
            terminateds[aid] = done  # normal agents: done only when episode ends

        terminateds["__all__"] = done

        # truncateds (no truncation here)
        truncateds = {aid: False for aid in self._agent_ids}
        truncateds["__all__"] = False

        # --- Observations: build only for currently active agents ---
        obs = {}
        for aid in sorted(self._agent_ids):
            if aid.startswith("orch_"):
                obs[aid] = self.orchestrator_agent_instances[aid]._get_orchestrator_observation(aid)
            elif aid.startswith("orchcont_"):
                obs[aid] = self.orchcont_agent_instances[aid]._get_controller_observation(aid)
            elif aid.startswith("ctrl_"):
                obs[aid] = self.controller_agent_instances[aid]._get_power_observation(aid)

        # --- Infos: must be a subset of obs.keys() (RLlib requirement) ---
        infos = {}
        for aid in obs.keys():
            if aid.startswith("orch_"):
                infos[aid] = {
                    "position": self.orchestrator_positions.get(aid),
                    "num_controllers": sum(1 for c, o in self.controller_assignments.items() if o == aid),
                    "custom_metrics": {
                        "orchestrator_count": self.num_orchestrators,
                        "controller_count": self.num_controllers,
                    }
                }
            elif aid.startswith("orchcont_"):
                base_orch = f"orch_{aid.split('_')[1]}"
                infos[aid] = {
                    "position": self.orchestrator_positions.get(base_orch),
                    "num_controllers": sum(1 for c, o in self.controller_assignments.items() if o == base_orch),
                    "custom_metrics": {
                        "orchestrator_count": self.num_orchestrators,
                        "controller_count": self.num_controllers,
                    }
                }
            elif aid.startswith("ctrl_"):
                pdr_per_user = {}
                for bs_id, bs in self.base_stations.items():
                    if hasattr(bs, "connected_users"):
                        for u in bs.connected_users:
                            pdr_per_user[(bs_id, u)] = self.controller_agent_instances[
                                aid].calculate_packet_delivery_ratio(bs_id, u)

                infos[aid] = {
                    "user_power_allocation": self.user_power_allocation,
                    "pdr_per_user": pdr_per_user,
                    "custom_metrics": {
                        "orchestrator_count": self.num_orchestrators,
                        "controller_count": self.num_controllers,
                    }
                }

        # Record totals and return
        self.record_group_totals(self.current_step, action_dict, obs, rewards)

        self.orchestrator_record(self.current_step)

        self.controller_record(self.current_step)

        os.makedirs(self.output_dir, exist_ok=True)

        return obs, rewards, terminateds, truncateds, infos

    def orchestrator_record(self, step):
        # Filter valid orchestrator-controller agents
        valid_orchs = [
            oid for oid in self.orchcont_agents
            if oid in self.orchcont_agent_instances
               and getattr(self.orchcont_agent_instances[oid], 'orchcont_policy', None) is not None
        ]

        num_orchestrators = len(valid_orchs)
        if num_orchestrators == 0:
            return  # <-- prevents crashes

        # Collect weights from all orchestrators (even though they are identical)
        weights_list = []
        for oid in valid_orchs:
            local_policy = self.orchcont_agent_instances[oid].orchcont_policy
            weights = self.get_weights(local_policy)
            weights_list.append(weights)

        # Use weights from the first orchestrator
        model_size_bytes = sum(w.nbytes for w in weights_list[0].values())

        orchestrator_uplink = model_size_bytes * num_orchestrators

        # Server broadcasts aggregated model to all controllers
        orchestrator_downlink = model_size_bytes * num_orchestrators

        # Total communication per training round
        total_communication = orchestrator_uplink + orchestrator_downlink

        # Log to CSV
        logfile = os.path.join(self.output_dir, "orchestrator_comm_cost.csv")
        os.makedirs(self.output_dir, exist_ok=True)
        write_header = not os.path.exists(logfile)

        with open(logfile, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "step",
                    "model_size_bytes",
                    "num_controllers",
                    "uplink_bytes",
                    "downlink_bytes",
                    "total_bytes"
                ])
            writer.writerow([
                step,
                model_size_bytes,
                num_orchestrators,
                orchestrator_uplink,
                orchestrator_downlink,
                total_communication
            ])

    def controller_record(self, step):
        """
        Record centralized training communication cost for fair comparison.
        Call this at the SAME FREQUENCY as your FL aggregation!
        """

        valid_ctrls = [
            cid for cid in self.controller_agents
            if cid in self.controller_agent_instances
               and getattr(self.controller_agent_instances[cid], 'controller_policy', None) is not None
        ]
        num_controllers = len(valid_ctrls)

        if num_controllers == 0:
            return

        weights_list = []
        for oid in valid_ctrls:
            local_policy = self.controller_agent_instances[oid].controller_policy
            weights = self.get_weights(local_policy)
            weights_list.append(weights)

        model_size_bytes = sum(sum(w.nbytes for w in weights.values()) for weights in weights_list)

        # Centralized training communication:
        # All controllers send models to server
        controller_uplink = model_size_bytes * num_controllers

        # Server broadcasts aggregated model to all controllers
        controller_downlink = model_size_bytes * num_controllers

        # Total communication per training round
        total_communication = controller_uplink + controller_downlink

        # Log to CSV
        logfile = os.path.join(self.output_dir, "controller_comm_cost.csv")
        os.makedirs(self.output_dir, exist_ok=True)
        write_header = not os.path.exists(logfile)

        with open(logfile, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "step",
                    "model_size_bytes",
                    "num_controllers",
                    "uplink_bytes",
                    "downlink_bytes",
                    "total_bytes"
                ])
            writer.writerow([
                step,
                model_size_bytes,
                num_controllers,
                controller_uplink,
                controller_downlink,
                total_communication
            ])

    def get_weights(self, policy):
        """Extract weights from a policy's model."""
        if policy is None:
            raise ValueError("Cannot get weights from None policy")
        if not hasattr(policy, 'model'):
            raise ValueError("Policy has no 'model' attribute")
        return {k: v.detach().cpu().numpy() for k, v in policy.model.state_dict().items()}

    def record_group_totals(self, step, action_dict, obs_dict, rewards_dict):
        import numpy as np
        import csv
        import os

        # Prepare output folder
        os.makedirs(self.output_dir, exist_ok=True)

        # Two groups
        groups = {
            "orchcont": self.orchcont_agents,
            "controller": self.controller_agents
        }

        for group_name, agent_list in groups.items():

            total_obs_bytes = 0
            total_action_bytes = 0
            total_reward_sum = 0.0

            for agent_id in agent_list:

                if agent_id in obs_dict:
                    total_obs_bytes += np.asarray(obs_dict[agent_id]).nbytes

                if agent_id in action_dict:
                    total_action_bytes += np.asarray(action_dict[agent_id]).nbytes

                if agent_id in rewards_dict:
                    reward = float(rewards_dict[agent_id])

                    if group_name == "orchcont":
                        reward = abs(reward)

                    total_reward_sum += reward

            logfile = os.path.join(self.output_dir, f"{group_name}_totals.csv")

            write_header = not os.path.exists(logfile)

            with open(logfile, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["step", "total_obs_bytes", "total_action_bytes", "total_reward_sum"])
                writer.writerow([step, total_obs_bytes, total_action_bytes, total_reward_sum])

