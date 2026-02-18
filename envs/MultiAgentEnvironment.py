import csv
import os
from collections import defaultdict
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

        # RNG
        self.rng = np.random.default_rng(self.seed)

        # ---- Mobility & positions ----
        self.mobility_manager = UserMobilityManager(self.num_users, seed=self.seed)
        self.mobility_manager.initialize_model()
        self.user_positions = self.mobility_manager.get_positions()
        self.dimensions = self.mobility_manager.dimensions

        # Hosts laid out on a line (keep your original idea)
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
        self.orchestrator_agents = set(self.orchestrator_host_indices.keys())
        self.orchcont_agents = {oid.replace("orch", "orchcont") for oid in self.orchestrator_agents}

        self.controller_agents   = {f"ctrl_{i+1}"     for i in range(self.num_controllers)}

        self._agent_ids = set().union(
            self.orchestrator_agents,
            self.orchcont_agents,
            self.controller_agents,
        )
        self.global_exp_buffer = defaultdict(list)
        # Orchestrator positions: sample and map to actual agent IDs
        selected = self.rng.choice(len(self.orchestrator_hosts), len(self.orchestrator_agents), replace=False)
        self.orchestrator_positions = {}
        for i, orch_id in enumerate(sorted(self.orchestrator_agents)):
            self.orchestrator_positions[orch_id] = np.array(self.orchestrator_hosts[selected[i]], dtype=np.float64)

        # Controller host positions (anything that’s not an orchestrator host)
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

        # ---- Assignments & managers ----
        self.user_power_allocation   = {}
        self.controller_assignments  = {}
        self.user_bs_assignments     = {}
        self.bs_user_assignments     = {}
        self.base_station_assignments= {}
        self.controller_to_bs_map    = {}

        self.domain_manager = VoronoiDomainManager(self)

        # Initial assignments
        self._update_user_bs_assignments()
        self._update_base_station_assignments()
        self._initialize_power_allocation()

        # ---- Action/Observation spaces (derive sizes, avoid magic numbers) ----
        # NOTE: ctrl_candidates = hosts minus reserved orchestrator hosts
        ctrl_candidates = max(0, self.num_hosts - len(self.orchestrator_host_indices))

        # Orchestrator actions: move/terminate/duplicate etc.
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

        # Observations — compute sizes from config when possible; keep your comments
        self.observation_orchestrator_spaces = {
            agent: Box(low=-1, high=1, shape=(28,), dtype=np.float32)
            for agent in self.orchestrator_agents
        }
        # If you want this to be dynamic, compute it; otherwise leave as fixed and ensure your agent matches it.
        self.observation_orchcont_spaces = {
            agent: Box(low=-1, high=1, shape=(84,), dtype=np.float32)
            for agent in self.orchcont_agents
        }
        self.observation_controller_spaces = {
            agent: Box(low=-1, high=1, shape=(3 * self.num_users + 26,), dtype=np.float32)
            for agent in self.controller_agents
        }

        # ---- Topology ----
        self.G, self.link_properties = create_geant_topology(self.num_hosts)


        self.orchestrator_agent_instances = {
            agent_id: OrchestratorAgent(self, seed=self.seed)
            for agent_id in (self.orchestrator_agents | self.orchcont_agents)
        }


        self.controller_agent_instances = {
            ctrl_id: ControllerAgent(self, controller_id=ctrl_id)
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
        self.controller_agent_instances[ctrl_id] = ControllerAgent(self, controller_id=ctrl_id)
        self._agent_ids.add(ctrl_id)
        self.num_controllers = len(self.controller_agents)

    def remove_controller(self, ctrl_id):
        self.controller_agents.discard(ctrl_id)
        self.controller_positions.pop(ctrl_id, None)
        self.controller_assignments.pop(ctrl_id, None)
        self.controller_agent_instances.pop(ctrl_id, None)
        self._agent_ids.discard(ctrl_id)
        self.num_controllers = len(self.controller_agents)

    # ---------- Assignments / domains ----------
    def orchestrator_controller_assignments(self):
        """Evenly assign controllers to alive orchestrators."""
        orch_ids = [oid for oid, pos in self.orchestrator_positions.items()
                    if not np.array_equal(pos, [-1, -1])]
        if not orch_ids:
            return
        ctrl_ids = list(self.controller_positions.keys())
        for idx, ctrl_id in enumerate(ctrl_ids):
            self.controller_assignments[ctrl_id] = orch_ids[idx % len(orch_ids)]

    def _update_user_bs_assignments(self):
        """Assign users to base stations with basic fairness-aware heuristic."""
        old = dict(getattr(self, "user_bs_assignments", {}))
        self.user_bs_assignments = {}

        for bs in self.base_stations.values():
            bs.connected_users.clear()
            bs.allocated_resource_blocks = 0

        # Prioritize keeping connections
        priority = []
        for u_id, u_pos in enumerate(self.user_positions):
            if u_id in old:
                old_bs_id = old[u_id]
                old_pos   = self.base_stations[old_bs_id].position
                d = np.linalg.norm(u_pos - old_pos)
                priority.append((u_id, d, True))
            else:
                priority.append((u_id, 0.0, False))
        priority.sort(key=lambda x: (not x[2], x[1]))

        # Precompute capacity-normalized load
        bs_load_factors = {}
        max_users_cap = max(bs.max_users for bs in self.base_stations.values())
        for bs_id, bs in self.base_stations.items():
            capacity_factor = bs.max_users / max_users_cap if max_users_cap > 0 else 1.0
            current_users = 0
            load_factor = current_users / bs.max_users if bs.max_users > 0 else 1.0
            bs_load_factors[bs_id] = load_factor / max(1e-6, capacity_factor)

        for user_id, _, had_prev in priority:
            u_pos = self.user_positions[user_id]
            u_vel = float(np.linalg.norm(self.mobility_manager.velocities[user_id]))

            scored = []
            for bs_id, bs in self.base_stations.items():
                dist = np.linalg.norm(u_pos - bs.position)
                if dist > bs.coverage_radius:
                    continue
                if len([u for u, b in self.user_bs_assignments.items() if b == bs_id]) >= bs.max_users:
                    continue

                normalized_distance = dist / bs.coverage_radius
                continuity_bonus = 0.2 if (had_prev and old.get(user_id) == bs_id) else 0.0
                score = 0.7 * normalized_distance + 0.3 * bs_load_factors[bs_id] - continuity_bonus
                scored.append((score, bs_id, dist))

            scored.sort()
            for _, bs_id, _ in scored:
                bs = self.base_stations[bs_id]
                self.user_bs_assignments[user_id] = bs_id
                if hasattr(bs, "connect_user"):
                    if not bs.connect_user(user_id=user_id, position=u_pos, velocity=u_vel):
                        continue
                # update load factor (simple)
                connected = len([u for u, b in self.user_bs_assignments.items() if b == bs_id])
                bs_load_factors[bs_id] = connected / max(1, bs.max_users)
                break

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

        # Ensure each controller has ≥1 BS
        for cid in controller_ids:
            if cid not in controller_to_bs:
                # give it one from the controller with most BSs
                donor, bs_list = max(controller_to_bs.items(), key=lambda kv: len(kv[1]))
                moved = bs_list.pop()
                self.base_station_assignments[moved] = cid
                controller_to_bs[cid] = [moved]

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

        # Reconnect users to their BS (idempotent if BaseStation handles duplicates)
        for user_id, bs_id in self.user_bs_assignments.items():
            if bs_id in self.base_stations:
                bs = self.base_stations[bs_id]
                u_pos = self.user_positions[user_id]
                u_vel = float(np.linalg.norm(self.mobility_manager.velocities[user_id]))
                bs.connect_user(user_id, u_pos, u_vel)

        for bs_id, bs in self.base_stations.items():
            if bs_id in self.base_station_assignments:
                ctrl_id = self.base_station_assignments[bs_id]
                ctrl_pos = self.controller_positions.get(ctrl_id)
                orch_id  = self.controller_assignments.get(ctrl_id)
                if orch_id and orch_id.startswith("orchcont"):
                    orch_id = f"orch_{orch_id.split('_')[1]}"
                orch_pos = self.orchestrator_positions.get(orch_id) if orch_id else None

                if ctrl_pos is not None and orch_pos is not None and hasattr(bs, "update_control_plane_metrics"):
                    bs.update_control_plane_metrics(ctrl_pos, orch_pos)

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
        fairness   = self._calculate_network_fairness_factor()

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
                mind = min(mind, d); dsum += d; cnt += 1
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
            s = sum(all_user_tput); ss = sum(x * x for x in all_user_tput)
            return (s * s) / (len(all_user_tput) * ss) if ss != 0 else 0.0
        return 1.0

    # ---------- RLlib API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_step = 0

        # Mobility & users
        self.mobility_manager = UserMobilityManager(self.num_users, seed=seed)
        self.mobility_manager.initialize_model()
        self.user_positions   = self.mobility_manager.get_positions()

        # Orchestrators on random high-cap hosts (use actual agent IDs)
        indices = self.rng.choice(len(self.orchestrator_hosts), len(self.orchestrator_agents), replace=False)
        self.orchestrator_positions = {}
        for i, orch_id in enumerate(sorted(self.orchestrator_agents)):
            self.orchestrator_positions[orch_id] = self.orchestrator_hosts[indices[i]].copy()

        # Keep controller positions; update assignments & metrics
        self.orchestrator_controller_assignments()
        self._update_user_bs_assignments()
        self._update_base_station_assignments()
        self._update_base_station_metrics()

        # Recreate agent instances (if they keep internal state per episode)
        self.orchestrator_agent_instances = {
            agent_id: OrchestratorAgent(self, seed=self.seed)
            for agent_id in (self.orchestrator_agents | self.orchcont_agents)
        }

        self.controller_agent_instances = {
            ctrl_id: ControllerAgent(self, controller_id=ctrl_id)
            for ctrl_id in self.controller_agents
        }

        # Domains
        self.domain_manager.update_domains(self.orchestrator_positions)
        self.domain_manager.update_domains(self.controller_positions)

        # Observations
        obs = {}
        for aid in self.orchestrator_agents:
            obs[aid] = self.orchestrator_agent_instances[aid]._get_orchestrator_observation(aid)
        for aid in self.orchcont_agents:
            obs[aid] = self.orchestrator_agent_instances[aid]._get_controller_observation(aid)
        for aid in self.controller_agents:
            obs[aid] = self.controller_agent_instances[aid]._get_power_observation(aid)

        # Infos
        infos = {}
        for aid in self.orchestrator_agents:
            infos[aid] = {
                "position": self.orchestrator_positions[aid],
                "num_controllers": sum(1 for c in self.controller_assignments.values() if c == aid),
                "orchestrator_count": float(self.num_orchestrators),
                "controller_count": float(self.num_controllers),
                "user_count": float(self.num_users),
            }
        for aid in self.orchcont_agents:
            orch_id = f"orch_{aid.split('_')[1]}"
            infos[aid] = {
                "position": self.orchestrator_positions.get(orch_id, None),
                "num_controllers": sum(1 for c in self.controller_assignments.values() if c == orch_id),
                "orchestrator_count": float(self.num_orchestrators),
                "controller_count": float(self.num_controllers),
                "user_count": float(self.num_users),
            }

        for aid in self.controller_agents:
            infos[aid] = {
                "position": self.controller_positions.get(aid),
                "num_users": sum(1 for u, b in self.user_bs_assignments.items()
                                 if self.base_station_assignments.get(b) == aid),
                "orchestrator_count": self.num_orchestrators,
                "controller_count": self.num_controllers,
            }

        return obs, infos

    def step(self, action_dict):
        self.current_step += 1

        # Movement & state
        self.mobility_manager.update_positions()
        self.mobility_manager.initialize_model()
        self.user_positions = self.mobility_manager.get_positions()

        # Apply actions once per agent type (avoid duplicates)
        for aid in tuple(self.orchestrator_agents):
            if aid in action_dict:
                self.orchestrator_agent_instances[aid].process_orchestrator_action({aid: action_dict[aid]})
        for aid in tuple(self.orchcont_agents):
            if aid in action_dict:
                self.orchestrator_agent_instances[aid].process_controller_actions({aid: action_dict[aid]})
        for aid in tuple(self.controller_agents):
            if aid in action_dict:
                self.controller_agent_instances[aid].power_allocation_action({aid: action_dict[aid]})

        # Recompute network state
        self._update_base_station_assignments()
        self._update_user_bs_assignments()
        self._update_interference_model()
        global_fairness = self._calculate_global_fairness_index()
        self._update_base_station_metrics()


        # Rewards
        rewards = {}
        for aid in tuple(self.orchestrator_agents | self.orchcont_agents):
            rewards[aid] = self.orchestrator_agent_instances[aid].calculate_reward(aid)

        for aid in tuple(self.controller_agents):
            rewards[aid] = self.controller_agent_instances[aid].calculate_reward(aid)

        for aid in tuple(self.orchcont_agents):
            orch_id = f"orch_{aid.split('_')[1]}"
            own_ctrls = [c for c, o in self.controller_assignments.items() if o == orch_id]

            if own_ctrls and not np.array_equal(self.orchestrator_positions.get(orch_id, [-1, -1]), [-1, -1]):
                try:
                    obs = self.orchestrator_agent_instances[aid]._get_controller_observation(aid)
                    act = np.asarray(self.last_actions.get(aid, np.zeros(26, np.int32)), dtype=np.int32)
                    rew = np.asarray(rewards[aid], dtype=np.float32)

                    # Store in ENVIRONMENT'S buffer (not agent's)
                    for ctrl_id in own_ctrls:
                        self.global_exp_buffer[ctrl_id].append(  # Use self.global_exp_buffer
                            (obs, act, rew)
                        )
                except Exception as e:
                    print(f"[WARN] Could not record experience for {aid}: {e}")

        # flush communication logs
        self._flush_nonfl_comm(self.current_step)

        done = self.current_step >= self.episode_length
        current_agent_ids = tuple(
            self.orchestrator_agents | self.orchcont_agents | self.controller_agents
        )
        terminateds = {aid: done for aid in current_agent_ids}
        terminateds["__all__"] = done
        truncateds  = {aid: False for aid in self._agent_ids}
        truncateds["__all__"] = False

        # Observations
        obs = {}
        for aid in tuple(self.orchestrator_agents):
            obs[aid] = self.orchestrator_agent_instances[aid]._get_orchestrator_observation(aid)
        for aid in tuple(self.orchcont_agents):
            obs[aid] = self.orchestrator_agent_instances[aid]._get_controller_observation(aid)
        for aid in tuple(self.controller_agents):
            obs[aid] = self.controller_agent_instances[aid]._get_power_observation(aid)

        # Infos
        infos = {}
        for aid in tuple(self.orchestrator_agents):
            infos[aid] = {
                "position": self.orchestrator_positions[aid],
                "num_controllers": sum(1 for c in self.controller_assignments.values() if c == aid),
                "custom_metrics": {
                    "orchestrator_count": self.num_orchestrators,
                    "controller_count": self.num_controllers,
                    "global_fairness": global_fairness,
                }
            }
        for aid in tuple(self.orchcont_agents):
            base_orch = f"orch_{aid.split('_')[1]}"
            infos[aid] = {
                "position": self.orchestrator_positions.get(base_orch),
                "num_controllers": sum(1 for c in self.controller_assignments.values() if c == base_orch),
                "custom_metrics": {
                    "orchestrator_count": self.num_orchestrators,
                    "controller_count": self.num_controllers,
                }
            }
        for aid in tuple(self.controller_agents):
            # Per-user PDR map (bs_id, user_id) -> pdr
            pdr_per_user = {}
            for bs_id, bs in self.base_stations.items():
                if hasattr(bs, "connected_users"):
                    for u in bs.connected_users:
                        pdr_per_user[(bs_id, u)] = self.controller_agent_instances[aid].calculate_packet_delivery_ratio(bs_id, u)
            infos[aid] = {
                "user_power_allocation": self.user_power_allocation,
                "pdr_per_user": pdr_per_user,
                "custom_metrics": {
                    "orchestrator_count": self.num_orchestrators,
                    "controller_count": self.num_controllers,
                }
            }


        return obs, rewards, terminateds, truncateds, infos

    def _flush_nonfl_comm(self, round_num: int):
        """Write total non-FL communication bytes per orchestrator."""

        def exp_bytes(obs: np.ndarray, act: np.ndarray, rew: np.ndarray) -> int:
            return obs.nbytes + act.nbytes + rew.nbytes

        buffer = self.global_exp_buffer

        log_path = os.path.join(self.output_dir, "nonfl_communication_metrics.csv")
        write_header = not os.path.exists(log_path)

        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["round", "orchestrator_id", "total_controllers",
                            "total_samples", "total_data_bytes"])

            # Group by orchestrator
            orch_groups = defaultdict(list)
            for ctrl_id, samples in buffer.items():
                orch_id = self.controller_assignments.get(ctrl_id)
                if orch_id:
                    orch_groups[orch_id].extend(samples)

            # Write one line per orchestrator
            for orch_id, samples in orch_groups.items():
                total_bytes = sum(exp_bytes(o, a, r) for (o, a, r) in samples)
                total_samples = len(samples)
                ctrl_count = sum(1 for c, o in self.controller_assignments.items() if o == orch_id)

                w.writerow([round_num, orch_id, ctrl_count, total_samples, total_bytes])

            # Global summation
            if orch_groups:
                global_ctrls = sum(sum(1 for c, o in self.controller_assignments.items() if o == orch)
                                   for orch in orch_groups.keys())
                global_samples = sum(len(samples) for samples in orch_groups.values())
                global_bytes = sum(sum(exp_bytes(o, a, r) for (o, a, r) in samples)
                                   for samples in orch_groups.values())
                w.writerow([round_num, "global", global_ctrls, global_samples, global_bytes])

        buffer.clear()