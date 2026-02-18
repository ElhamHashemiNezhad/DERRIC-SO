# agents/orchestratoragent.py
import numpy as np
import networkx as nx
from envs.networktopology import create_geant_topology




class OrchestratorAgent:
    """
    Agent responsible for orchestrator actions and controller management in a distributed network.
    """

    def __init__(self, env, seed=None):
        """
        Args:
            env: reference to the environment
            seed: optional RNG seed
        """
        self.env = env
        self.seed = seed
        self.mobility_manager = env.mobility_manager
    # ---------------------------------------------------------------------
    # Orchestrator observation (for orch_* agents)
    # ---------------------------------------------------------------------
    def _get_orchestrator_observation(self, agent_id: str) -> np.ndarray:
        """
        Fixed-shape vector with:
          - own (x,y) position (normalized)
          - other orchestrators' positions (padded)
          - controller counts (own + others, normalized)
          - normalized users belonging to this orchestrator
          - avg end-to-end latency (user→BS→orchestrator), normalized
        NOTE: This reads *current* env state and does not mutate it.
        """
        # Build topology (heavy; keep if acceptable)
        G, _ = create_geant_topology(self.env.num_hosts)

        dims_max = float(max(self.mobility_manager.dimensions))
        obs = []

        # --- 1) Own position (normalized) ---
        own_pos = self.env.orchestrator_positions[agent_id]
        obs.extend([
            np.clip(own_pos[0] / dims_max, -1, 1),
            np.clip(own_pos[1] / dims_max, -1, 1),
        ])

        # Other orchestrators (deterministic order)
        other_ids = [oid for oid in sorted(self.env.orchestrator_agents) if oid != agent_id]
        max_orch = len(self.env.orchestrator_host_indices)  # advertised max slots

        for i in range(max_orch - 1):
            if i < len(other_ids):
                other_pos = self.env.orchestrator_positions[other_ids[i]]
                obs.extend([
                    np.clip(other_pos[0] / dims_max, -1, 1),
                    np.clip(other_pos[1] / dims_max, -1, 1),
                ])
            else:
                obs.extend([0.0, 0.0])

        # --- 2) Controller counts (own + others, normalized) ---
        own_ctrls = [c for c, o in self.env.controller_assignments.items() if o == agent_id]
        obs.append(np.clip(len(own_ctrls) / max(1, self.env.num_controllers), -1, 1))

        for i in range(max_orch - 1):
            if i < len(other_ids):
                oid = other_ids[i]
                ctrls = [c for c, o in self.env.controller_assignments.items() if o == oid]
                obs.append(np.clip(len(ctrls) / max(1, self.env.num_controllers), -1, 1))
            else:
                obs.append(0.0)

        # --- 3) Users belonging to this orchestrator (via its controllers) ---
        # Use snapshots (no env mutation!)
        user_bs_map = dict(self.env.user_bs_assignments)  # {user_id -> bs_id}
        ctrl_to_bs = self.env._get_controller_to_bs_mapping()  # {ctrl_id -> [bs_ids]}
        orch_to_ctrls = {o: [] for o in self.env.orchestrator_agents}
        for ctrl, o in self.env.controller_assignments.items():
            if o in orch_to_ctrls:
                orch_to_ctrls[o].append(ctrl)

        # Count users per controller (by BS assignment)
        ctrl_user_counts = {ctrl: 0 for ctrl in self.env.controller_assignments}
        for _, bs in user_bs_map.items():
            for ctrl_id, bs_list in ctrl_to_bs.items():
                if bs in bs_list:
                    ctrl_user_counts[ctrl_id] += 1
                    break

        # Sum to orchestrator
        orch_user_counts = {
            o: sum(ctrl_user_counts.get(c, 0) for c in orch_to_ctrls[o])
            for o in orch_to_ctrls
        }
        obs.append(np.clip(orch_user_counts.get(agent_id, 0) / max(1, self.env.num_users), -1, 1))

        # --- 4) Avg user→BS→orchestrator latency (normalized) ---
        user_latencies_ms = []
        # Need mapping: BS -> controller, then controller -> orchestrator -> host index
        bs_to_ctrl = dict(self.env.base_station_assignments)  # {bs_id -> ctrl_id}

        # Orchestrator host index from config mapping (keyed by orch_id)
        orch_host_idx_map = self.env.orchestrator_host_indices  # {orch_id -> host_index}

        for user, bs in user_bs_map.items():
            try:
                # user -> BS
                path1 = nx.shortest_path(G, source=user, target=bs, weight="latency_ms")
                lat_user_bs = sum(G[path1[i]][path1[i + 1]]["latency_ms"] for i in range(len(path1) - 1))
            except Exception:
                continue

            ctrl_id = bs_to_ctrl.get(bs)
            if ctrl_id is None:
                continue
            orch_id_for_user = self.env.controller_assignments.get(ctrl_id)
            if orch_id_for_user is None:
                continue
            orch_host_idx = orch_host_idx_map.get(orch_id_for_user)
            if orch_host_idx is None:
                continue

            try:
                # BS -> orchestrator host
                path2 = nx.shortest_path(G, source=bs, target=orch_host_idx, weight="latency_ms")
                lat_bs_orch = sum(G[path2[i]][path2[i + 1]]["latency_ms"] for i in range(len(path2) - 1))
            except Exception:
                continue
            net_path_latency = lat_user_bs + lat_bs_orch
            total_user_plane_latency = net_path_latency + self.env.user_plane_latency
            user_latencies_ms.append(total_user_plane_latency)

        avg_ms = float(np.mean(user_latencies_ms)) if user_latencies_ms else 1000.0
        obs.append(np.clip((avg_ms / 1000.0), -1, 1))
        MAX_OBS_LEN = 28

        if len(obs) < MAX_OBS_LEN:
            obs.extend([0.0] * (MAX_OBS_LEN - len(obs)))
        elif len(obs) > MAX_OBS_LEN:
            obs = obs[:MAX_OBS_LEN]

        return np.asarray(obs, dtype=np.float32)

    # ---------------------------------------------------------------------
    # Orch-controller observation (for orchcont_* agents)
    # ---------------------------------------------------------------------
    def _get_controller_observation(self, agent_id: str) -> np.ndarray:
        """
        Observation for the controller-management policy living at an orchestrator.
        Layout:
          - own controller count (normalized)
          - other orchestrators' controller counts (padded)
          - per-controller user counts (padded to 26)
          - per-controller avg user latency (padded to 26)
          - per-controller controller→orchestrator latency (padded to 26)
        """
        # Map "orchcont_X" -> "orch_X"
        orch_id = agent_id.replace("orchcont", "orch")

        G, _ = create_geant_topology(self.env.num_hosts)
        obs = []

        # Other orchestrators
        other_orchs = [oid for oid in sorted(self.env.orchestrator_agents) if oid != orch_id]
        max_orch = len(self.env.orchestrator_host_indices)

        # 1) Own controller count
        own_ctrls = [c for c, o in self.env.controller_assignments.items() if o == orch_id]
        obs.append(np.clip(len(own_ctrls) / max(1, self.env.num_controllers), -1, 1))

        # 2) Other orchestrators' controller counts
        for i in range(max_orch - 1):
            if i < len(other_orchs):
                oid = other_orchs[i]
                ctrls = [c for c, o in self.env.controller_assignments.items() if o == oid]
                obs.append(np.clip(len(ctrls) / max(1, self.env.num_controllers), -1, 1))
            else:
                obs.append(0.0)

        # 3) Per-controller user counts (max 26)
        user_bs_map = dict(self.env.user_bs_assignments)
        ctrl_to_bs = self.env._get_controller_to_bs_mapping()
        ctrl_user_counts = {ctrl: 0 for ctrl in self.env.controller_positions.keys()}

        for _, bs in user_bs_map.items():
            for ctrl_id, bs_list in ctrl_to_bs.items():
                if bs in bs_list:
                    ctrl_user_counts[ctrl_id] += 1
                    break

        # Deterministic order + cap to 26
        all_ctrl_ids = sorted(ctrl_user_counts.keys())[:26]
        for ctrl_id in all_ctrl_ids:
            obs.append(np.clip(ctrl_user_counts[ctrl_id] / max(1, self.env.num_users), -1, 1))


        # 4) Per-controller avg user latency (max 26)
        per_ctrl_lat = []
        for ctrl_id in all_ctrl_ids:
            bs_list = ctrl_to_bs.get(ctrl_id, [])
            ctrl_host = self.env.controller_host_indices.get(ctrl_id)
            if ctrl_host is None:
                per_ctrl_lat.append(1.0)
                continue

            total_ms, cnt = 0.0, 0
            for user, bs in user_bs_map.items():
                if bs not in bs_list:
                    continue
                try:
                    path1 = nx.shortest_path(G, source=user, target=bs, weight="latency_ms")
                    l1 = sum(G[path1[i]][path1[i + 1]]["latency_ms"] for i in range(len(path1) - 1)) + self.env.user_plane_latency

                    path2 = nx.shortest_path(G, source=bs, target=ctrl_host, weight="latency_ms")
                    l2 = sum(G[path2[i]][path2[i + 1]]["latency_ms"] for i in range(len(path2) - 1))

                    total_ms += (l1 + l2)
                    cnt += 1
                except Exception:
                    continue

            avg_ms = (total_ms / cnt) if cnt > 0 else 1000.0
            per_ctrl_lat.append(np.clip((avg_ms / 1000.0), -1, 1))

        # 5) Controller → Orchestrator latencies (max 26)
        ctrl_to_orch_lat = []
        orch_host_idx = self.env.orchestrator_host_indices.get(orch_id)  # FIX: key by orch_id
        for ctrl_id in all_ctrl_ids:
            ctrl_host = self.env.controller_host_indices.get(ctrl_id)
            if ctrl_host is None or orch_host_idx is None:
                ctrl_to_orch_lat.append(1.0)
                continue
            try:
                path = nx.shortest_path(G, source=ctrl_host, target=orch_host_idx, weight="latency_ms")
                lat = sum(G[path[i]][path[i + 1]]["latency_ms"] for i in range(len(path) - 1)) + self.env.control_plane_latency
                ctrl_to_orch_lat.append(np.clip(lat / 1000.0, -1, 1))
            except Exception:
                ctrl_to_orch_lat.append(1.0)

        MAX_OBS_LEN = 84

        if len(obs) < MAX_OBS_LEN:
            obs.extend([0.0] * (MAX_OBS_LEN - len(obs)))
        elif len(obs) > MAX_OBS_LEN:
            obs = obs[:MAX_OBS_LEN]


        return np.asarray(obs, dtype=np.float32)

    # ---------------------------------------------------------------------
    # Orchestrator actions (terminate / move / duplicate)
    # ---------------------------------------------------------------------
    def process_orchestrator_action(self, action_dict: dict):
        """
        Applies actions for orch_* agents.
          0                 -> terminate (if ≥2 active remain)
          1..N_hosts        -> move to allowed host index
          N_hosts+1..2N     -> duplicate at allowed host (if slot available)
        Only host indexes [0, 5, 6, 9, 10, 11] are valid.
        """
        allowed_host_indices = [0, 5, 6, 9, 10, 11]
        logical_to_array_idx = {i: idx for idx, i in enumerate(allowed_host_indices)}
        num_hosts = len(allowed_host_indices)

        for agent_id, action in action_dict.items():
            if agent_id not in self.env.orchestrator_agents:
                continue

            # --- terminate ---
            if action == 0:
                active = [oid for oid, pos in self.env.orchestrator_positions.items()
                          if not np.array_equal(pos, [-1, -1])]
                if len(active) > 2:
                    self.env.orchestrator_positions[agent_id] = np.array([-1, -1])

                    # Reassign controllers from terminated orch
                    self._reassign_controllers_on_termination(agent_id)

                    # Remove agents
                    self.env.orchestrator_agents.discard(agent_id)
                    self.env.orchestrator_agent_instances.pop(agent_id, None)

                    # Paired orchcont
                    orchcont_id = agent_id.replace("orch", "orchcont")

                    self.env.orchcont_agents.discard(orchcont_id)
                    self.env.orchestrator_agent_instances.pop(orchcont_id, None)

                    self.env.num_orchestrators = len(self.env.orchestrator_agents)

            # --- move ---
            elif 1 <= action <= num_hosts:
                logical_id = allowed_host_indices[action - 1]
                array_idx = logical_to_array_idx[logical_id]
                target = self.env.orchestrator_hosts[array_idx].copy()

                # skip if occupied by another active orch
                occupied = any(np.array_equal(target, pos)
                               for oid, pos in self.env.orchestrator_positions.items()
                               if oid != agent_id and not np.array_equal(pos, [-1, -1]))
                if not occupied:
                    self.env.orchestrator_positions[agent_id] = target

            # --- duplicate ---
            elif num_hosts < action <= 2 * num_hosts:
                logical_id = allowed_host_indices[action - num_hosts - 1]
                array_idx = logical_to_array_idx[logical_id]
                target = self.env.orchestrator_hosts[array_idx].copy()

                # skip if target host is already occupied
                occupied = any(np.array_equal(target, pos)
                               for pos in self.env.orchestrator_positions.values()
                               if not np.array_equal(pos, [-1, -1]))
                if occupied:
                    continue

                # keep under limits
                if len(self.env.orchestrator_agents) < num_hosts and \
                        len(self.env.orchestrator_agents) < len(self.env.controller_agents):

                    # create ID based on logical host index
                    new_orch = f"orch_{logical_id}"
                    new_orchcont = f"orchcont_{logical_id}"


                    # assign position and add agents
                    self.env.orchestrator_positions[new_orch] = target
                    self.env.orchestrator_agents.add(new_orch)
                    self.env.orchcont_agents.add(new_orchcont)

                    # instantiate paired agents
                    self.env.orchestrator_agent_instances[new_orch] = OrchestratorAgent(self.env)
                    self.env.orchestrator_agent_instances[new_orchcont] = OrchestratorAgent(self.env)
                    self.env.num_orchestrators = len(self.env.orchestrator_agents)

        # After all actions: update domains and assignments
        self.env.update_domains()
        self.env.orchestrator_controller_assignments()
        self.env._update_base_station_assignments()
        self.env._update_base_station_metrics()

    # ---------------------------------------------------------------------
    # Controller placement actions (for orchcont_* agents)
    # ---------------------------------------------------------------------
    def process_controller_actions(self, action_dict: dict):
        """
        action_dict: {orchcont_*: MultiDiscrete([0/1 ...])} where 1 = deploy, 0 = remove
        """
        host_positions = list(self.env.controller_hosts)

        for agent_id, action_array in action_dict.items():
            orch_id = agent_id.replace("orchcont", "orch")

            # skip if orchestrator inactive
            if np.array_equal(self.env.orchestrator_positions.get(orch_id, np.array([-1, -1])), [-1, -1]):
                continue

            action_array = np.asarray(action_array)
            if not hasattr(self.env, "last_actions"):
                self.env.last_actions = {}
            self.env.last_actions[agent_id] = np.asarray(action_array, dtype=np.int32)

            for i, host_pos in enumerate(host_positions):
                if i >= len(action_array):
                    break

                # find controller at host_pos if any
                existing_ctrl = None
                for ctrl_id, pos in self.env.controller_positions.items():
                    if np.allclose(pos, host_pos, atol=1e-6):
                        existing_ctrl = ctrl_id
                        break

                if action_array[i] == 1:
                    # deploy
                    if existing_ctrl is None and self.env.num_controllers < self.env.num_base_stations:
                        # pick the lowest unused numeric suffix
                        used = set()
                        for cid in self.env.controller_positions:
                            try:
                                used.add(int(cid.split("_")[1]))
                            except Exception:
                                pass
                        new_idx = 0
                        while new_idx in used:
                            new_idx += 1
                        new_ctrl_id = f"ctrl_{new_idx}"

                        self.env.add_controller(new_ctrl_id, host_pos.copy(), orch_id)
                        self.env.orchestrator_controller_assignments()
                        self.env._update_base_station_assignments()
                        self.env._update_base_station_metrics()
                        self.env._initialize_power_allocation()

                else:
                    # remove
                    if existing_ctrl:
                        owner = self.env.controller_assignments.get(existing_ctrl)
                        owner_ctrls = [cid for cid, o in self.env.controller_assignments.items() if o == owner]
                        # keep at least one controller per owner and total controllers ≥ orchestrators
                        if len(owner_ctrls) > 1 and len(self.env.controller_agents) > len(self.env.orchestrator_agents):
                            self.env.remove_controller(existing_ctrl)
                            # re-assign & recompute
                            self.env.orchestrator_controller_assignments()
                            self.env._update_base_station_assignments()
                            self.env._update_base_station_metrics()
                            self.env._initialize_power_allocation()

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _reassign_controllers_on_termination(self, terminated_orch_id: str):
        """
        When an orchestrator is terminated, reassign its controllers to nearest active orchestrator.
        """
        orphaned = [c for c, o in self.env.controller_assignments.items() if o == terminated_orch_id]
        active_orchs = [
            o for o, pos in self.env.orchestrator_positions.items()
            if not np.array_equal(pos, [-1, -1]) and o != terminated_orch_id
        ]
        if not active_orchs:
            return

        for ctrl_id in orphaned:
            ctrl_pos = self.env.controller_positions.get(ctrl_id)
            if ctrl_pos is None:
                continue
            closest = min(
                active_orchs,
                key=lambda o: np.linalg.norm(self.env.orchestrator_positions[o] - ctrl_pos)
            )
            self.env.controller_assignments[ctrl_id] = closest

    # ---------------------------------------------------------------------
    # Rewards (delegated to env metrics; kept here for symmetry)
    # ---------------------------------------------------------------------
    def calculate_reward(self, agent_id: str) -> float:
        """
        Delegates to specialized reward calculators based on agent type.
        """
        if agent_id in self.env.orchestrator_agents:
            return self._calculate_orchestrator_reward(agent_id)
        if agent_id in self.env.orchcont_agents:
            return self._calculate_controller_reward(agent_id)
        return 0.0

    def _calculate_orchestrator_reward(self, agent_id: str) -> float:
        """
        Throughput + Jain fairness across users of this orchestrator.
        """
        if np.array_equal(self.env.orchestrator_positions[agent_id], [-1, -1]):
            return 0.0

        own_ctrls = [c for c, o in self.env.controller_assignments.items() if o == agent_id]
        own_bs = [bs for bs, c in self.env.base_station_assignments.items() if c in own_ctrls]
        if not own_bs:
            return 0.1

        bs_metrics = self.env._update_base_station_metrics()
        total_cap_mbps, total_users = 0.0, 0
        per_user_list = []

        for bs_id in own_bs:
            if bs_id in bs_metrics:
                m = bs_metrics[bs_id]
                total_cap_mbps += m["total_capacity"]
                total_users += m["user_count"]
                if m["user_count"] > 0:
                    per_user_list.extend([m["capacity_per_user"] / 1e6] * m["user_count"])

        thr_r = 0.0
        if total_users > 0:
            avg_thr = total_cap_mbps / total_users
            thr_r = min(1.0, avg_thr / 10.0)

        fairness = 0.0
        if per_user_list:
            s = sum(per_user_list)
            ss = sum(x * x for x in per_user_list)
            fairness = (s * s) / (len(per_user_list) * ss) if ss > 0 else 0.0

        return float(6 * thr_r + 4 * fairness)

    def _calculate_controller_reward(self, agent_id: str) -> float:
        """
        Reward for controller placement policy at an orchestrator:
          - mean( user→BS + BS→controller latency ) [high is bad]
          - mean( controller→orchestrator latency ) [high is bad]
        """
        orch_id = agent_id.replace("orchcont", "orch")
        # Ensure last_actions dictionary exists
        if not hasattr(self.env, "last_actions"):
            self.env.last_actions = {}

        if np.array_equal(self.env.orchestrator_positions.get(orch_id, np.array([-1, -1])), [-1, -1]):
            return 0.0

        own_ctrls = [c for c, o in self.env.controller_assignments.items() if o == orch_id]
        G, _ = create_geant_topology(self.env.num_hosts)

        own_bs = [bs for bs, c in self.env.base_station_assignments.items() if c in own_ctrls]
        user_bs_map = dict(self.env.user_bs_assignments)

        # 1) User-Controller latencies
        u_lat_rewards = []
        min_lat = nx.get_edge_attributes(G, "latency_ms").values()
        min_lat, max_lat = min(min_lat), max(min_lat)
        for user_id, bs_id in user_bs_map.items():
            if bs_id not in own_bs:
                continue
            ctrl_id = self.env.base_station_assignments.get(bs_id)
            ctrl_host = self.env.controller_host_indices.get(ctrl_id)
            if ctrl_host is None:
                continue
            try:
                p1 = nx.shortest_path(G, source=user_id, target=bs_id, weight="latency_ms")
                l1 = sum(G[p1[i]][p1[i + 1]]["latency_ms"] for i in range(len(p1) - 1)) + self.env.user_plane_latency
                p2 = nx.shortest_path(G, source=bs_id, target=ctrl_host, weight="latency_ms")
                l2 = sum(G[p2[i]][p2[i + 1]]["latency_ms"] for i in range(len(p2) - 1))
                total = l1 + l2
                u_lat_rewards.append(total)
            except Exception:
                u_lat_rewards.append(0.0)

        # 2) Controller-Orchestrator latencies
        co_lat_rewards = []
        orch_host = self.env.orchestrator_host_indices.get(orch_id)
        if orch_host is not None:
            for ctrl_id in own_ctrls:
                ctrl_host = self.env.controller_host_indices.get(ctrl_id)
                if ctrl_host is None:
                    co_lat_rewards.append(0.0)
                    continue
                try:
                    p = nx.shortest_path(G, source=ctrl_host, target=orch_host, weight="latency_ms")
                    lat = sum(G[p[i]][p[i + 1]]["latency_ms"] for i in range(len(p) - 1)) + self.env.control_plane_latency
                    co_lat_rewards.append(lat)
                except Exception:
                    co_lat_rewards.append(0.0)

        # Weighted combination
        mean_u = float(np.mean(u_lat_rewards)) if u_lat_rewards else 0.0
        mean_co = float(np.mean(co_lat_rewards)) if co_lat_rewards else 0.0
        max_latency = 150

        reward = 1.0 - ((0.8 * mean_u + 0.2 * mean_co) / max_latency)

        return reward



