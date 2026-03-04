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
        self.env.ctrl_to_orch_latency = {}
        self.env.controller_avg_latency = {}
        self.orchcont_policy = None

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
        index = int(agent_id.split("_")[-1])
        orch_ids = sorted(self.env.orchestrator_host_indices.keys())

        if index >= len(orch_ids):
            index = 0  # fallback

        orch_id = orch_ids[index]

        G, _ = create_geant_topology(self.env.num_hosts)
        obs = []

        # -------------------------------
        # 1) Own + other orchestrator counts
        # -------------------------------
        other_orchs = [oid for oid in sorted(self.env.orchestrator_agents) if oid != orch_id]
        max_orch = len(self.env.orchestrator_host_indices)

        # Own controller count
        own_ctrls = [c for c, o in self.env.controller_assignments.items() if o == orch_id]
        obs.append(float(len(own_ctrls)))

        # Other orchestrators
        for i in range(max_orch - 1):
            if i < len(other_orchs):
                oid = other_orchs[i]
                ctrls = [c for c, o in self.env.controller_assignments.items() if o == oid]
                obs.append(float(len(ctrls)))
            else:
                obs.append(0.0)

        # -------------------------------
        # 2) Controllers actually deployed
        # -------------------------------
        all_ctrl_ids = sorted(
            c for c in self.env.controller_assignments.keys()
            if c in self.env.controller_host_indices
        )[:26]

        # -------------------------------
        # 3) User count per controller
        # -------------------------------
        user_bs_map = dict(self.env.user_bs_assignments)
        ctrl_to_bs = self.env._get_controller_to_bs_mapping()

        ctrl_user_counts = {c: 0 for c in all_ctrl_ids}

        for _, bs in user_bs_map.items():
            for ctrl_id in all_ctrl_ids:
                if bs in ctrl_to_bs.get(ctrl_id, []):
                    ctrl_user_counts[ctrl_id] += 1
                    break

        for ctrl_id in all_ctrl_ids:
            obs.append(float(ctrl_user_counts[ctrl_id]))

        # -------------------------------
        # 4) Per-controller avg user latency
        # -------------------------------
        for ctrl_id in all_ctrl_ids:
            bs_list = ctrl_to_bs.get(ctrl_id, [])
            ctrl_host = self.env.controller_host_indices[ctrl_id]

            total_ms, cnt = 0.0, 0
            for user, bs in user_bs_map.items():
                if bs not in bs_list:
                    continue

                path1 = nx.shortest_path(G, source=user, target=bs, weight="latency_ms")
                l1 = sum(G[path1[i]][path1[i + 1]]["latency_ms"] for i in range(len(path1) - 1))
                l1 += self.env.user_plane_latency

                path2 = nx.shortest_path(G, source=bs, target=ctrl_host, weight="latency_ms")
                l2 = sum(G[path2[i]][path2[i + 1]]["latency_ms"] for i in range(len(path2) - 1))

                total_ms += l1 + l2
                cnt += 1

            avg_lat = total_ms / max(1, cnt)

            self.env.controller_avg_latency[ctrl_id] = avg_lat

            # Add to observation
            obs.append(avg_lat)

        # -------------------------------
        # 5) Controller → Orchestrator latencies
        # -------------------------------
        orch_host_idx = self.env.orchestrator_host_indices[orch_id]

        for ctrl_id in all_ctrl_ids:
            ctrl_host = self.env.controller_host_indices[ctrl_id]

            path = nx.shortest_path(G, source=ctrl_host, target=orch_host_idx, weight="latency_ms")
            lat = sum(G[path[i]][path[i + 1]]["latency_ms"] for i in range(len(path) - 1))
            lat += self.env.control_plane_latency

            self.env.ctrl_to_orch_latency[ctrl_id] = lat
            obs.append(lat)

        # -------------------------------
        # 6) Pad observation
        # -------------------------------
        MAX_OBS_LEN = 84
        obs = obs[:MAX_OBS_LEN] + [0.0] * max(0, MAX_OBS_LEN - len(obs))

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

                # Ensure at least 2 remain
                if len(active) > 2:
                    # Mark orchestrator as inactive
                    self.env.orchestrator_positions[agent_id] = np.array([-1, -1])

                    # Redistribute its controllers
                    self._reassign_controllers_on_termination(agent_id)

                    # Build paired IDs
                    orchcont_id = agent_id.replace("orch_", "orchcont_")

                    # Remove orchestrator agent
                    self.env.orchestrator_agents.discard(agent_id)
                    self.env.orchestrator_agent_instances.pop(agent_id, None)

                    # Remove orchcont agent (FL level 1)
                    self.env.orchcont_agents.discard(orchcont_id)
                    self.env.orchcont_agent_instances.pop(orchcont_id, None)

                    # Update count
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

                # skip if target already occupied
                occupied = any(np.array_equal(target, pos)
                               for pos in self.env.orchestrator_positions.values()
                               if not np.array_equal(pos, [-1, -1]))
                if occupied:
                    continue

                # ensure limits
                if len(self.env.orchestrator_agents) < num_hosts and \
                        len(self.env.orchestrator_agents) < len(self.env.controller_agents):
                    # build IDs
                    new_orch = f"orch_{logical_id}"
                    new_orchcont = f"orchcont_{logical_id}"

                    # add to positions & agent registries
                    self.env.orchestrator_positions[new_orch] = target

                    self.env.orchestrator_agents.add(new_orch)
                    self.env.orchcont_agents.add(new_orchcont)

                    # instantiate ORCHESTRATOR agent
                    self.env.orchestrator_agent_instances[new_orch] = OrchestratorAgent(self.env)

                    # instantiate ORCHCONT agent (LEVEL-1 FL)
                    self.env.orchcont_agent_instances[new_orchcont] = OrchestratorAgent(
                        self.env,
                    )

                    # update count
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
    # Rewards
    # ---------------------------------------------------------------------

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
        Reward for controller placement policy:
          - penalize high user→BS→controller latency
          - penalize high controller→orchestrator latency
        """
        orch_id = agent_id.replace("orchcont", "orch")

        # If orchestrator not placed, huge penalty
        if np.array_equal(
                self.env.orchestrator_positions.get(orch_id, np.array([-1, -1])),
                [-1, -1]
        ):
            return -100.0

        # Extract float values
        u_lat_list = list(self.env.controller_avg_latency.values())
        co_lat_list = list(self.env.ctrl_to_orch_latency.values())

        # Handle empty lists safely
        u_lat_norm = np.mean(u_lat_list) if u_lat_list else 0.0
        co_lat_norm = np.mean(co_lat_list) if co_lat_list else 0.0


        reward = 1000 - (0.8 * u_lat_norm + 0.2 * co_lat_norm)

        return reward




