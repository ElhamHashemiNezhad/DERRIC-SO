import os
import numpy as np
import pandas as pd


class ControllerAgent:
    """
    Controller Agent that manages power allocation to users across base stations.
    Power decisions now flow through to SINR/capacity -> PDR -> reward.
    """

    def __init__(
        self,
        env,
        controller_id,
        controller_policy,
        seed=None,
        max_power_per_bs: float = 43.0,  # dBm (≈20 W)
        target_pdr: float = 0.95,
    ):
        self.env = env
        self.controller_id = controller_id
        self.seed = seed
        self.mobility_manager = env.mobility_manager
        self.max_power_per_bs = float(max_power_per_bs)
        self.target_pdr = float(target_pdr)
        self.base_stations = self.env.base_stations
        self.controller_policy = controller_policy


    # ----------------------- OBSERVATION -----------------------
    def _get_power_observation(self, agent_id):
        """
        Local observation for controller-side power RL.

        For up to MAX_USERS_PER_CTRL users managed by this controller:
            [ latency_norm, snr_norm, capacity_norm, power_norm ]

        Final shape = 4 * MAX_USERS_PER_CTRL
        """

        # Safety check: unknown controller
        if agent_id not in self.env.controller_agents:
            return np.zeros(self.env.OBS_DIM_CONTROLLER, dtype=np.float32)

        # Load global constants computed at reset()
        MAX_USERS_PER_CTRL = self.env.MAX_USERS_PER_CTRL
        OBS_DIM = self.env.OBS_DIM_CONTROLLER  # = 4 * MAX_USERS_PER_CTRL

        ctrl_id = agent_id
        ctrl_to_bs = self.env._get_controller_to_bs_mapping()
        bs_list = ctrl_to_bs.get(ctrl_id, [])

        # -----------------------------------------------------
        # Collect all local (bs_id, user_id) pairs for this controller
        # -----------------------------------------------------
        pairs = []
        for bs_id in bs_list:
            bs = self.env.base_stations.get(bs_id)
            if bs is None:
                continue

            for u_id in bs.connected_users.keys():
                pairs.append((bs_id, u_id))

        # -----------------------------------------------------
        # Truncate to max allowed
        # -----------------------------------------------------
        pairs = pairs[:MAX_USERS_PER_CTRL]

        features = []

        # -----------------------------------------------------
        # Extract per-user features
        # -----------------------------------------------------
        for bs_id, user_id in pairs:

            bs = self.env.base_stations.get(bs_id)
            if bs is None or user_id not in bs.connected_users:
                # dead entry → all-zero encoding
                features.extend([0.0, 0.0, 0.0, 0.0])
                continue

            udata = bs.connected_users[user_id]

            # ---- Latency: normalize [0..200] ms → [-1,1]
            lat = float(udata.get("latency_ms", 0.0))
            lat = np.clip(lat, 0.0, 200.0)
            lat_norm = 2.0 * (lat / 200.0) - 1.0

            # ---- SNR: normalize [-10..40] dB → [-1,1]
            snr = float(udata.get("sinr", -10.0))
            snr = np.clip(snr, -10.0, 40.0)
            snr_norm = 2.0 * ((snr + 10.0) / 50.0) - 1.0

            # ---- Capacity: normalize [0..1Gbps] → [-1,1]
            cap = float(udata.get("capacity", 0.0))
            cap = np.clip(cap, 0.0, 1e9)
            cap_norm = 2.0 * (cap / 1e9) - 1.0

            # ---- Current power: normalize [0..43 dBm] → [-1,1]
            p_dbm = 0.0
            if bs_id in self.env.user_power_allocation:
                p_dbm = float(self.env.user_power_allocation[bs_id].get(user_id, 0.0))
                if not np.isfinite(p_dbm):
                    p_dbm = 0.0
            p_dbm = np.clip(p_dbm, 0.0, 43.0)
            pwr_norm = 2.0 * (p_dbm / 43.0) - 1.0

            features.extend([lat_norm, snr_norm, cap_norm, pwr_norm])

        # -----------------------------------------------------
        # Pad if fewer than MAX_USERS_PER_CTRL
        # -----------------------------------------------------
        current_users = len(pairs)
        if current_users < MAX_USERS_PER_CTRL:
            pad_entries = (MAX_USERS_PER_CTRL - current_users) * 4
            features.extend([0.0] * pad_entries)

        return np.array(features, dtype=np.float32)

        # ----------------------- ACTION -----------------------
    def power_allocation_action(self, action):
        """
        Allocate per-user transmit power based on the agent's action values.
        Each user's power is proportional to its action value, and total BS power
        does not exceed the BS's maximum transmit power (default 43 dBm).
        """

        # --- Flatten and sanitize the action input ---
        if isinstance(action, dict) and len(action) == 1:
            action = next(iter(action.values()))
        if isinstance(action, (np.ndarray, list, tuple)):
            actions = [float(x) for x in action]
        elif isinstance(action, (int, float, np.integer, np.floating)):
            actions = [float(action)]
        else:
            actions = []

        # --- Get BSs managed by this controller ---
        bs_list = sorted(self.env._get_controller_to_bs_mapping().get(self.controller_id, []))
        pairs = []
        per_bs_users = {bs_id: [] for bs_id in bs_list}

        for bs_id in bs_list:
            bs = self.env.base_stations[bs_id]
            users = sorted(bs.connected_users.keys())
            for u in users:
                pairs.append((bs_id, u))
                per_bs_users[bs_id].append(u)

        n_pairs = len(pairs)
        if n_pairs == 0:
            return self.env.user_power_allocation

        # Pad or truncate action list to match number of users
        if len(actions) < n_pairs:
            actions = actions + [0.0] * (n_pairs - len(actions))
        else:
            actions = actions[:n_pairs]

        # --- Distribute actions per BS ---
        per_bs_actions = {bs_id: [] for bs_id in bs_list}
        for (bs_id, _u), a in zip(pairs, actions):
            per_bs_actions[bs_id].append(np.clip(a, 0.0, 1.0))

        # --- Allocate power per BS (no normalization across users) ---
        for bs_id in bs_list:
            acts = np.array(per_bs_actions.get(bs_id, []), dtype=float)
            users = per_bs_users.get(bs_id, [])
            if len(acts) == 0 or len(users) == 0:
                continue

            # Ensure values are clipped in [0, 1]
            acts = np.clip(acts, 0.0, 1.0)

            # Get BS maximum transmit power (linear scale)
            bs_pmax_dbm = getattr(self.env.base_stations[bs_id], "transmit_power_dBm", self.max_power_per_bs)
            pmax_dbm = min(self.max_power_per_bs, bs_pmax_dbm)
            pmax_linear = 10 ** (pmax_dbm / 10.0)

            # Directly scale by action value instead of normalizing
            alloc_linear = acts * pmax_linear

            # (Optional) Enforce total ≤ Pmax_BS (clip overflow)
            total_alloc = alloc_linear.sum()
            if total_alloc > pmax_linear:
                alloc_linear = alloc_linear * (pmax_linear / total_alloc)

            # Update environment with per-user power (in dBm)
            self.env.user_power_allocation.setdefault(bs_id, {})
            for u, p_lin in zip(users, alloc_linear):
                power_dbm = 10 * np.log10(p_lin) if p_lin > 0.0 else -np.inf
                self.env.user_power_allocation[bs_id][u] = power_dbm

            self.env.base_stations[bs_id].recompute_connected_users_metrics()

        # --- Log per-user power values ---
        try:
            records = []
            for bs_id, user_dict in self.env.user_power_allocation.items():
                for user_id, p_dbm in user_dict.items():
                    records.append({
                        "controller_id": self.controller_id,
                        "base_station_id": bs_id,
                        "user_id": user_id,
                        "power_dBm": p_dbm
                    })

            df = pd.DataFrame(records)
            log_file = os.path.join(self.env.output_dir, f"user_power_log_{self.controller_id}.csv")
            write_header = not os.path.exists(log_file)
            df.to_csv(log_file, mode="a", header=write_header, index=False)
        except Exception as e:
            print(f"[WARNING] Power log failed: {e}")

        return self.env.user_power_allocation

    # ----------------------- REWARD -----------------------
    def calculate_reward(self, agent_id: str):
        """
        Reward = 10 × average PDR over users of BSs managed by this controller.
        PDR is now a smooth, strictly increasing function of per-user capacity.
        """
        if agent_id not in self.env.controller_agents:
            return 0.0

        ctrl_to_bs = self.env._get_controller_to_bs_mapping()
        bs_list = ctrl_to_bs.get(agent_id, [])

        total_pdr, count = 0.0, 0
        for bs_id in bs_list:
            bs = self.env.base_stations[bs_id]
            for user_id in list(bs.connected_users.keys()):
                total_pdr += self.calculate_packet_delivery_ratio(bs_id, user_id)
                count += 1

        avg_pdr = (total_pdr / count) if count > 0 else 0.0
        return  10 * avg_pdr

    def calculate_packet_delivery_ratio(
            self,
            bs_id: int,
            user_id: int,
            alpha_d: float = 0.001,  # distance sensitivity
            alpha_n: float = 0.05,  # load sensitivity
    ) -> float:
        """
        PDR model:

            PDR = exp( - (alpha_d * distance + alpha_n * N_c) )

        - distance: UE–BS distance (meters)
        - N_c: number of users managed by the same controller

        This keeps PDR in a reasonable range without assuming a hard max distance.
        """
        bs = self.env.base_stations[bs_id]
        ue_pos = bs.connected_users[user_id].get("position")
        bs_pos = bs.get_status().get("position")

        distance = np.linalg.norm(np.array(ue_pos) - np.array(bs_pos))

        ctrl_id = self.env.base_station_assignments[bs_id]

        # All BSs handled by this controller
        bs_list = [
            bs for bs, c in self.env.base_station_assignments.items()
            if c == ctrl_id
        ]

        # Number of users managed by this controller
        N_c = len(bs.connected_users)

        tx_power_dBm = self.env.user_power_allocation.get(bs_id, {}).get(user_id, 0.0)
        tx_power_linear = 10 ** (tx_power_dBm / 10.0)
        k=0.1
        pdr = np.exp(-(alpha_d * distance + alpha_n * N_c)) * (1 - np.exp(-k * tx_power_linear))

        return pdr

    # ----------------------- DIAGNOSTIC -----------------------
    def get_average_allocated_power(self) -> float:
        """Average allocated power (dBm) across all managed users (for debugging)."""
        powers = []
        for bs_id, bs_powers in self.env.user_power_allocation.items():
            if isinstance(bs_powers, dict):
                for _uid, p in bs_powers.items():
                    if p is not None and np.isfinite(p):
                        powers.append(float(p))
        return float(np.mean(powers)) if powers else 0.0







