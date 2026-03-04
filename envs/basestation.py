import csv
import os
from envs.networktopology import create_geant_topology
import networkx as nx
import numpy as np


class BaseStation:
    """
    Base Station class that models capacity using Shannon theory.
    Designed for 5G/6G-like sims with simple RB accounting and interference.
    """

    def __init__(
        self,
        env,
        station_id,
        position,
        max_users=50,
        bandwidth=200e6,     # Hz
        num_prbs=546,
        max_transmit_power=43.0, # dBm
        noise_figure=7.0,    # dB
        coverage_radius=500  # m
    ):
        self.env = env
        self.station_id = int(station_id)
        self.position = np.asarray(position, dtype=float)

        # Radio/resource config
        self.bandwidth = float(bandwidth)
        self.num_prbs = int(num_prbs)
        self.transmit_power_dBm = float(max_transmit_power)
        self.noise_figure = float(noise_figure)
        self.coverage_radius = float(coverage_radius)
        self.max_users = int(max_users)

        # 6G-ish constants
        self.modulation_efficiency = 0.9
        self.mimo_gain_dB = 10.0
        self.thermal_noise_density = -174.0  # dBm/Hz
        self.frequency = 7.2e9               # Hz

        # Connections & resources
        self.connected_users = {}
        self.available_resource_blocks = 546
        self.allocated_resource_blocks = 0

        # Control-plane / orchestration
        self.active = True
        self.controller_id = None
        self.nearest_controller_distance = float("inf")
        self.nearest_orchestrator_distance = float("inf")


        # Interference (list of dicts: {bs_id, position, transmit_power_dBm})
        self.interference_sources = []

    def assign_to_controller(self, controller_id):
        self.controller_id = controller_id
        return True

    # ---------- User attach/detach ----------
    def connect_user(self, user_id, position, velocity):
        """
        Connect (or update) a user if coverage & RBs allow.
        Idempotent: re-connecting same user updates stats without double-counting RBs.
        """
        position = np.asarray(position, dtype=float)
        distance = float(np.linalg.norm(position - self.position))

        # Capacity guard (only for *new* users)
        if user_id not in self.connected_users and len(self.connected_users) >= self.max_users:
            return False

        # Link budget & capacity
        sinr_dB = float(self.calculate_sinr_dB(position))
        user_capacity = float(self.calculate_user_capacity_shannon(sinr_dB))

        # Required RBs (≥ 1), ensure int
        req_rbs = int(np.ceil(max(1.0, self.calculate_required_resource_blocks(user_capacity))))

        prev = self.connected_users.get(user_id)
        if prev is not None:
            # Update without double-counting
            prev_rbs = int(prev.get("allocated_rbs", 0))
            delta = req_rbs - prev_rbs
            if delta > 0 and self.allocated_resource_blocks + delta > self.available_resource_blocks:
                return False
            if delta > 0:
                self.allocated_resource_blocks += delta
        else:
            if self.allocated_resource_blocks + req_rbs > self.available_resource_blocks:
                return False
            self.allocated_resource_blocks += req_rbs

        latency_ms = float(self.env.user_latency_map.get(user_id, float("inf")))
        latency_ms = np.clip(latency_ms, 0.0, 200.0)

        self.connected_users[user_id] = {
            "position": position,
            "distance": distance,
            "velocity": velocity,
            "sinr": sinr_dB,
            "capacity": user_capacity,
            "allocated_rbs": req_rbs,
            "latency_ms": latency_ms,
        }
        # --- CSV Logging ---
        out_dir = getattr(self.env, "output_dir", "./outputs")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"bs_{self.station_id}_users.csv")

        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "station_id", "user_id",
                "distance_m", "velocity_mps",
                "sinr_dB", "capacity_Mbps", "allocated_rbs", "latency_ms"
            ])
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "station_id": self.station_id,
                "user_id": user_id,
                "distance_m": distance,
                "velocity_mps": velocity,
                "sinr_dB": sinr_dB,
                "capacity_Mbps": user_capacity / 1e6,
                "allocated_rbs": req_rbs,
                "latency_ms": latency_ms,
            })

        return True

    # ---------- RF / capacity ----------

    def calculate_path_loss_dB(self, distance):
        """Free-space path loss (dB)."""
        distance = max(1.0, float(distance))  # avoid log(0)
        c = 3e8
        fspl_dB = 20 * np.log10(distance) + 20 * np.log10(self.frequency) + 20 * np.log10(4 * np.pi / c)
        return float(fspl_dB)

    def calculate_noise_power_dBm(self):
        """Thermal noise power (dBm) over self.bandwidth + NF."""
        return float(self.thermal_noise_density + 10 * np.log10(self.bandwidth) + self.noise_figure)

    def calculate_received_power_dBm(self, distance):
        """Rx power in dBm from Tx power and path loss."""
        pl = self.calculate_path_loss_dB(distance)
        return float(self.transmit_power_dBm - pl)

    def calculate_interference_power_dBm(self, user_position):
        """Aggregate interference (dBm) from other BSs within ~2x coverage radius."""
        total_mW = 0.0
        upos = np.asarray(user_position, dtype=float)

        for src in self.interference_sources:
            if src.get("bs_id") == self.station_id:
                continue
            sp = np.asarray(src.get("position"), dtype=float)
            d = float(np.linalg.norm(upos - sp))
            if d > 2.0 * self.coverage_radius:
                continue
            pl = self.calculate_path_loss_dB(d)
            ip_dBm = float(src.get("transmit_power_dBm", -np.inf)) - pl
            ip_dBm = min(ip_dBm, -60.0)  # cap
            total_mW += 10 ** (ip_dBm / 10.0)

        return float(10 * np.log10(total_mW)) if total_mW > 0 else -np.inf

    def _user_tx_power_dBm(self, user_id, position):
        """Per-user TX power override from env.user_power_allocation; fallback to BS power."""
        pwr = None
        if self.station_id in self.env.user_power_allocation:
            pwr = self.env.user_power_allocation[self.station_id].get(user_id, None)
        return float(pwr) if (pwr is not None and np.isfinite(pwr)) else float(self.transmit_power_dBm)

    def calculate_sinr_dB(self, user_position, user_id=None):
        """Use per-user TX power if user_id given; else fallback to BS power."""
        distance = np.linalg.norm(user_position - self.position)
        tx_dBm = self._user_tx_power_dBm(user_id, user_position) if user_id is not None else self.transmit_power_dBm
        path_loss_dB = self.calculate_path_loss_dB(distance)
        signal_power_dBm = tx_dBm - path_loss_dB

        noise_power_dBm = self.calculate_noise_power_dBm()
        interference_power_dBm = self.calculate_interference_power_dBm(user_position)

        signal_mW = 10 ** (signal_power_dBm / 10)
        noise_mW = 10 ** (noise_power_dBm / 10)
        interf_mW = 0 if interference_power_dBm == -np.inf else 10 ** (interference_power_dBm / 10)

        sinr_lin = signal_mW / (noise_mW + interf_mW + 1e-12)
        return 10 * np.log10(sinr_lin + 1e-12)

    def recompute_connected_users_metrics(self):
        """Recompute each connected user's SINR/capacity/RBs after power changes."""
        self.allocated_resource_blocks = 0
        for user_id, u in list(self.connected_users.items()):
            pos = np.asarray(u['position'], dtype=float)
            sinr_dB = self.calculate_sinr_dB(pos, user_id=user_id)
            capacity = self.calculate_user_capacity_shannon(sinr_dB)
            rbs = int(np.ceil(max(1.0, float(self.calculate_required_resource_blocks(capacity)))))
            self.connected_users[user_id].update({
                'sinr': float(sinr_dB),
                'capacity': float(capacity),
                'allocated_rbs': rbs,
            })
            self.allocated_resource_blocks += rbs

    def calculate_user_capacity_shannon(self, sinr_dB, allocated_bandwidth=None):
        """
        Shannon capacity (bps) with modulation & orchestration efficiency factors.
        """
        sinr_linear = 10 ** (sinr_dB / 10.0)
        sinr_linear = float(np.clip(sinr_linear, 1e-10, 1e12))

        if allocated_bandwidth is None:
            n = max(1, len(self.connected_users))
            allocated_bandwidth = self.bandwidth / n

        cap = float(allocated_bandwidth * np.log2(1.0 + sinr_linear))
        cap *= self.modulation_efficiency

        return cap

    def calculate_required_resource_blocks(self, capacity_bps):
        rb_capacity = 5e6  # 5 Mbps per RB
        if capacity_bps <= 0:
            return 1.0
        return min(capacity_bps / rb_capacity, float(self.available_resource_blocks))

    def calculate_total_capacity(self):
        """Sum of per-user capacities (bps)."""
        if not self.connected_users:
            return 0.0
        return float(sum(u["capacity"] for u in self.connected_users.values()))

    def calculate_per_user_capacity(self, user_id=None):
        """Capacity for specific user or average per user (bps)."""
        if not self.connected_users:
            return 0.0
        if user_id is not None and user_id in self.connected_users:
            return float(self.connected_users[user_id]["capacity"])
        return float(self.calculate_total_capacity() / len(self.connected_users))

    # ---------- Interference bookkeeping ----------
    def add_interference_source(self, bs_id, position, transmit_power_dBm):
        """Register another BS as an interference source (with range limit)."""
        if bs_id == self.station_id:
            return
        pos = np.asarray(position, dtype=float)
        if float(np.linalg.norm(pos - self.position)) > 2.0 * self.coverage_radius:
            return
        self.interference_sources.append({
            "bs_id": bs_id,
            "position": pos,
            "transmit_power_dBm": float(transmit_power_dBm),
        })

    # ---------- Status helpers ----------
    def enforce_throughput_cap(self, global_target_mbps):
        """
        Cap per-user throughput if this BS is far above target (very rough fairness tool).
        Scales allocated_rbs and capacity consistently.
        """
        if not self.connected_users:
            return
        avg_mbps = self.calculate_per_user_capacity() / 1e6
        if avg_mbps <= 1.2 * global_target_mbps:
            return

        cap_factor = float(global_target_mbps / max(1e-9, avg_mbps))
        for u in self.connected_users.values():
            u["allocated_rbs"] = int(max(1, np.floor(u["allocated_rbs"] * cap_factor)))
            u["capacity"] *= cap_factor

        self.allocated_resource_blocks = int(sum(u["allocated_rbs"] for u in self.connected_users.values()))

    def get_status(self):
        util = (
                    self.allocated_resource_blocks / self.available_resource_blocks) if self.available_resource_blocks > 0 else 0.0
        return {
            "station_id": self.station_id,
            "position": self.position.tolist(),
            "active": self.active,
            "controller_id": self.controller_id,
            "connected_users": len(self.connected_users),
            "capacity_mbps": self.calculate_total_capacity() / 1e6,
            "capacity_per_user_mbps": self.calculate_per_user_capacity() / 1e6,
            "resource_utilization": util,
            "max_users": self.max_users,
            "orchestration_efficiency": self.env._calculate_orchestration_efficiency,
        }

    def get_user_info(self, user_id):
        """Return a copy of the tracked user record, with capacity in Mbps added."""
        u = self.connected_users.get(user_id)
        if not u:
            return None
        info = dict(u)
        info["capacity_mbps"] = info["capacity"] / 1e6
        return info
