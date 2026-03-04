import numpy as np
import logging
from scipy.spatial.distance import cdist
from pymobility.models.mobility import (
    gauss_markov, reference_point_group, tvc,
    truncated_levy_walk, random_direction,
    random_waypoint, random_walk
)

logging.basicConfig(format='%(asctime)-15s - %(message)s', level=logging.INFO)
logger = logging.getLogger("mobility_simulation")



class UserMobilityManager:
    def __init__(self, num_users, dimensions=(5000, 5000),
                 min_v=0.1, max_v=1.0, max_wait=100.0, draw=True,
                 calculate_contacts=False, contact_range=1.0, seed=None):
        """
        Mobility Manager that runs different mobility models
        identical to the reference simulation script.
        """
        self.num_users = num_users
        self.dimensions = dimensions
        self.min_v = min_v
        self.max_v = max_v
        self.max_wait = max_wait
        self.draw = draw
        self.seed = seed
        self.calculate_contacts = calculate_contacts
        self.contact_range = contact_range
        self.model = None
        self.model_iter = None
        self.positions = None
        self.velocities = np.zeros((self.num_users, 2))
        self.timestep = 1.0


    def initialize_model(self):
        """
        Initialize the mobility model.
        Uncomment the one you want to use (exactly like the reference script).
        """

        ## Random Walk model
        #self.model = random_walk(self.num_users, dimensions=self.dimensions)

        ## Truncated Levy Walk model
        #self.model = truncated_levy_walk(self.num_users, dimensions=self.dimensions)

        ## Random Direction model
        # self.model = random_direction(self.num_users, dimensions=self.dimensions)

        ## Random Waypoint model
        #self.model = random_waypoint(
            #self.num_users, dimensions=self.dimensions,
            #velocity=(self.min_v, self.max_v), wt_max=self.max_wait
        #)

        ## Gauss-Markov model
        self.model = gauss_markov(
             self.num_users, dimensions=self.dimensions, alpha=0.99
         )

        ## Reference Point Group model
        # groups = [4 for _ in range(10)]
        # self.num_users = sum(groups)
        # self.model = reference_point_group(
        #     groups, dimensions=self.dimensions, aggregation=0.5
        # )

        ## Time-Variant Community Mobility Model
        # groups = [4 for _ in range(10)]
        # self.num_users = sum(groups)
        # self.model = tvc(
        #     groups, dimensions=self.dimensions,
        #     aggregation=[0.5, 0.], epoch=[100, 100]
        # )

        # initialize iterator
        self.model_iter = iter(self.model)
        self.positions = np.array(next(self.model_iter))  # initial position

    def update_positions(self):
        """Advance the mobility model by one step and update positions."""
        if self.model_iter is None:
            raise RuntimeError("Mobility model not initialized. Call initialize_model() first.")

        prev_positions = self.positions.copy() if self.positions is not None else None

        try:
            self.positions = np.array(next(self.model_iter))
        except StopIteration:
            # reinitialize if the generator ends
            self.initialize_model()
            self.positions = np.array(next(self.model_iter))

        # --- Compute per-user velocity vectors (vx, vy) ---
        if prev_positions is not None:
            displacement = self.positions - prev_positions
            speeds = np.linalg.norm(displacement, axis=1) / self.timestep
            # Normalize to get direction
            directions = np.divide(
                displacement,
                np.expand_dims(speeds, axis=1),
                out=np.zeros_like(displacement),
                where=speeds[:, None] != 0
            )
            self.velocities = directions * speeds[:, None]
        else:
            self.velocities = np.zeros_like(self.positions)

        return self.positions

    def get_positions(self):
        """Return current positions for all users."""
        return self.positions

    def get_velocities(self):
        """Return current velocity vectors for all users."""
        return self.velocities

    def get_speeds(self):
        """Return scalar speeds for all users (m/s or units/s)."""
        return np.linalg.norm(self.velocities, axis=1)


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from usermobility import UserMobilityManager
#
#     plt.rcParams["font.family"] = "Times New Roman"
#     plt.rcParams['mathtext.fontset'] = 'stix'
#     plt.rcParams['mathtext.rm'] = 'Times New Roman'
#     plt.rcParams['axes.titlesize'] = 16
#     plt.rcParams['axes.labelsize'] = 16
#     plt.rcParams['legend.fontsize'] = 16
#     plt.rcParams['xtick.labelsize'] = 16
#     plt.rcParams['ytick.labelsize'] = 16
#     manager = UserMobilityManager(
#         num_users=10,
#         dimensions=(1000, 1000),
#         min_v=5,
#         max_v=30,
#         max_wait=100.0,
#         draw=False,
#         calculate_contacts=False,
#         contact_range=1.0,
#         seed=None
#     )
#
#     manager.initialize_model()
#
#     # Collect position history
#     trajectories = []
#     num_steps = 500
#     for t in range(num_steps):
#         pos = manager.update_positions()
#         trajectories.append(pos.copy())
#
#     trajectories = np.array(trajectories)  # shape: (time_steps, num_users, 2)
#
#     # ---- Plot final user movement paths ----
#     plt.figure(figsize=(8.8, 6.5))
#     num_users = trajectories.shape[1]
#     for u in range(num_users):
#         plt.plot(
#             trajectories[:, u, 0],
#             trajectories[:, u, 1],
#             linestyle='--',
#             linewidth=2,
#             alpha=0.8,
#             label=f"$u_{{{u+1}}}$"
#         )
#         # Start point (green)
#         plt.scatter(
#             trajectories[0, u, 0],
#             trajectories[0, u, 1],
#             color='green',
#             s=60,
#             edgecolor='black',
#             zorder=5
#         )
#         # End point (red)
#         plt.scatter(
#             trajectories[-1, u, 0],
#             trajectories[-1, u, 1],
#             color='red',
#             s=60,
#             edgecolor='black',
#             zorder=6
#         )
#
#     plt.xlabel("X position [m]")
#     plt.ylabel("Y position [m]")
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend()
#     plt.show()
