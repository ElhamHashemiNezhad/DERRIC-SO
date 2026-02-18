import numpy as np
from scipy.spatial import Voronoi


class VoronoiDomainManager:
    """
    Manages orchestrator domains using Voronoi tessellation.
    Each orchestrator claims nodes in its Voronoi cell as part of its domain.
    """

    def __init__(self, env):
        self.env = env
        self.domains = {}  # Orchestrator ID -> domain info
        self.update_domains()

    def update_domains(self, active_orchestrators=None):
        """Update the Voronoi domains based on current orchestrator positions."""
        if active_orchestrators is None:
            active_orchestrators = {
                orch_id: orch_pos
                for orch_id, orch_pos in self.env.orchestrator_positions.items()
                if not np.array_equal(orch_pos, [-1, -1])
            }

        num_active = len(active_orchestrators)

        if num_active == 0:
            self.domains = {}
            return

        if num_active == 1:
            orch_id = next(iter(active_orchestrators))
            self.domains[orch_id] = {
                'controllers': list(range(self.env.num_controllers)),
                'base_stations': list(range(self.env.num_base_stations)),
                'users': list(range(self.env.num_users)),
                'vertices': None
            }
            return

        elif num_active == 2:
            orch_ids = list(active_orchestrators.keys())
            # Convert the dict values to a proper numpy array of positions
            orch_positions = np.array([active_orchestrators[oid] for oid in orch_ids], dtype=np.float64)

            self.domains = {
                orch_id: {'controllers': [], 'base_stations': [], 'users': [], 'vertices': None}
                for orch_id in orch_ids
            }

            controller_positions = self.env.controller_positions
            if isinstance(controller_positions, dict):
                for ctrl_id, pos in controller_positions.items():
                    pos = np.array(pos, dtype=float)
                    dist_0 = np.linalg.norm(pos - orch_positions[0])
                    dist_1 = np.linalg.norm(pos - orch_positions[1])
                    closest_orch = orch_ids[0] if dist_0 < dist_1 else orch_ids[1]
                    self.domains[closest_orch]['controllers'].append(ctrl_id)
            else:
                # Here, assume it's a list of position arrays (not names like 'ctrl_4')
                for ctrl_id, pos in enumerate(controller_positions):
                    pos = np.array(pos, dtype=float)
                    dist_0 = np.linalg.norm(pos - orch_positions[0])
                    dist_1 = np.linalg.norm(pos - orch_positions[1])
                    closest_orch = orch_ids[0] if dist_0 < dist_1 else orch_ids[1]
                    self.domains[closest_orch]['controllers'].append(ctrl_id)

            # Process base station positions
            for bs_id, bs_pos in enumerate(self.env.base_station_positions):
                dist_0 = np.linalg.norm(bs_pos - orch_positions[0])
                dist_1 = np.linalg.norm(bs_pos - orch_positions[1])
                closest_orch = orch_ids[0] if dist_0 < dist_1 else orch_ids[1]
                self.domains[closest_orch]['base_stations'].append(bs_id)

            # Process user positions
            for user_id, user_pos in enumerate(self.env.user_positions):
                dist_0 = np.linalg.norm(user_pos - orch_positions[0])
                dist_1 = np.linalg.norm(user_pos - orch_positions[1])
                closest_orch = orch_ids[0] if dist_0 < dist_1 else orch_ids[1]
                self.domains[closest_orch]['users'].append(user_id)

            return

        # For more than 2 orchestrators, use Voronoi tessellation
        points = np.array([pos for pos in active_orchestrators.values()])
        unique_points = np.unique(points, axis=0)

        if len(unique_points) < 3:
            return

        try:
            vor = Voronoi(unique_points)
            self.domains = {
                orch_id: {'controllers': [], 'base_stations': [], 'users': [], 'vertices': None}
                for orch_id in active_orchestrators
            }

            self._assign_controllers_to_domains(active_orchestrators, vor)
            self._assign_base_stations_to_domains(active_orchestrators, vor)
            self._assign_users_to_domains(active_orchestrators, vor)

            dimensions = self.env.mobility_manager.dimensions
            bounds = np.array([[0, 0], [dimensions[0], 0], [dimensions[0], dimensions[1]], [0, dimensions[1]]])
            region_to_orch = {i: orch_id for i, (orch_id, _) in enumerate(active_orchestrators.items())}
            self._calculate_domain_boundaries(vor, region_to_orch, bounds)
        except Exception as e:
            # Fall back to nearest-neighbor assignment
            for orch_id in active_orchestrators:
                self.domains[orch_id] = {
                    'controllers': [],
                    'base_stations': [],
                    'users': [],
                    'vertices': None
                }

            # Assign everything based on proximity
            self._assign_all_by_proximity(active_orchestrators)

    def _assign_all_by_proximity(self, active_orchestrators):
        """Assign all entities to their closest orchestrator"""
        # Process controller positions
        if hasattr(self.env, "controller_positions"):
            controller_positions = self.env.controller_positions

            if isinstance(controller_positions, dict):
                for ctrl_id, pos in controller_positions.items():
                    orch_id = self._find_closest_orchestrator(pos, active_orchestrators)
                    if orch_id:
                        self.domains[orch_id]['controllers'].append(ctrl_id)
            else:
                for i, pos in enumerate(controller_positions):
                    orch_id = self._find_closest_orchestrator(pos, active_orchestrators)
                    if orch_id:
                        self.domains[orch_id]['controllers'].append(i)

        # Process base station positions
        for bs_id, bs_pos in enumerate(self.env.base_station_positions):
            orch_id = self._find_closest_orchestrator(bs_pos, active_orchestrators)
            if orch_id:
                self.domains[orch_id]['base_stations'].append(bs_id)

        # Process user positions
        for user_id, user_pos in enumerate(self.env.user_positions):
            orch_id = self._find_closest_orchestrator(user_pos, active_orchestrators)
            if orch_id:
                self.domains[orch_id]['users'].append(user_id)

    def _assign_controllers_to_domains(self, active_orchestrators, vor):
        """Assign controllers to domains based on proximity to orchestrators"""
        if not hasattr(self.env, "controller_positions"):
            return

        controller_positions = self.env.controller_positions

        # Handle different data structures for controller_positions
        # Handle different data structures for controller_positions
        if isinstance(controller_positions, dict):
            for ctrl_id, pos in controller_positions.items():
                closest_orch_id = self._find_closest_orchestrator(pos, active_orchestrators)
                if closest_orch_id:
                    self.domains[closest_orch_id]['controllers'].append(ctrl_id)
                    print(f"Controller {ctrl_id} assigned to Orchestrator {closest_orch_id}")
        else:
            for i, pos in enumerate(controller_positions):
                if i >= len(controller_positions):
                    continue
                closest_orch_id = self._find_closest_orchestrator(pos, active_orchestrators)
                if closest_orch_id:
                    self.domains[closest_orch_id]['controllers'].append(i)
                    print(f"Controller {i} assigned to Orchestrator {closest_orch_id}")

    def _assign_base_stations_to_domains(self, active_orchestrators, vor):
        """Assign base stations to domains based on proximity to orchestrators"""
        for bs_idx, bs_pos in enumerate(self.env.base_station_positions):
            closest_orch_id = self._find_closest_orchestrator(bs_pos, active_orchestrators)
            if closest_orch_id:
                self.domains[closest_orch_id]['base_stations'].append(bs_idx)

    def _assign_users_to_domains(self, active_orchestrators, vor):
        """Assign users to domains based on proximity to orchestrators"""
        for user_idx, user_pos in enumerate(self.env.user_positions):
            closest_orch_id = self._find_closest_orchestrator(user_pos, active_orchestrators)
            if closest_orch_id:
                self.domains[closest_orch_id]['users'].append(user_idx)

    def _find_closest_orchestrator(self, position, active_orchestrators):
        """Find the closest orchestrator to a given position"""
        if not active_orchestrators:
            return None

        min_dist = float('inf')
        closest_orch_id = None

        for orch_id, orch_pos in active_orchestrators.items():
            dist = np.linalg.norm(position - orch_pos)
            if dist < min_dist:
                min_dist = dist
                closest_orch_id = orch_id

        return closest_orch_id

    def _calculate_domain_boundaries(self, vor, region_to_orch, bounds):
        """Calculate the domain boundaries from Voronoi regions"""
        dimensions = self.env.mobility_manager.dimensions
        bound_rect = np.array([0, 0, dimensions[0], dimensions[1]])

        # For each Voronoi region
        for i, region_idx in enumerate(vor.point_region):
            if i >= len(region_to_orch):
                continue

            orch_id = region_to_orch[i]
            region = vor.regions[region_idx]

            if -1 in region:  # Skip infinite regions for now
                # Use clipped region
                vertices = self._clip_voronoi_region_to_bounds(vor, region_idx, bound_rect)
            else:
                # Use finite region vertices
                vertices = np.array([vor.vertices[v] for v in region])

            if vertices is not None and len(vertices) > 0:
                self.domains[orch_id]['vertices'] = vertices

    def _clip_voronoi_region_to_bounds(self, vor, region_idx, bounds):
        """Clip infinite Voronoi regions to a rectangular bounding box"""
        region = vor.regions[vor.point_region[region_idx]]
        if -1 in region:
            # We need to compute a finite polygon for this infinite region
            # First, get the ridges (lines) of this region
            ridges = []
            for (p1, p2), rv in zip(vor.ridge_points, vor.ridge_vertices):
                if p1 == region_idx or p2 == region_idx:
                    ridges.append(rv)

            # Create a polygon from the finite ridges and clip to bounds
            vertices = []
            for ridge in ridges:
                if -1 not in ridge:
                    vertices.extend([vor.vertices[v] for v in ridge])

            if len(vertices) >= 3:
                # Create a convex hull of the points
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(vertices)
                    return np.array([vertices[i] for i in hull.vertices])
                except:
                    pass

        return None

    def find_domain(self, position):
        """Find which domain a position belongs to"""
        min_dist = float('inf')
        domain_orch = None

        position = np.array(position, dtype=np.float64)

        # Check each domain (orchestrator)
        for orch_id, domain in self.domains.items():
            # Get orchestrator position
            if orch_id in self.env.orchestrator_positions:
                orch_pos = np.array(self.env.orchestrator_positions[orch_id], dtype=np.float64)

                # Calculate distance
                dist = np.linalg.norm(position - orch_pos)

                if dist < min_dist:
                    min_dist = dist
                    domain_orch = orch_id

        return domain_orch

    def get_domain_controller_hosts(self, orch_id, controller_hosts):
        """Get controller hosts that are in this orchestrator's domain"""
        if orch_id not in self.domains:
            return []

        # Get all controller host positions in this domain
        domain_points = []
        for ctrl_id, ctrl_position in controller_hosts.items():
            # Find which domain this controller position belongs to
            closest_orch = self.find_domain(ctrl_position)  # Pass position, not ID

            # If this controller belongs to the specified orchestrator's domain
            if closest_orch == orch_id:
                domain_points.append({
                    'controller_id': ctrl_id,
                    'position': ctrl_position,
                    'domain': closest_orch
                })

        return domain_points

# Integration with the existing environment
def integrate_voronoi_domains_with_env(env):
    """
    Integrate Voronoi domain management with the existing environment.
    Only creates domains without modifying controller assignments.

    Args:
        env: The OrchestratorEnv instance

    Returns:
        VoronoiDomainManager: The created domain manager
    """
    # Create the domain manager
    domain_manager = VoronoiDomainManager(env)

    # Store domain manager in environment for easier access
    env.domain_manager = domain_manager

    # Add a method to update domains after environment changes
    def update_domains():
        domain_manager.update_domains()

    # Add this method to the environment for easy access
    env.update_domains = update_domains

    return domain_manager