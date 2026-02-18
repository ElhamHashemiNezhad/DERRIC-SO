import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from .usermobility import UserMobilityManager


np.random.seed(42)


def create_geant_topology(num_hosts=34, num_users=300):
    """Create the GEANT network topology with mobile users."""
    G = nx.Graph()

    # Define node capacities for different types
    NODE_PROPERTIES = {
        'orchestrator': {
            'processing_capacity': 1000,
            'memory_gb': 256,
            'storage_tb': 100
        },
        'controller': {
            'processing_capacity': 500,
            'memory_gb': 128,
            'storage_tb': 50
        },
        'base_station': {
            'processing_capacity': 200,
            'memory_gb': 64,
            'storage_tb': 20
        },
        'host': {
            'processing_capacity': 100,
            'memory_gb': 16,
            'storage_tb': 10
        }
    }

    # Define link properties
    LINK_PROPERTIES = {
        'backbone': {
            'latency_ms': (2, 5),
            'capacity_gbps': 100
        },
        'regional': {
            'latency_ms': (5, 10),
            'capacity_gbps': 50
        },
        'access': {
            'latency_ms': (10, 15),
            'capacity_gbps': 20
        },
        'ran': {
            'latency_ms': (20, 30),
            'capacity_gbps': 1
        }
    }

    # Add infrastructure nodes (0-33)
    for i in range(num_hosts):
        if i in [0, 5, 6, 9, 10, 11, 25]:
            node_type = 'orchestrator'
        elif i in [2, 3, 7, 12, 14, 15, 27]:
            node_type = 'controller'
        elif i in [3, 4, 7, 8, 13, 17, 18, 19, 22, 24, 28, 29, 30, 31, 33]:
            node_type = 'base_station'
        else:
            node_type = 'host'

        G.add_node(i,
                   node_type=node_type,
                   **NODE_PROPERTIES[node_type])

    # Add infrastructure edges
    backbone_paths = [
        (0, 1), (0, 2), (0, 4), (6, 5), (5, 9),
        (5, 25), (10, 15), (25, 4), (6, 25), (2, 11), (11, 13),
        (5, 11), (11, 15), (10, 33), (10, 32), (32, 17), (17, 12), (12, 33), (9, 33)
    ]
    for n1, n2 in backbone_paths:
        G.add_edge(n1, n2,
                   capacity_gbps=LINK_PROPERTIES['backbone']['capacity_gbps'],
                   latency_ms=np.random.uniform(*LINK_PROPERTIES['backbone']['latency_ms']),
                   link_type='backbone')

    regional_paths = [
        (2, 13), (6, 30), (31, 9), (10, 20),
        (13, 16), (28, 31), (29, 5), (15, 18), (5, 10), (6, 26),
        (20, 21), (26, 27), (6, 3), (6, 29),
        (27, 28), (5, 31), (15, 13), (3, 30)
    ]
    for n1, n2 in regional_paths:
        G.add_edge(n1, n2,
                   capacity_gbps=LINK_PROPERTIES['regional']['capacity_gbps'],
                   latency_ms=np.random.uniform(*LINK_PROPERTIES['regional']['latency_ms']),
                   link_type='regional')

    access_paths = [
        (6, 24), (8, 30), (14, 19), (20, 22), (0, 16), (2, 7), (5, 7),
        (12, 14), (15, 23), (12, 19)
    ]
    for n1, n2 in access_paths:
        G.add_edge(n1, n2,
                   capacity_gbps=LINK_PROPERTIES['access']['capacity_gbps'],
                   latency_ms=np.random.uniform(*LINK_PROPERTIES['access']['latency_ms']),
                   link_type='access')
    pos = {}
    user_start_index = num_hosts
    for i in range(num_users):
        user_id = user_start_index + i
        G.add_node(user_id, node_type='user', processing_capacity=0, memory_gb=0, storage_tb=0)
        pos[user_id] = (random.uniform(-10, 10), random.uniform(-10, 10))

        # Connect each user to a random base station via a 'ran' link
        base_stations = [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'base_station']
        bs = np.random.choice(base_stations)
        G.add_edge(user_id, bs,
                   capacity_gbps=LINK_PROPERTIES['ran']['capacity_gbps'],
                   latency_ms=np.random.uniform(*LINK_PROPERTIES['ran']['latency_ms']),
                   link_type='ran')


    return G, LINK_PROPERTIES


def get_geant_positions():
    """Returns geographical positions for GÉANT2 nodes."""
    positions = {
        0: (1.7, 5.1),  # UK
        1: (0.9, 5.7),  # IE
        2: (1.7, 3.9),  # FR
        3: (2.3, 7.8),  # NO
        4: (2.3, 4.8),  # BE
        5: (3.0, 4.3),  # DE
        6: (2.7, 6.4),  # DK
        7: (2.3, 4.3),  # LU
        8: (4.6, 8.4),  # FI
        9: (3.6, 4.3),  # CZ
        10: (3.6, 3.9),  # AT
        11: (2.5, 3.6),  # CH
        12: (4.3, 3.6),  # HU
        13: (1.5, 2.3),  # ES
        14: (4.9, 3.9),  # RO
        15: (2.8, 3.1),  # IT
        16: (0.9, 2.3),  # PT
        17: (3.9, 3.1),  # HR
        18: (3.2, 1.8),  # MT
        19: (4.9, 3.2),  # BG
        20: (4.0, 2.1),  # GR
        21: (5.0, 2.1),  # TR
        22: (5.3, 1.5),  # CY
        23: (5.7, 1.3),  # IL
        24: (-0.5, 9.2),  # IS
        25: (2.3, 5.3),  # NL
        26: (4.4, 7.3),  # EE
        27: (4.4, 6.7),  # LV
        28: (4.4, 6.2),  # LT
        29: (5.2, 6.4),  # RU
        30: (3.1, 7.5),  # SE
        31: (3.9, 5.5),  # PL
        32: (3.5, 3.1),  # SI
        33: (4.1, 4.1),  # SK
    }
    return positions


def map_users_to_base_stations(mobility_manager, base_station_positions, max_range=0.8):
    """
    Map users to their nearest base stations based on geographic proximity.

    Args:
        mobility_manager: UserMobilityManager instance
        base_station_positions: Dict of base station positions {node_id: (x, y)}
        max_range: Maximum connection range for base stations

    Returns:
        user_connections: Dict {user_id: connected_base_station_id}
    """
    user_connections = {}

    # Scale user positions to match the GEANT coordinate system (0-7 range)
    mobility_manager.initialize_model()
    mobility_manager.update_positions()  # generate at least one position update
    user_positions = np.array(mobility_manager.get_positions())
    # Scale user positions to match the GEANT coordinate system (0-7 range)
    scale_x = 7.0 / mobility_manager.dimensions[0]
    scale_y = 9.0 / mobility_manager.dimensions[1]

    scaled_user_positions = user_positions * [scale_x, scale_y]

    for user_id, user_pos in enumerate(scaled_user_positions):
        min_distance = float('inf')
        closest_bs = None

        for bs_id, bs_pos in base_station_positions.items():
            distance = np.linalg.norm(np.array(user_pos) - np.array(bs_pos))

            if distance < min_distance and distance <= max_range:
                min_distance = distance
                closest_bs = bs_id

        # If no base station is in range, connect to the closest one anyway
        if closest_bs is None:
            for bs_id, bs_pos in base_station_positions.items():
                distance = np.linalg.norm(np.array(user_pos) - np.array(bs_pos))
                if distance < min_distance:
                    min_distance = distance
                    closest_bs = bs_id

        user_connections[user_id] = closest_bs

    return user_connections, scaled_user_positions


def visualize_network_with_mobile_users(num_users=300, show_animation=False):
    """
    Visualize the GEANT topology with mobile users.
    """
    # Create network and mobility manager
    G, link_props = create_geant_topology(num_users=num_users)
    mobility_manager = UserMobilityManager(num_users=num_users, dimensions=(1000, 1000))

    # Get infrastructure positions
    pos = get_geant_positions()

    # Get base station nodes and their positions
    base_station_nodes = [i for i, data in G.nodes(data=True)
                          if data.get('node_type') == 'base_station' and i in pos]
    base_station_positions = {bs_id: pos[bs_id] for bs_id in base_station_nodes}


    return create_static_visualization(G, pos, mobility_manager, base_station_positions)


def create_static_visualization(G, pos, mobility_manager, base_station_positions):
    """Create a static visualization of the network with current user positions."""

    # Map users to base stations
    user_connections, user_positions = map_users_to_base_stations(
        mobility_manager, base_station_positions
    )

    user_node_offset = max(G.nodes) + 1  # start user node IDs after last infrastructure node
    for user_id, bs_id in user_connections.items():
        user_node_id = user_node_offset + user_id
        G.add_node(user_node_id,
                   node_type='user',
                   pos=tuple(user_positions[user_id]),
                   processing_capacity=1,
                   memory_gb=1,
                   storage_tb=0.01)
        G.add_edge(user_node_id, bs_id,
                   latency_ms=np.random.uniform(20, 30),
                   capacity_gbps=1,
                   link_type='ran')
        # Add user position to pos dictionary for visualization
        pos[user_node_id] = tuple(user_positions[user_id])

    # Set up the figure
    plt.figure(figsize=(17, 13))
    bg_img = mpimg.imread("europe_map.png")  # <-- replace with your image filename

    # Show the image as background
    ax = plt.gca()
    ax.imshow(bg_img, extent=[-1, 6, 1, 10], aspect='auto', zorder=0)

    # Define node types
    orchestrator_nodes = [i for i, data in G.nodes(data=True)
                          if data.get('node_type') == 'orchestrator' and i in pos]
    controller_nodes = [i for i, data in G.nodes(data=True)
                        if data.get('node_type') == 'controller' and i in pos]
    base_station_nodes = [i for i, data in G.nodes(data=True)
                          if data.get('node_type') == 'base_station' and i in pos]
    host_nodes = [i for i, data in G.nodes(data=True)
                  if data.get('node_type') == 'host' and i in pos]
    user_nodes = [i for i, data in G.nodes(data=True) if data.get('node_type') == 'user']

    # Draw infrastructure edges
    backbone_edges = [(u, v) for u, v, data in G.edges(data=True)
                      if data.get('link_type') == 'backbone' and u in pos and v in pos]
    regional_edges = [(u, v) for u, v, data in G.edges(data=True)
                      if data.get('link_type') == 'regional' and u in pos and v in pos]
    access_edges = [(u, v) for u, v, data in G.edges(data=True)
                    if data.get('link_type') == 'access' and u in pos and v in pos]
    ran_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('link_type') == 'ran']

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=backbone_edges, width=4.0,
                           edge_color='black', alpha=0.8, label='Backbone')
    nx.draw_networkx_edges(G, pos, edgelist=regional_edges, width=2.5,
                           edge_color='purple', alpha=0.8, label='Regional')
    nx.draw_networkx_edges(G, pos, edgelist=access_edges, width=1.5,
                           edge_color='#48cae4', alpha=0.8, label='Access')


    # Draw infrastructure nodes
    nx.draw_networkx_nodes(G, pos, nodelist=orchestrator_nodes,
                           node_color='#0066CC', node_size=1200, label='Orchestrator')
    nx.draw_networkx_nodes(G, pos, nodelist=controller_nodes,
                           node_color='#FF9900', node_size=1200, label='Controller')
    nx.draw_networkx_nodes(G, pos, nodelist=base_station_nodes,
                           node_color='#669900', node_size=1000,
                           node_shape='^', label='Base Station')
    nx.draw_networkx_nodes(G, pos, nodelist=host_nodes,
                           node_color='#CCCCCC', node_size=1200, label='Host')


    # Draw users
    #plt.scatter(user_positions[:, 0], user_positions[:, 1],
                #c='red', s=50, alpha=0.7, label='Mobile Users')

    # Draw connections from users to base stations
    #for user_id, bs_id in user_connections.items():
        #if bs_id is not None:
            #user_pos = user_positions[user_id]
            #bs_pos = pos[bs_id]
            #plt.plot([user_pos[0], bs_pos[0]], [user_pos[1], bs_pos[1]],
                     #'r--', alpha=0.3, linewidth=0.5)

    # Add country labels
    country_names = {
        0: "UK", 1: "IE", 2: "FR", 3: "NO", 4: "BE", 5: "DE",
        6: "DK", 7: "LU", 8: "FI", 9: "CZ", 10: "AT", 11: "CH",
        12: "HU", 13: "ES", 14: "RO", 15: "IT", 16: "PT", 17: "HR",
        18: "MT", 19: "BG", 20: "GR", 21: "TR", 22: "CY", 23: "IL",
        24: "IS", 25: "NL", 26: "EE", 27: "LV", 28: "LT", 29: "RU",
        30: "SE", 31: "PL", 32: "SI", 33: "SK"
    }

    for node, (x, y) in pos.items():
        if node in country_names:
            plt.text(x, y, f"{node}\n{country_names[node]}",
                     fontsize=8, ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))



    # Statistics
    connected_users = sum(1 for bs in user_connections.values() if bs is not None)
    stats = (
        f"Network with Mobile Users:\n"
        f"Orchestrator Nodes: {len(orchestrator_nodes)}\n"
        f"Controller Nodes: {len(controller_nodes)}\n"
        f"Mobile Users: {len(user_positions)}\n"
        f"Base Stations: {len(base_station_nodes)}\n"

    )

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    # --- Custom legend handles ---
    backbone_line = mlines.Line2D([], [], color='black', linewidth=3, label='Backbone (100 Gbps)')
    regional_line = mlines.Line2D([], [], color='purple', linewidth=2, label='Regional (50 Gbps)')
    access_line = mlines.Line2D([], [], color='#48cae4', linewidth=1.5, label='Access (20 Gbps)')

    orch_patch = mpatches.Circle((0, 0), radius=0.2, color='#0066CC', label='Orchestrator')
    ctrl_patch = mpatches.Circle((0, 0), radius=0.2, color='#FF9900', label='Controller')
    bs_patch = mpatches.RegularPolygon((0, 0), numVertices=3, radius=0.25, color='#669900', label='Base Station')
    host_patch = mpatches.Circle((0, 0), radius=0.2, color='#CCCCCC', label='Host')

    handles = [backbone_line, regional_line, access_line,
               orch_patch, ctrl_patch, bs_patch, host_patch]
    ax = plt.gca()
    # --- Create custom legend ---
    ax.legend(handles=handles,
              loc='lower left',
              bbox_to_anchor=(0, 0.06),
              fontsize=14,
              frameon=True,
              fancybox=True,
              framealpha=0.9,
              handlelength=2.5,
              markerscale=0.6,
              title_fontsize=10)

    #plt.figtext(0.87, 0.02, stats, fontsize=11,
                #bbox=dict(facecolor='white', alpha=0.8))
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("geant_topology_with_users.png", dpi=300, bbox_inches='tight')
    plt.show()

    return G, mobility_manager, user_connections



# Main execution
#if __name__ == "__main__":
    #np.random.seed(42)

    #print("Creating network visualization with mobile users...")

    #Static visualization
    #G, mobility_manager, connections = visualize_network_with_mobile_users(
        #num_users=100, show_animation=False
    #)

    #print(f"Network created with {len(connections)} mobile users")
