"""
Contain the functions to take the igraph brainconn
function names for weigthed/binary graphs
"""
import igraph as ig
import brainconn as bc

from ..exceptions import invalid_graph
from .wrappers import (_coreness_bc,
                       _dijkstra_bc,
                       _louvain_ig,
                       _local_efficiency_bin_ig,
                       _local_efficiency_wei_ig)

###########################################################
# Brainconn method names
###########################################################


def _get_func(backend, metric, is_weighted):
    funcs = {}

    funcs["igraph"] = {}
    funcs["brainconn"] = {}

    # Clustering
    funcs["igraph"]["clustering"] = {
        False: ig.Graph.transitivity_local_undirected,
        True: ig.Graph.transitivity_local_undirected,
    }

    funcs["brainconn"]["clustering"] = {
        False: bc.clustering.clustering_coef_bu,
        True: bc.clustering.clustering_coef_wu
    }

    # Coreness
    funcs["igraph"]["coreness"] = {
        False: ig.Graph.coreness,
        True: invalid_graph,
    }

    funcs["brainconn"]["coreness"] = {
        False: _coreness_bc,
        True: _coreness_bc
    }

    # Distances
    funcs["igraph"]["shortest_path"] = {
        False: ig.Graph.shortest_paths_dijkstra,
        True: ig.Graph.shortest_paths_dijkstra,
    }

    funcs["brainconn"]["shortest_path"] = {
        False: _dijkstra_bc,
        True: _dijkstra_bc
    }

    # Betweenness
    funcs["igraph"]["betweenness"] = {
        False: ig.Graph.betweenness,
        True: ig.Graph.betweenness,
    }

    funcs["brainconn"]["betweenness"] = {
        False: bc.centrality.betweenness_bin,
        True: bc.centrality.betweenness_wei
    }

    # Modularity
    funcs["igraph"]["modularity"] = {
        False: _louvain_ig,
        True: _louvain_ig,
    }

    funcs["brainconn"]["modularity"] = {
        False: bc.modularity.modularity_louvain_und,
        True: bc.modularity.modularity_louvain_und
    }

    # Efficiency
    funcs["igraph"]["efficiency"] = {
        False: _local_efficiency_bin_ig,
        True: _local_efficiency_wei_ig
    }

    funcs["brainconn"]["efficiency"] = {
        False: bc.distance.efficiency_bin,
        True: bc.distance.efficiency_wei
    }

    return funcs[backend][metric][is_weighted]
