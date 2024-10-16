#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

# LIBRARY IMPORTS
import os
import sys
import textwrap
import argparse
import statistics
import random as rd
from pathlib import Path
from typing import Iterator, Dict, List

from loguru import logger
import matplotlib
import matplotlib.pyplot as plt
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw_networkx_nodes,
    draw_networkx_edges,
)
rd.seed(9001) # For reproducibility
matplotlib.use("Agg") # For headless mode


# METADATAS
__author__ = "Bounsay Helene"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Bounsay Helene"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Bounsay Helene"
__email__ = "helene.bounsay@etu.u-paris.Fr"
__status__ = ""


# FUNCTIONS
def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, 'r', encoding= "utf-8") as f:
        while True:
            f.readline() 
            sequence = f.readline().strip()
            f.readline()  
            f.readline()  # ignore the quality line
            if not sequence:
                break
            yield sequence


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(len(read) - kmer_size + 1):
        yield read[i : i + kmer_size] # return a kmer of size kmer_size


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    logger.info(f"Building kmer dictionnary from {fastq_file}...")
    kmer_dict = {}
    #  read the fastq file and cut the reads into kmers
    for read in read_fastq(fastq_file):
        # count the number of occurrences of each kmer
        for kmer in cut_kmer(read, kmer_size):
            # kmer already seen at least once
            if kmer in kmer_dict:
                kmer_dict[kmer] += 1
            else:
                kmer_dict[kmer] = 1

    logger.success(f"{len(kmer_dict)} kmers found successfully! \n")
    return kmer_dict


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    logger.info("Building the de Bruijn graph...")
    # create a directed graph
    graph = DiGraph()

    # add the edges between the kmers
    for kmer, weight in kmer_dict.items():
        # add the first k-1 prefix of the kmer
        prefix = kmer[:-1]
        # add the last k-1 suffix of the kmer
        suffix = kmer[1:]
        # add the edge between the prefix and the suffix
        # with the weight that is the number of occurrences of the kmer
        graph.add_edge(prefix, suffix, weight=weight)

    logger.debug(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    logger.success("Graph built successfully! \n")

    return graph


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        if delete_entry_node and delete_sink_node:
            # Remove all nodes in the path
            graph.remove_nodes_from(path)
        elif delete_entry_node:
            # Remove all nodes except the last one
            graph.remove_nodes_from(path[:-1])
        elif delete_sink_node:
            # Remove all nodes except the first one
            graph.remove_nodes_from(path[1:])
        else:
            # Remove all nodes except the first and last one
            graph.remove_nodes_from(path[1:-1])

    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    if len(path_list) == 1:
        # If there's only one path, return the original graph
        return graph.copy()

    # Calculate standard deviation of average weights and path lengths
    weight_stddev = statistics.stdev(weight_avg_list)
    length_stddev = statistics.stdev(path_length)

    if weight_stddev > 0:
        # If weights vary, select the path with the highest weight
        logger.debug("The weights of the paths vary. Selecting the path with the highest weight.")
        best_path_index = weight_avg_list.index(max(weight_avg_list))
    elif length_stddev > 0:
        # If weights are equal but lengths vary, select the longest path
        logger.debug("The weights of the paths are equal. Selecting the longest path.")
        best_path_index = path_length.index(max(path_length))
    else:
        # If weights and lengths are equal, choose randomly
        logger.debug("The weights and lengths of the paths are equal. Selecting a path randomly.")
        best_path_index = rd.randint(0, len(path_list) - 1)

    # Remove other paths from the graph based on the best path index
    for i, path in enumerate(path_list):
        if i != best_path_index:
            modified_graph = remove_paths(graph, [path], delete_entry_node, delete_sink_node)

    return modified_graph


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    logger.info(f"Solving bubble issue between {ancestor_node} and {descendant_node}...")
    all_paths = list(all_simple_paths(graph, ancestor_node, descendant_node))
    all_weigths = [path_average_weight(graph, path) for path in all_paths]
    all_length = [len(path)-1 for path in all_paths]
    graph = select_best_path(graph, all_paths, all_length, all_weigths)

    logger.debug(f"There are {len(all_paths)} paths between {ancestor_node} and {descendant_node}.")
    logger.success(f"Bubble issue between {ancestor_node} and {descendant_node} solved! \n")

    return graph


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    logger.info("Detecting and exploding bubbles... \n")
    bubble_detected = False
    bubble_descriptions = []
    count = 0

    for node in graph:
        predecessors = list(graph.predecessors(node))
        if len(predecessors) > 1:
            for i, node_i in enumerate(predecessors):
                for node_j in predecessors[i + 1:]:
                    common_ancestor = lowest_common_ancestor(graph, node_i, node_j)
                    if common_ancestor is not None:
                        bubble_detected = True
                        bubble_descriptions.append((common_ancestor, node))
                        break
                if bubble_detected:
                    break

    if bubble_detected:
        for ancestor, descendant in bubble_descriptions:
            count += 1
            logger.debug(f"Bubble {count} detected :")
            graph = solve_bubble(graph, ancestor, descendant)

    logger.success(f"{count} bubble(s) exploded successfully!\n")
    return graph


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    logger.info("Removing entry tips...")
    while True:
        entry_tips = [node for node in graph if len(list(graph.predecessors(node))) > 1]

        if not entry_tips:
            break

        for node in entry_tips:
            # Find all paths from starting nodes to the entry tip
            paths = [list(all_simple_paths(graph, node_start_i, node)) for node_start_i in starting_nodes]
            paths = [path[0] for path in paths if len(path) > 0] # Remove empty paths
            lengths = [len(path) - 1 for path in paths]
            # Compute the average weight of the path if the path length is greater than 1
            weights = [path_average_weight(graph, path) if lengths[i] > 1 else graph.get_edge_data(*path)["weight"]
                       for i, path in enumerate(paths)]

            # Remove all paths except the best one
            graph = select_best_path(graph, paths, lengths, weights,
                                     delete_entry_node=True, delete_sink_node=False)


    logger.success("Entry tips removed successfully! \n")
    return graph


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    logger.info("Removing out tips...")
    while True:
        found_tip = False
        for node in graph:
            node_success = list(graph.successors(node))
            if len(node_success) > 1:
                paths = [list(all_simple_paths(graph, node, node_end_i))\
                         for node_end_i in ending_nodes]
                paths = [path[0] for path in paths if len(path) > 0]
                lengths = [len(path) - 1 for path in paths]
                weights = [path_average_weight(graph, path) if lengths[i] > 1 else \
                           graph.get_edge_data(*path)["weight"]
                           for i, path in enumerate(paths)]

                graph = select_best_path(graph, paths, lengths, weights,
                                         delete_entry_node=False, delete_sink_node=True)
                found_tip = True
                break

        if not found_tip:
            break
        
    logger.success("Out tips removed successfully! \n")
    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    starting_nodes = [node for node in graph.nodes() if len(list(graph.predecessors(node))) == 0]
    return starting_nodes


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    sink_nodes = [node for node in graph.nodes() if len(list(graph.successors(node))) == 0]
    return sink_nodes


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    logger.info("Extracting contigs...")
    contigs = []
    for start_node in starting_nodes:
        for end_node in ending_nodes:
            if start_node in graph and end_node in graph:
                if has_path(graph, start_node, end_node):
                    for path in all_simple_paths(graph, start_node, end_node):
                        contig = path[0]  # Start with the first kmer
                        for node in path[1:]:
                            contig += node[-1]  # Add the last character of the kmer
                        contigs.append((contig, len(contig)))

    logger.success(f"{len(contigs)} contig(s) extracted successfully! \n")
    return contigs


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    logger.info(f"Saving contigs to {output_file}...")
    # Create the output file if it does not exist
    os.makedirs(output_file.parent, exist_ok=True)
    with open(output_file, "w", encoding=" utf-8") as file:
        for i, (contig, length) in enumerate(contigs_list):
            file.write(f">contig_{i} len={length}\n")
            file.write(textwrap.fill(contig, width=80)) # Wrap the sequence to 80 characters per line : fasta format
            file.write("\n")

    logger.success("Contigs saved successfully! \n")


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    _fig, _ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = random_layout(graph)
    draw_networkx_nodes(graph, pos, node_size=6)
    draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Get kmers from the fastq file
    kmers_dict = build_kmer_dict(args.fastq_file, args.kmer_size)

    # Build the graph from the kmers
    graph = build_graph(kmers_dict)

    # Resolve bubbles in the graph
    graph = simplify_bubbles(graph)

    # Get starting and ending nodes
    starting_nodes = get_starting_nodes(graph)
    ending_nodes = get_sink_nodes(graph)

    # Resolve entry and out tips
    graph = solve_entry_tips(graph, starting_nodes)
    graph = solve_out_tips(graph, ending_nodes)

    # Get contigs from the graph
    contigs = get_contigs(graph, starting_nodes, ending_nodes)

    # Write the contigs to a file
    output_file = args.output_file if args.output_file else "contigs.fa"
    save_contigs(contigs, output_file)

    # Plot the graph if specified
    if args.graphimg_file:
        draw_graph(graph, args.graphimg_file)
        logger.success(f"Graph image saved to {args.graphimg_file} \n")


if __name__ == "__main__":  # pragma: no cover
    main()