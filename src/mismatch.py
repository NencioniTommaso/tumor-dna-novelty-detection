"""
mismatch.py
Mismatch neighborhood generation for k-mer feature spaces.
Contains the epigenetic alphabet definition, combinatorial neighborhood
expansion, and full vocabulary enumeration.

These are pure functions with no I/O, no sklearn, and no scipy dependencies —
designed to be easy to unit test in isolation.
"""

import itertools
from functools import lru_cache
from typing import Tuple

# Expanded Epigenetic Alphabet
EPIGENETIC_ALPHABET = ('A', 'C', 'G', 'T', 'M', 'H')


@lru_cache(maxsize=100000)
def generate_mismatch_neighborhood(kmer: str, m: int = 1, alphabet: Tuple[str, ...] = EPIGENETIC_ALPHABET) -> Tuple[str, ...]:
    """
    Generates all k-mers within 'm' mismatches of the given kmer.
    Uses the 6-letter epigenetic alphabet to generate states.
    """
    if m == 0:
        return (kmer,)
        
    neighborhood = set([kmer])
    kmer_list = list(kmer)
    indices = list(range(len(kmer)))
    
    # Loop from 1 mismatch up to 'm' mismatches
    for num_mismatches in range(1, m + 1):
        for positions in itertools.combinations(indices, num_mismatches):
            for replacement_chars in itertools.product(alphabet, repeat=num_mismatches):
                is_true_mismatch = True
                for pos, char in zip(positions, replacement_chars):
                    if kmer_list[pos] == char:
                        is_true_mismatch = False
                        break
                
                if is_true_mismatch:
                    mutated_kmer = kmer_list.copy()
                    for pos, char in zip(positions, replacement_chars):
                        mutated_kmer[pos] = char
                    
                    neighborhood.add("".join(mutated_kmer))
                    
    return tuple(neighborhood)


@lru_cache(maxsize=100000)
def generate_weighted_mismatch_neighborhood(
    kmer: str, m: int = 1, alphabet: Tuple[str, ...] = EPIGENETIC_ALPHABET
) -> Tuple[Tuple[str, int], ...]:
    """
    Like generate_mismatch_neighborhood, but returns (neighbor, hamming_distance) tuples.
    The hamming distance is the minimum number of substitutions from the original kmer.
    """
    if m == 0:
        return ((kmer, 0),)

    # Dict mapping neighbor -> minimum Hamming distance from original
    neighbors = {kmer: 0}
    kmer_list = list(kmer)
    indices = list(range(len(kmer)))

    for num_mismatches in range(1, m + 1):
        for positions in itertools.combinations(indices, num_mismatches):
            for replacement_chars in itertools.product(alphabet, repeat=num_mismatches):
                is_true_mismatch = True
                for pos, char in zip(positions, replacement_chars):
                    if kmer_list[pos] == char:
                        is_true_mismatch = False
                        break

                if is_true_mismatch:
                    mutated = kmer_list.copy()
                    for pos, char in zip(positions, replacement_chars):
                        mutated[pos] = char
                    neighbor = "".join(mutated)
                    # Keep the minimum distance if reachable via multiple paths
                    if neighbor not in neighbors or num_mismatches < neighbors[neighbor]:
                        neighbors[neighbor] = num_mismatches

    return tuple(neighbors.items())


def build_full_vocabulary(k: int, alphabet: Tuple[str, ...] = EPIGENETIC_ALPHABET) -> dict:
    """
    Pre-enumerates all possible k-mers of length k from the given alphabet.
    Returns a deterministic vocabulary mapping each k-mer to a unique column index.
    This allows fully independent parallel feature extraction with no shared mutable state.
    """
    return {"".join(kmer): idx for idx, kmer in enumerate(itertools.product(alphabet, repeat=k))}
