"""Fixed-content FKM generation and exact factor-two lifts.

For an even binary sign sequence ``x`` of length ``L=2d``, genuine
factor-two compression is

``c[i] = x[i] + x[i+d]`` for ``0 <= i < d``.

Thus every compressed symbol is ``-2``, ``0``, or ``+2``.  A zero has two
possible lifts, while ``+/-2`` has a unique lift.  This module deliberately
places the two lifted signs at ``i`` and ``i+d``; adjacent-pair expansion is
not the compression used by the PACF aliasing theorem.

The FKM generator below emits one ternary necklace per cyclic orbit with an
exact ``(count(-2), count(0), count(+2))`` content.  Candidate lifts are
filtered/constructed to preserve the exact even/odd one counts required by a
Project target-content profile.  These are structured initial candidates,
not PQCP solutions or hard pruning decisions.
"""

import hashlib
import random
from typing import Iterator, List, Optional, Sequence, Tuple

CompressedWord = Tuple[int, ...]
BinaryWord = Tuple[int, ...]
CompressedContent = Tuple[int, int, int]

_ALPHABET = (-2, 0, 2)
_RNG_NAMESPACE = "pqcp2-compressed-fkm-v1"


def generate_compressed_fkm_sequences(
    content: CompressedContent,
    limit: Optional[int] = None,
) -> Iterator[CompressedWord]:
    """Yield ternary necklaces with exact ``(-2,0,+2)`` content.

    The recurrence is the fixed-content form of FKM.  Count bounds are
    carried through recursion, so impossible multiset prefixes are never
    completed.  ``limit`` bounds yielded representatives without building a
    large list.
    """
    wanted = _validate_content(content)
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
    ):
        raise ValueError("limit must be a non-negative integer or None")
    length = sum(wanted)
    word = [0] * (length + 1)
    used = [0, 0, 0]
    yielded = 0

    def visit(position: int, period: int) -> Iterator[CompressedWord]:
        nonlocal yielded
        if limit is not None and yielded >= limit:
            return
        if position > length:
            if length % period == 0 and tuple(used) == wanted:
                yielded += 1
                yield tuple(_ALPHABET[index] for index in word[1:])
            return

        symbol = word[position - period]
        if used[symbol] < wanted[symbol]:
            word[position] = symbol
            used[symbol] += 1
            yield from visit(position + 1, period)
            used[symbol] -= 1
        for symbol in range(word[position - period] + 1, len(_ALPHABET)):
            if used[symbol] >= wanted[symbol]:
                continue
            word[position] = symbol
            used[symbol] += 1
            yield from visit(position + 1, position)
            used[symbol] -= 1

    yield from visit(1, 1)


def factor2_compressed_contents(
    length: int,
    even_ones: int,
    odd_ones: int,
) -> Tuple[CompressedContent, ...]:
    """Return every compressed content compatible with the total one count.

    If ``n_-``, ``n_0`` and ``n_+`` count the three compressed symbols, then
    a lift contains exactly ``2*n_- + n_0`` one bits.  Even/odd feasibility is
    checked again for each concrete necklace because for ``L`` divisible by
    four it also depends on which compressed indices carry each symbol.
    """
    _validate_target_counts(length, even_ones, odd_ones)
    compressed_length = length // 2
    weight = even_ones + odd_ones
    values = []
    for negative in range(compressed_length + 1):
        zero = weight - 2 * negative
        positive = compressed_length - negative - zero
        if zero >= 0 and positive >= 0:
            values.append((negative, zero, positive))
    return tuple(values)


def lift_factor2_compressed(
    compressed: Sequence[int],
    zero_first_bits: Sequence[int],
) -> BinaryWord:
    """Lift a compressed sign word to bits at positions ``i`` and ``i+d``.

    ``zero_first_bits`` supplies the bit at position ``i`` for each zero
    symbol in encounter order; the bit at ``i+d`` is its complement.  Sign
    ``+1`` maps to bit 0 and sign ``-1`` maps to bit 1.
    """
    values = _validate_compressed_word(compressed)
    orientations = tuple(zero_first_bits)
    zero_count = values.count(0)
    if len(orientations) != zero_count or any(
        bit not in (0, 1) or isinstance(bit, bool) for bit in orientations
    ):
        raise ValueError("zero_first_bits must contain one binary bit per zero")
    distance = len(values)
    bits = [0] * (2 * distance)
    orientation_index = 0
    for index, value in enumerate(values):
        if value == 2:
            first, second = 0, 0
        elif value == -2:
            first, second = 1, 1
        else:
            first = orientations[orientation_index]
            second = 1 - first
            orientation_index += 1
        bits[index] = first
        bits[index + distance] = second
    return tuple(bits)


def compressed_fkm_lift_candidates(
    length: int,
    even_ones: int,
    odd_ones: int,
    seed: int,
    limit: int = 16,
    max_necklaces: Optional[int] = None,
    lifts_per_necklace: int = 4,
) -> Tuple[BinaryWord, ...]:
    """Return deterministic genuine-compression FKM lifts with exact content.

    Compatible compressed contents are visited round-robin after a
    deterministic shuffle.  Each necklace receives several independently
    sampled valid lifts, and an even cyclic shift removes the artificial
    absolute origin while retaining even/odd counts.  Work is bounded by
    ``max_necklaces`` (default ``max(64,16*limit)``).
    """
    _validate_target_counts(length, even_ones, odd_ones)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    for name, value in (("limit", limit), ("lifts_per_necklace", lifts_per_necklace)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("{} must be a positive integer".format(name))
    if max_necklaces is None:
        max_necklaces = max(64, 16 * limit)
    if (
        not isinstance(max_necklaces, int)
        or isinstance(max_necklaces, bool)
        or max_necklaces <= 0
    ):
        raise ValueError("max_necklaces must be a positive integer or None")

    compressed_length = length // 2
    if compressed_length % 2 == 0:
        # When d is even, i and i+d have the same parity.  Construct the even
        # and odd compressed subsequences from their own FKM necklaces instead
        # of hoping that a globally canonical ternary necklace happens to have
        # the requested positional parity content.
        return _same_parity_compressed_lifts(
            length, even_ones, odd_ones, seed, limit,
            max_necklaces, lifts_per_necklace,
        )

    contents = list(factor2_compressed_contents(length, even_ones, odd_ones))
    order_rng = _domain_rng(seed, "content-order", length, even_ones, odd_ones)
    order_rng.shuffle(contents)
    streams = [iter(generate_compressed_fkm_sequences(content)) for content in contents]
    active = list(range(len(streams)))
    candidates: List[BinaryWord] = []
    seen = set()
    examined = 0
    round_index = 0
    while active and len(candidates) < limit and examined < max_necklaces:
        next_active = []
        for stream_index in active:
            if len(candidates) >= limit or examined >= max_necklaces:
                break
            try:
                compressed = next(streams[stream_index])
            except StopIteration:
                continue
            next_active.append(stream_index)
            examined += 1
            for lift_index in range(lifts_per_necklace):
                lift_rng = _domain_rng(
                    seed, "lift", length, even_ones, odd_ones,
                    round_index, stream_index, lift_index, compressed,
                )
                lifted = _random_valid_lift(
                    compressed, even_ones, odd_ones, lift_rng
                )
                if lifted is None:
                    break
                # Only even rotations are used, so exact parity content stays
                # invariant.  This changes origin, not the candidate's PACF.
                offset = 2 * lift_rng.randrange(length // 2)
                lifted = lifted[offset:] + lifted[:offset]
                if lifted not in seen:
                    seen.add(lifted)
                    candidates.append(lifted)
                    if len(candidates) >= limit:
                        break
        active = next_active
        round_index += 1
    return tuple(candidates)


def _same_parity_compressed_lifts(
    length: int,
    even_ones: int,
    odd_ones: int,
    seed: int,
    limit: int,
    max_necklaces: int,
    lifts_per_necklace: int,
) -> Tuple[BinaryWord, ...]:
    """Build L divisible-by-four lifts from two parity-necklace pools."""
    compressed_length = length // 2
    parity_length = compressed_length // 2
    side_limit = max(8, min(max_necklaces, max(4 * limit, 16)))
    even_pool = _bounded_weight_necklaces(
        parity_length, even_ones, seed, "even-compressed", side_limit
    )
    odd_pool = _bounded_weight_necklaces(
        parity_length, odd_ones, seed, "odd-compressed", side_limit
    )
    if not even_pool or not odd_pool:
        return ()
    pairs = [(left, right) for left in range(len(even_pool))
             for right in range(len(odd_pool))]
    pair_rng = _domain_rng(seed, "parity-pair-order", length, even_ones, odd_ones)
    pair_rng.shuffle(pairs)
    candidates: List[BinaryWord] = []
    seen = set()
    for pair_index, (left, right) in enumerate(pairs[:max_necklaces]):
        compressed_list = [0] * compressed_length
        compressed_list[0::2] = even_pool[left]
        compressed_list[1::2] = odd_pool[right]
        compressed = tuple(compressed_list)
        for lift_index in range(lifts_per_necklace):
            lift_rng = _domain_rng(
                seed, "parity-lift", length, even_ones, odd_ones,
                pair_index, lift_index, compressed,
            )
            lifted = _random_valid_lift(
                compressed, even_ones, odd_ones, lift_rng
            )
            if lifted is None:  # pragma: no cover - construction guarantees it
                raise RuntimeError("parity-compressed FKM construction is inconsistent")
            offset = 2 * lift_rng.randrange(length // 2)
            lifted = lifted[offset:] + lifted[:offset]
            if lifted not in seen:
                seen.add(lifted)
                candidates.append(lifted)
                if len(candidates) >= limit:
                    return tuple(candidates)
    return tuple(candidates)


def _bounded_weight_necklaces(
    compressed_length: int,
    lifted_ones: int,
    seed: int,
    domain: str,
    limit: int,
) -> Tuple[CompressedWord, ...]:
    """Round-robin ternary contents whose factor-two lift has a fixed weight."""
    contents = []
    for negative in range(compressed_length + 1):
        zero = lifted_ones - 2 * negative
        positive = compressed_length - negative - zero
        if zero >= 0 and positive >= 0:
            contents.append((negative, zero, positive))
    order_rng = _domain_rng(
        seed, domain + "-content-order", compressed_length, lifted_ones
    )
    order_rng.shuffle(contents)
    streams = [iter(generate_compressed_fkm_sequences(content)) for content in contents]
    active = list(range(len(streams)))
    words: List[CompressedWord] = []
    while active and len(words) < limit:
        next_active = []
        for stream_index in active:
            try:
                words.append(next(streams[stream_index]))
            except StopIteration:
                continue
            next_active.append(stream_index)
            if len(words) >= limit:
                break
        active = next_active
    return tuple(words)


def _random_valid_lift(
    compressed: CompressedWord,
    even_ones: int,
    odd_ones: int,
    rng: random.Random,
) -> Optional[BinaryWord]:
    """Construct one uniformly oriented lift under exact parity counts."""
    distance = len(compressed)
    zero_positions = [index for index, value in enumerate(compressed) if value == 0]
    negative = compressed.count(-2)
    if distance % 2 == 0:
        # Positions i and i+d have the same parity.  A zero contributes one one
        # to that parity regardless of its orientation.
        even = sum(
            2 if value == -2 else int(value == 0)
            for index, value in enumerate(compressed) if index % 2 == 0
        )
        odd = sum(
            2 if value == -2 else int(value == 0)
            for index, value in enumerate(compressed) if index % 2 == 1
        )
        if (even, odd) != (even_ones, odd_ones):
            return None
        orientations = [rng.randrange(2) for _ in zero_positions]
    else:
        # Positions i and i+d have opposite parity.  Every -2 pair contributes
        # one one to each parity; each zero assigns its sole one to one side.
        ones_for_even = even_ones - negative
        if not 0 <= ones_for_even <= len(zero_positions):
            return None
        if odd_ones - negative != len(zero_positions) - ones_for_even:
            return None
        chosen = set(rng.sample(zero_positions, ones_for_even))
        orientations = []
        for index in zero_positions:
            one_at_first = index in chosen if index % 2 == 0 else index not in chosen
            orientations.append(1 if one_at_first else 0)
    lifted = lift_factor2_compressed(compressed, orientations)
    if (sum(lifted[0::2]), sum(lifted[1::2])) != (even_ones, odd_ones):
        raise RuntimeError("constructed compressed lift changed target content")
    return lifted


def _validate_content(content: Sequence[int]) -> CompressedContent:
    values = tuple(content)
    if len(values) != 3 or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise ValueError("content must be three non-negative integer counts")
    if sum(values) <= 0:
        raise ValueError("compressed content must have positive total length")
    return values[0], values[1], values[2]


def _validate_compressed_word(compressed: Sequence[int]) -> CompressedWord:
    values = tuple(compressed)
    if not values or any(
        value not in _ALPHABET or isinstance(value, bool) for value in values
    ):
        raise ValueError("compressed word must contain only -2, 0, and 2")
    return values


def _validate_target_counts(length: int, even_ones: int, odd_ones: int) -> None:
    if (
        not isinstance(length, int) or isinstance(length, bool)
        or length < 4 or length % 2
    ):
        raise ValueError("length must be an even integer at least four")
    half = length // 2
    for name, value in (("even_ones", even_ones), ("odd_ones", odd_ones)):
        if (
            not isinstance(value, int) or isinstance(value, bool)
            or not 0 <= value <= half
        ):
            raise ValueError("{} must be between zero and L/2".format(name))


def _domain_rng(seed: int, domain: str, *fields: object) -> random.Random:
    payload = (_RNG_NAMESPACE, domain, str(seed)) + tuple(map(str, fields))
    digest = hashlib.sha256("|".join(payload).encode("ascii")).digest()
    return random.Random(int.from_bytes(digest, "big"))


__all__ = (
    "compressed_fkm_lift_candidates", "factor2_compressed_contents",
    "generate_compressed_fkm_sequences", "lift_factor2_compressed",
)
