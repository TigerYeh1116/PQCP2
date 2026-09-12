"""Bridge MPS-relaxed elites into the existing exact discrete SA completion.

PyTorch remains responsible for producing structured low-score centers on MPS.
The former pipeline stopped after rank quantization, which is not a local-search
algorithm on binary pairs. This bridge adds the missing discrete completion:
each elite is retargeted to the closest Project-2 target compatible with its
unchanged four parity counts, then written to a bounded, validated seed bank.
The compiled same-parity swap SA consumes that bank. Its candidates still pass
the independent Python verifier and exact/A-B-swap deduplication before L.txt.

Using a compiled completion is intentional: for L around 44, sequential SA has
tiny O(L) states and frequent dependencies; a GPU dispatch per move measured far
slower than the CPU C kernel. MPS is used for its strength—parallel continuous
candidate formation—while the existing exact incremental kernel finishes them.
"""

from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

from .checkpoint import atomic_write_json
from .objective import pqcp_objective
from .torch_polish import closest_content_target
from .target_profiles import TargetContentProfile, pair_content, target_content_profiles
from .verifier import verify_pqcp


def elite_pair_key(a: Sequence[int], b: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return exact/A-B-swap key; no rotation/complement farming is used."""
    left, right = tuple(a), tuple(b)
    return tuple(sorted((left, right)))


def write_elite_pair_seed_bank(
    elites: Iterable[Mapping], path: Path, *, length: int, compression: bool = True,
    retarget: bool = True,
) -> Tuple[dict, ...]:
    """Validate, deduplicate and atomically persist MPS elites for discrete SA.

    The C seed parser reads the established L.txt-shaped fields. A JSON sidecar
    retains full source scores/profiles and the selected completion target.
    Existing solutions may be present in an archive but are excluded: hybrid
    completion must not farm known pairs. The operation never changes L.txt.
    """
    records = []
    seen = set()
    source_targets = target_content_profiles(length, decimation_reduced=False) if not retarget else ()
    target_lookup = {
        (p.k, p.eta, p.a_even_ones, p.a_odd_ones, p.b_even_ones, p.b_odd_ones): p
        for p in source_targets
    }
    for source in sorted(elites, key=lambda item: (
        int(item.get("target_energy", item["score"])),
        int(item["score"]),
        int(item.get("iteration", 0)),
    )):
        a, b = tuple(source["A"]), tuple(source["B"])
        if len(a) != length or len(b) != length:
            raise ValueError("elite length does not match hybrid length")
        # The independent verifier already performs the full integer PACF
        # recomputation. Reuse its result instead of doing that O(L^2) work
        # twice for every exported seed.
        verification = verify_pqcp(a, b)
        exact = verification.profile
        if not exact:
            raise ValueError("elite is not a valid binary pair")
        score = pqcp_objective(exact)
        if list(exact) != list(source["profile"]) or score != int(source["score"]):
            raise ValueError("elite profile/score fails full recomputation")
        if verification.is_valid:
            continue
        key = elite_pair_key(a, b)
        if key in seen:
            continue
        seen.add(key)
        if retarget:
            target = closest_content_target(a, b, compression=compression)
        else:
            content = pair_content(a, b)
            target = target_lookup.get((int(source["k"]), int(source["sign"]), *content))
            if target is None:
                raise ValueError("elite source target is incompatible with its exact content")
        records.append({
            "L": length, "A": list(a), "B": list(b), "profile": list(exact),
            "score": score, "source_k": source.get("k"), "source_sign": source.get("sign"),
            "k": target.k, "sign": target.eta,
            "target_energy": source.get("target_energy"),
        })
    if not records:
        raise ValueError("no non-solution MPS elites are available for completion")
    blocks = []
    for record in records:
        blocks.append(
            "L={L}\nnonzero shifts={k},{mirror}\nnonzero PACS={pacs}\n"
            "a={a}\nb={b}\n".format(
                L=length, k=record["k"], mirror=length - record["k"],
                pacs=4 * record["sign"], a="".join(map(str, record["A"])),
                b="".join(map(str, record["B"])),
            )
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp".format(path.name))
    temporary.write_text("\n".join(blocks), encoding="ascii")
    temporary.replace(path)
    atomic_write_json(path.with_suffix(path.suffix + ".json"), {
        "format": "pqcp-torch-elite-seeds", "version": 1,
        "L": length, "compression": compression, "retarget": retarget,
        "records": records,
    })
    return tuple(records)


__all__ = ("elite_pair_key", "write_elite_pair_seed_bank")
