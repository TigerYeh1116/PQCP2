from itertools import product
from pathlib import Path
import json
import pytest

from solver.coarse_row_completion import coarse_l1_anchors
from solver.full_row_completion import crt_trit_rows
from experiments.complete_l86_full_rows import audited_coarse_centers,anchor_candidates


def test_direct_trit_l1_balls_equal_every_literal_small_row():
    rows=tuple(product((-1,0,1),repeat=3))
    for row in rows:
        for radius in (0,1,2):
            outputs=list(coarse_l1_anchors(row,radius));actual=[r for r,_ in outputs]
            expected={r for r in rows if sum(abs(a-b) for a,b in zip(r,row))<=radius}
            assert len(actual)==len(set(actual)) and set(actual)==expected


def test_direct_coarse_balls_equal_all_binary_row_edit_images():
    for word in ((0,)*86,tuple(j%2 for j in range(86)),tuple(int(j%5==0) for j in range(86))):
        item=dict(words=(word,));c=crt_trit_rows(word)[0]
        literal={coarse for _,_,_,coarse,_ in anchor_candidates(item,'coarse-row',2)}
        direct={r for r,_ in coarse_l1_anchors(c,2)}
        assert literal==direct


def test_actual_coarse_bank_is_integer_audited_before_new_search(tmp_path):
    root=Path(__file__).resolve().parents[1]
    path=root/'results/L86_independent_norm_long_20261007/coarse_witnesses.json'
    result=audited_coarse_centers(path,0)
    assert len(result['seeds'])==9 and all(len(r)==43 for s in result['seeds'] for r in s['coarse_rows'])
    assert all(not s['source_full_verified'] for s in result['seeds'])
    bad=json.loads(path.read_text());bad['rows'][0]['result']['attempts']['coarse']['words'][0][0]=1
    changed=tmp_path/'corrupt.json';changed.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='integer norm audit'):audited_coarse_centers(changed,0)


def test_side_selection_preserves_the_entire_selected_literal_ball():
    pair=((-1,0,1),(0,0,1));item=dict(coarse_rows=pair)
    for side in (0,1):
        selected=list(anchor_candidates(item,'coarse-row',2,side))
        assert all(row[0]==side for row in selected)
        assert {row[3] for row in selected}=={
            row for row in product((-1,0,1),repeat=3)
            if sum(abs(a-b) for a,b in zip(row,pair[side]))<=2}
    assert len(list(anchor_candidates(item,'coarse-row',2)))==sum(
        len(list(anchor_candidates(item,'coarse-row',2,side))) for side in (0,1))
    for invalid in (-1,2,True):
        with pytest.raises(ValueError):list(anchor_candidates(item,'coarse-row',1,invalid))
