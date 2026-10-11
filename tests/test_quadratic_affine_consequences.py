"""Exhaustive independent polynomial/affine parametrization checks."""
from random import Random

import pytest

from solver.quadratic_affine_consequences import affine_combinations, restrict_affine


def evaluate(constant, linear, quadratic, x):
    result = constant
    for i in range(len(linear)):
        if x >> i & 1:
            result ^= linear[i]
            for j in range(i + 1, len(linear)):
                if x >> j & 1:
                    result ^= quadratic[i * len(linear) + j]
    return result


@pytest.mark.parametrize('variables', range(7))
def test_kernel_spans_all_cancellations_and_restriction_preserves_all_roots(variables):
    rng = Random(860140 + variables)
    equations = 5
    for _ in range(8):
        constant = rng.randrange(1 << equations)
        linear = tuple(rng.randrange(1 << equations) for _ in range(variables))
        quadratic = [0] * (variables * variables)
        for i in range(variables):
            for j in range(i + 1, variables):
                value = rng.randrange(1 << equations)
                quadratic[i * variables + j] = quadratic[j * variables + i] = value
        kernel = affine_combinations(quadratic, variables, equations)
        expected = {mask for mask in range(1 << equations)
                    if all(bin(value & mask).count('1') % 2 == 0 for value in quadratic)}
        actual = {0}
        for mask in kernel:
            actual |= {value ^ mask for value in actual}
        assert actual == expected
        allowed = {x for x in range(1 << variables)
                   if all(bin(evaluate(constant, linear, quadratic, x) & mask).count('1') % 2 == 0
                          for mask in kernel)}
        result = restrict_affine(kernel, constant, linear)
        images = set()
        if result['consistent']:
            for y in range(1 << len(result['toggles'])):
                x = result['base']
                for j, toggle in enumerate(result['toggles']):
                    if y >> j & 1:
                        x ^= toggle
                images.add(x)
            assert len(images) == 1 << (variables - result['rank'])
        assert images == allowed
        roots = {x for x in range(1 << variables) if evaluate(constant, linear, quadratic, x) == 0}
        assert roots <= images


def test_two_nonlinear_equations_give_a_new_affine_equation():
    # x0*x1+x0=0, x0*x1+x1+1=0: their sum forces x0+x1=1.
    kernel = affine_combinations((0, 3, 3, 0), 2, 2)
    assert kernel == (3,)
    result = restrict_affine(kernel, 2, (1, 2))
    assert result['consistent'] and result['rank'] == 1
    assert {result['base'], result['base'] ^ result['toggles'][0]} == {1, 2}


def test_constant_contradiction_is_exact():
    kernel = affine_combinations((0, 3, 3, 0), 2, 2)
    assert not restrict_affine(kernel, 2, (0, 0))['consistent']
