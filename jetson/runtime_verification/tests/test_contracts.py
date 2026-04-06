"""Tests for contracts module — 40+ tests."""

import pytest
from jetson.runtime_verification.contracts import (
    Contract,
    ContractResult,
    ContractChecker,
)


# ---------- Contract dataclass tests ----------

class TestContractDataclass:
    def test_create_minimal(self):
        c = Contract(name="test")
        assert c.name == "test"
        assert c.preconditions == []
        assert c.postconditions == []
        assert c.invariants == []
        assert c.description == ""

    def test_create_with_preconditions(self):
        c = Contract(
            name="div",
            preconditions=[lambda inp: inp.get("denominator", 0) != 0],
        )
        assert len(c.preconditions) == 1

    def test_create_with_postconditions(self):
        c = Contract(
            name="positive_result",
            postconditions=[lambda inp, out: out > 0],
        )
        assert len(c.postconditions) == 1

    def test_create_with_invariants(self):
        c = Contract(
            name="bounded",
            invariants=[lambda inp, out: out <= 1000],
        )
        assert len(c.invariants) == 1

    def test_create_full(self):
        c = Contract(
            name="full",
            preconditions=[lambda x: x > 0],
            postconditions=[lambda i, o: o < 100],
            invariants=[lambda i, o: o >= 0],
            description="Full contract",
        )
        assert c.description == "Full contract"


# ---------- ContractResult dataclass tests ----------

class TestContractResultDataclass:
    def test_create_passed(self):
        r = ContractResult(passed=True, phase="precondition")
        assert r.passed is True
        assert r.phase == "precondition"
        assert r.condition is None
        assert r.details is None

    def test_create_failed(self):
        r = ContractResult(
            passed=False,
            phase="postcondition",
            condition="postcondition_0",
            details="Output out of range",
        )
        assert r.passed is False
        assert r.condition == "postcondition_0"
        assert r.details == "Output out of range"


# ---------- ContractChecker tests ----------

class TestContractChecker:
    def setup_method(self):
        self.cc = ContractChecker()

    def test_register_contract(self):
        c = Contract(name="test")
        self.cc.register_contract(c)
        assert "test" in self.cc._contracts

    def test_register_multiple(self):
        for i in range(5):
            self.cc.register_contract(Contract(name=f"c{i}"))
        assert len(self.cc._contracts) == 5

    def test_register_overwrite(self):
        self.cc.register_contract(Contract(name="x", preconditions=[lambda s: True]))
        self.cc.register_contract(Contract(name="x", preconditions=[lambda s: False]))
        assert len(self.cc._contracts["x"].preconditions) == 1

    # --- check_preconditions ---
    def test_preconditions_all_pass(self):
        c = Contract(
            name="pos",
            preconditions=[
                lambda inp: inp["val"] > 0,
                lambda inp: inp["val"] < 100,
            ],
        )
        self.cc.register_contract(c)
        result = self.cc.check_preconditions("pos", {"val": 50})
        assert result.passed is True
        assert result.phase == "precondition"

    def test_preconditions_first_fails(self):
        c = Contract(
            name="pos",
            preconditions=[lambda inp: inp["val"] > 0],
        )
        self.cc.register_contract(c)
        result = self.cc.check_preconditions("pos", {"val": -1})
        assert result.passed is False
        assert result.phase == "precondition"

    def test_preconditions_second_fails(self):
        c = Contract(
            name="range",
            preconditions=[
                lambda inp: inp["val"] > 0,
                lambda inp: inp["val"] < 100,
            ],
        )
        self.cc.register_contract(c)
        result = self.cc.check_preconditions("range", {"val": 200})
        assert result.passed is False
        assert "precondition_1" in result.condition

    def test_preconditions_no_preconditions(self):
        c = Contract(name="empty")
        self.cc.register_contract(c)
        result = self.cc.check_preconditions("empty", {"val": 42})
        assert result.passed is True

    def test_preconditions_contract_not_found(self):
        result = self.cc.check_preconditions("missing", {})
        assert result.passed is False
        assert "not registered" in result.details

    def test_preconditions_exception_in_check(self):
        def bad_pre(inp):
            raise RuntimeError("boom")

        c = Contract(name="bad", preconditions=[bad_pre])
        self.cc.register_contract(c)
        result = self.cc.check_preconditions("bad", {})
        assert result.passed is False
        assert "boom" in result.details

    # --- check_postconditions ---
    def test_postconditions_all_pass(self):
        c = Contract(
            name="double",
            postconditions=[lambda inp, out: out == inp["x"] * 2],
        )
        self.cc.register_contract(c)
        result = self.cc.check_postconditions("double", {"x": 5}, 10)
        assert result.passed is True

    def test_postconditions_fail(self):
        c = Contract(
            name="double",
            postconditions=[lambda inp, out: out == inp["x"] * 2],
        )
        self.cc.register_contract(c)
        result = self.cc.check_postconditions("double", {"x": 5}, 11)
        assert result.passed is False
        assert result.phase == "postcondition"

    def test_postconditions_contract_not_found(self):
        result = self.cc.check_postconditions("missing", {}, None)
        assert result.passed is False

    def test_postconditions_exception_in_check(self):
        def bad_post(inp, out):
            raise ValueError("err")

        c = Contract(name="bp", postconditions=[bad_post])
        self.cc.register_contract(c)
        result = self.cc.check_postconditions("bp", {}, None)
        assert result.passed is False
        assert "err" in result.details

    def test_postconditions_with_invariants_pass(self):
        c = Contract(
            name="bounded",
            postconditions=[lambda i, o: o > 0],
            invariants=[lambda i, o: o < 1000],
        )
        self.cc.register_contract(c)
        result = self.cc.check_postconditions("bounded", {}, 500)
        assert result.passed is True

    def test_postconditions_with_invariants_fail(self):
        c = Contract(
            name="bounded",
            postconditions=[lambda i, o: o > 0],
            invariants=[lambda i, o: o < 100],
        )
        self.cc.register_contract(c)
        result = self.cc.check_postconditions("bounded", {}, 500)
        assert result.passed is False
        assert result.phase == "invariant"

    def test_postconditions_invariant_exception(self):
        def bad_inv(i, o):
            raise RuntimeError("inv_err")

        c = Contract(name="bi", invariants=[bad_inv])
        self.cc.register_contract(c)
        result = self.cc.check_postconditions("bi", {}, 1)
        assert result.passed is False
        assert result.phase == "invariant"

    # --- wrap_function ---
    def test_wrap_function_passes(self):
        c = Contract(
            name="square",
            preconditions=[lambda inp: inp["args"][0] >= 0],
            postconditions=[lambda inp, out: out >= 0],
        )
        self.cc.register_contract(c)
        wrapped = self.cc.wrap_function(lambda x: x * x, c)
        result = wrapped(5)
        assert result == 25

    def test_wrap_function_precondition_fails(self):
        c = Contract(
            name="positive",
            preconditions=[lambda inp: inp["args"][0] > 0],
        )
        self.cc.register_contract(c)
        wrapped = self.cc.wrap_function(lambda x: x, c)
        with pytest.raises(AssertionError, match="Precondition failed"):
            wrapped(-1)

    def test_wrap_function_postcondition_fails(self):
        c = Contract(
            name="always_pos",
            postconditions=[lambda inp, out: out > 0],
        )
        self.cc.register_contract(c)
        wrapped = self.cc.wrap_function(lambda x: -x, c)
        with pytest.raises(AssertionError, match="Postcondition failed"):
            wrapped(5)

    def test_wrap_function_preserves_name(self):
        c = Contract(name="named")
        self.cc.register_contract(c)
        wrapped = self.cc.wrap_function(lambda x: x, c)
        assert "contract_wrapped" in wrapped.__name__

    def test_wrap_function_has_contract_attribute(self):
        c = Contract(name="attr_test")
        self.cc.register_contract(c)
        wrapped = self.cc.wrap_function(lambda x: x, c)
        assert hasattr(wrapped, "_contract_name")

    # --- verify_function ---
    def test_verify_function_all_pass(self):
        c = Contract(
            name="double",
            preconditions=[lambda inp: inp >= 0],
            postconditions=[lambda inp, out: out == inp * 2],
        )
        self.cc.register_contract(c)
        results = self.cc.verify_function(lambda x: x * 2, c, [1, 5, 10])
        assert all(r.passed for r in results)

    def test_verify_function_pre_fail(self):
        c = Contract(
            name="pos",
            preconditions=[lambda inp: inp > 0],
            postconditions=[lambda inp, out: out > 0],
        )
        self.cc.register_contract(c)
        results = self.cc.verify_function(lambda x: x * 2, c, [1, -5, 10])
        # -5 should fail precondition
        failed = [r for r in results if not r.passed]
        assert len(failed) >= 1

    def test_verify_function_exception(self):
        def boom(x):
            raise RuntimeError("explode")

        c = Contract(
            name="boom_contract",
            preconditions=[lambda inp: True],
        )
        self.cc.register_contract(c)
        results = self.cc.verify_function(boom, c, [1, 2])
        failed = [r for r in results if not r.passed and r.phase == "execution"]
        assert len(failed) == 2

    def test_verify_function_empty_inputs(self):
        c = Contract(name="empty", preconditions=[lambda inp: True])
        self.cc.register_contract(c)
        results = self.cc.verify_function(lambda x: x, c, [])
        assert results == []

    # --- compute_contract_coverage ---
    def test_coverage_full(self):
        cov = ContractChecker.compute_contract_coverage(10, 10)
        assert cov == 100.0

    def test_coverage_half(self):
        cov = ContractChecker.compute_contract_coverage(8, 4)
        assert cov == 50.0

    def test_coverage_zero(self):
        cov = ContractChecker.compute_contract_coverage(0, 5)
        assert cov == 0.0

    def test_coverage_zero_contracts(self):
        cov = ContractChecker.compute_contract_coverage(3, 0)
        assert cov == 0.0

    def test_coverage_clamped_over_100(self):
        cov = ContractChecker.compute_contract_coverage(10, 12)
        assert cov == 100.0

    def test_coverage_clamped_negative(self):
        cov = ContractChecker.compute_contract_coverage(-2, 10)
        assert cov == 0.0
