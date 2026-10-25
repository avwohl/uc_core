"""#pragma uc80 <verb> <args>: the preprocessor records the pragma for the
compiler driver and removes the line from the token stream."""

from uc_core.preprocessor import Preprocessor


def run(src):
    pp = Preprocessor()
    out = pp.preprocess(src, "t.c")
    return pp, out


def test_pragma_is_recorded_in_order_with_its_position():
    pp, out = run("int a;\n#pragma uc80 keep f, g\n"
                  "#pragma uc80 vector 38h isr\nint b;\n")
    assert pp.target_pragmas == [
        ("keep", ["f", "g"], "t.c", 2),
        ("vector", ["38h", "isr"], "t.c", 3),
    ]
    assert "pragma" not in out and "int a;" in out and "int b;" in out


def test_pragma_in_a_skipped_branch_is_not_recorded():
    pp, _ = run("#if 0\n#pragma uc80 keep f\n#endif\n#pragma uc80 public x\n")
    assert [p[:2] for p in pp.target_pragmas] == [("public", ["x"])]


def test_other_pragmas_are_not_recorded():
    pp, _ = run("#pragma once\n#pragma pack(1)\n#pragma uc800 keep f\n")
    assert pp.target_pragmas == []
