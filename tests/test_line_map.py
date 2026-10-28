"""Preprocessor.line_map: the (file, line) of each output line."""

from uc_core.preprocessor import Preprocessor


def test_line_map_follows_directives_continuations_and_includes(tmp_path):
    (tmp_path / "h.h").write_text("// c\n#define X 1\nint hh(void);\n\nint\n gg(void);\n")
    main = tmp_path / "m.c"
    src = ('#include "h.h"\n#define Y \\\n  2\nint a;\nint b = Y +\n  X;\n'
           'int z;\n')
    main.write_text(src)
    pp = Preprocessor([])
    out = pp.preprocess(src, str(main))
    lines = out.split("\n")
    assert len(pp.line_map) == len(lines)
    where = {text: pp.line_map[i] for i, text in enumerate(lines) if text}
    h = str(tmp_path / "h.h")
    assert where["int hh(void);"] == (h, 3)
    assert where[" gg(void);"] == (h, 6)
    assert where["int a;"] == (str(main), 4)
    assert where["int b = 2 +"] == (str(main), 5)
    assert where["  1;"] == (str(main), 6)
    assert where["int z;"] == (str(main), 7)
