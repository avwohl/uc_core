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


def test_line_map_follows_a_line_directive(tmp_path):
    src = 'int a;\n#line 100 "fake.c"\nint b;\nint c;\n'
    pp = Preprocessor([])
    out = pp.preprocess(src, str(tmp_path / "m.c"))
    lines = out.split("\n")
    where = {text: pp.line_map[i] for i, text in enumerate(lines) if text}
    assert where["int a;"] == (str(tmp_path / "m.c"), 1)
    assert where["int b;"] == ("fake.c", 100)
    assert where["int c;"] == ("fake.c", 101)


def test_a_continued_line_records_where_each_source_line_begins(tmp_path):
    """A logical line that joined three source lines is one output line.
    line_breaks says at which columns the later source lines begin."""
    src = 'int a = 1 + \\\n  2 + \\\n  3; int z;\nint y;\n'
    pp = Preprocessor([])
    out = pp.preprocess(src, str(tmp_path / "m.c"))
    f = str(tmp_path / "m.c")
    lines = out.split("\n")
    assert lines[0] == "int a = 1 +   2 +   3; int z;"
    assert pp.line_map[0] == (f, 1)
    assert pp.line_map[1] == (f, 4)
    assert pp.line_breaks[0] == [(lines[0].index("  2"), f, 2),
                                 (lines[0].index("  3"), f, 3)]


def test_a_string_continuation_and_a_macro_in_the_line(tmp_path):
    src = '#define K 5\nconst char *s = "ab\\\ncd"; int z = K;\n'
    pp = Preprocessor([])
    out = pp.preprocess(src, str(tmp_path / "m.c"))
    f = str(tmp_path / "m.c")
    line = out.split("\n")[0]
    assert line == 'const char *s = "abcd"; int z = 5;'
    assert pp.line_breaks[0] == [(line.index("cd"), f, 3)]
