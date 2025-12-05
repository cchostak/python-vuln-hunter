from vuln_hunter.utils.tokenizer import tokenize_source


def test_tokenizer_strips_comments_and_nl():
    code = """# comment\nimport os\n\n\ndef foo(x):\n    return x + 1\n"""
    tokens = tokenize_source(code)
    assert "#" not in tokens
    assert "import" in tokens and "os" in tokens
    assert "foo" in tokens
