from vuln_hunter.utils.bytecode_extractor import extract_bytecode_ops


def test_extract_bytecode_ops_returns_instructions():
    code = """def add(a, b):\n    return a + b\n"""
    ops = extract_bytecode_ops(code)
    assert any(op.startswith("LOAD") for op in ops)
    assert "RETURN_VALUE" in ops
