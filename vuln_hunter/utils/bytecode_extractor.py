"""Extract bytecode operation names from Python source code."""
import dis
from typing import List


def extract_bytecode_ops(code: str) -> List[str]:
    instructions = []
    try:
        compiled = compile(code, filename="<string>", mode="exec")
    except Exception:
        return instructions
    for instr in dis.get_instructions(compiled):
        instructions.append(instr.opname)
    return instructions
