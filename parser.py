from __future__ import annotations

import re
from typing import Optional

from ptx import (
    BlockItem,
    Branch,
    EntryDirective,
    Immediate,
    Instruction,
    Label,
    MemoryDecl,
    MemoryRef,
    MemorySymbol,
    MemoryType,
    Module,
    Opaque,
    Operand,
    ParamRef,
    Register,
    RegisterDecl,
    ScopedBlock,
    Statement,
    Vector,
)


def parse_register_decl(line: str) -> RegisterDecl:
    """
    Parse a single .reg directive line into a RegisterDecl.

    Supports directives such as:
        .reg .b32 %r<1251>;
        .reg .pred %p<7>
    """
    # Drop inline comments and trailing punctuation.
    cleaned = line.split("//", 1)[0].strip()
    cleaned = cleaned.rstrip(",;")

    pattern = (
        r"^\.reg\s+"
        r"(?P<dtype>\.\S+)\s+"
        r"(?P<prefix>%?\w+)"
        r"(?:<(?P<count>\d+)>)?"
        r"$"
    )
    m = re.match(pattern, cleaned)
    if not m:
        raise ValueError(f"Could not parse .reg directive: {line!r}")

    dtype = m.group("dtype").lstrip(".")
    prefix = m.group("prefix").lstrip("%")
    count_str = m.group("count")
    count = int(count_str) if count_str else 1

    return RegisterDecl(datatype=dtype, prefix=prefix, num_regs=count)


def parse_param_directive(line: str) -> MemoryDecl:
    """
    Parse a single .param directive line into a MemoryDecl.

    Supports directives such as:
        .param .align 4 .b8 foo[12],
        .param .u64 bar

    Trailing commas/semicolons are ignored.
    """
    # Drop inline comments and trailing punctuation.
    cleaned = line.split("//", 1)[0].strip()
    cleaned = cleaned.rstrip(",;")

    # Regex captures: optional .align <n>, datatype, name, optional [N]
    pattern = (
        r"^\.param\s+"
        r"(?:\.align\s+(?P<align>\d+)\s+)?"
        r"(?P<dtype>\.\S+)\s+"
        r"(?P<name>[^\s\[]+)"
        r"(?:\[(?P<count>\d*)\])?"
        r"$"
    )
    m = re.match(pattern, cleaned)
    if not m:
        raise ValueError(f"Could not parse .param directive: {line!r}")

    raw_align = m.group("align")
    dtype = m.group("dtype").lstrip(".")
    name = m.group("name")
    count = m.group("count")

    alignment = int(raw_align) if raw_align else None
    num_elements = int(count) if count else 1

    return MemoryDecl(
        alignment=alignment,
        datatype=dtype,
        name=name,
        num_elements=num_elements,
        memory_type=MemoryType.Param,
    )


def parse_memory_directive(line: str) -> MemoryDecl:
    """
    Parse a .global or .shared directive into a MemoryDecl.
    Supports optional .extern and .align, plus optional array counts.
    """
    cleaned = line.split("//", 1)[0].strip()
    cleaned = cleaned.rstrip(",;")

    pattern = (
        r"^(?P<extern>\.extern\s+)?"
        r"(?P<space>\.(?:global|shared))\s+"
        r"(?:\.align\s+(?P<align>\d+)\s+)?"
        r"(?P<dtype>\.\S+)\s+"
        r"(?P<name>[^\s\[]+)"
        r"(?:\[(?P<count>\d*)\])?"
        r"$"
    )
    m = re.match(pattern, cleaned)
    if not m:
        raise ValueError(f"Could not parse memory directive: {line!r}")

    space = m.group("space")
    memory_type = MemoryType.Global if space == ".global" else MemoryType.Shared

    raw_align = m.group("align")
    dtype = m.group("dtype").lstrip(".")
    name = m.group("name")
    count = m.group("count")

    alignment = int(raw_align) if raw_align else None
    if count is None:
        num_elements = 1
    elif count == "":
        num_elements = 0
    else:
        num_elements = int(count)

    return MemoryDecl(
        alignment=alignment,
        datatype=dtype,
        name=name,
        num_elements=num_elements,
        memory_type=memory_type,
    )


def _parse_register(token: str) -> Register:
    if token.startswith("%"):
        body = token[1:]
    else:
        body = token
    m = re.match(r"([A-Za-z_][\w\.]*?)(\d+)?$", body)
    if not m:
        raise ValueError(f"Could not parse register token: {token!r}")
    prefix, idx = m.group(1), m.group(2)
    return Register(prefix=prefix, idx=int(idx) if idx is not None else None)


def _split_operands(operand_text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth_bracket = 0
    depth_brace = 0
    for ch in operand_text:
        if ch == "," and depth_bracket == 0 and depth_brace == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf.clear()
            continue
        if ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_operand(
    token: str, mem_map: Optional[dict[str, MemoryDecl]] = None
) -> Operand:
    if token.startswith("{") and token.endswith("}"):
        inner = token[1:-1]
        regs = [
            _parse_register(part.strip()) for part in inner.split(",") if part.strip()
        ]
        return Vector(values=regs)

    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        mem_re = re.match(
            r"(?P<base>%[\w\.]+|[A-Za-z_]\w*)(?:(?P<sign>[+-])(?P<off>\d+))?$",
            inner,
        )
        if not mem_re:
            raise ValueError(f"Could not parse memory reference: {token!r}")
        base_token = mem_re.group("base")
        sign = mem_re.group("sign")
        off = mem_re.group("off")
        offset = int(off) if off else 0
        if sign == "-":
            offset = -offset
        base: Register | ParamRef
        if base_token.startswith("%"):
            base = _parse_register(base_token)
        else:
            base = ParamRef(name=base_token)
        return MemoryRef(base=base, offset=offset)

    if re.match(r"^-?(0x[0-9a-fA-F]+|\d+)$", token):
        return Immediate(value=int(token, 0))
    if re.match(r"^[0-9a-fA-F]+$", token):
        return Immediate(value=int(token, 16))

    # If token matches a known memory symbol, treat as memory reference with zero offset
    if mem_map and token in mem_map:
        return MemorySymbol(decl=mem_map[token])

    # Fallback: treat as register (supports bare predicate regs like 'p')
    return _parse_register(token)


def parse_instruction(
    line: str, mem_map: Optional[dict[str, MemoryDecl]] = None
) -> Instruction:
    """
    Parse a non-branch instruction line into an Instruction.

    Supports optional predicate prefix (e.g., @%p1), opcode with
    modifiers (e.g., ld.param.u32), and operands including registers,
    immediates, vectors, and memory references.
    """
    cleaned = line.split("//", 1)[0].strip()
    cleaned = cleaned.rstrip(";")
    if not cleaned:
        raise ValueError("Empty instruction line")

    predicate: Optional[Register] = None
    if cleaned.startswith("@"):
        pred_token, _, rest = cleaned[1:].lstrip().partition(" ")
        if not rest:
            raise ValueError(f"Malformed predicated instruction: {line!r}")
        cleaned = rest.strip()
        pred_token = pred_token.lstrip("!")
        predicate = _parse_register(pred_token)

    if not cleaned:
        raise ValueError(f"Could not parse instruction: {line!r}")

    opcode, _, operand_text = cleaned.partition(" ")
    opcode = opcode.strip()
    if opcode.startswith("bra"):
        raise ValueError("Branch instruction should be parsed separately")

    operands: list[Operand] = []
    if operand_text.strip():
        for tok in _split_operands(operand_text.strip()):
            operands.append(_parse_operand(tok, mem_map=mem_map))

    return Instruction(predicate=predicate, opcode=opcode, operands=operands)


def parse_branch(line: str) -> Branch:
    """
    Parse a branch instruction (bra / bra.uni) into a Branch node.
    """
    cleaned = line.split("//", 1)[0].strip()
    cleaned = cleaned.rstrip(";")
    if not cleaned:
        raise ValueError("Empty branch line")

    predicate: Optional[Register] = None
    if cleaned.startswith("@"):
        pred_token, _, rest = cleaned[1:].lstrip().partition(" ")
        if not rest:
            raise ValueError(f"Malformed predicated branch: {line!r}")
        cleaned = rest.strip()
        pred_token = pred_token.lstrip("!")
        predicate = _parse_register(pred_token)

    opcode, _, target_text = cleaned.partition(" ")
    opcode = opcode.strip()
    if not opcode.startswith("bra"):
        raise ValueError(f"Not a branch opcode: {opcode!r}")

    target_token = target_text.split(",", 1)[0].strip()
    if not target_token:
        raise ValueError(f"Missing branch target: {line!r}")

    is_uniform = ".uni" in opcode
    target = Label(name=target_token.lstrip("$"))

    return Branch(predicate=predicate, is_uniform=is_uniform, target=target)


def parse_label(line: str) -> Label:
    """
    Parse a label line of the form '<name>:' into a Label.
    """
    cleaned = line.split("//", 1)[0].strip()
    if not cleaned.endswith(":"):
        raise ValueError(f"Not a label: {line!r}")
    name = cleaned.rstrip(":").strip()
    if not name:
        raise ValueError(f"Empty label name: {line!r}")
    return Label(name=name.lstrip("$"))


def _brace_tokens(lines: list[str]) -> list[str]:
    tokens: list[str] = []
    for raw in lines:
        line = raw.split("//", 1)[0]
        buf: list[str] = []
        i = 0
        n = len(line)
        while i < n:
            ch = line[i]
            if ch == "{":
                close = line.find("}", i + 1)
                if close != -1:
                    inner = line[i + 1 : close]
                    # Treat as vector literal if it looks like an operand (contains % but no ';')
                    if "%" in inner and ";" not in inner:
                        buf.append(line[i : close + 1])
                        i = close + 1
                        continue
                seg = "".join(buf).strip()
                if seg:
                    tokens.append(seg)
                tokens.append(ch)
                buf.clear()
                i += 1
                continue
            if ch == "}":
                seg = "".join(buf).strip()
                if seg:
                    tokens.append(seg)
                tokens.append(ch)
                buf.clear()
                i += 1
                continue
            buf.append(ch)
            i += 1
        tail = "".join(buf).strip()
        if tail:
            tokens.append(tail)
    return tokens


def _process_statement(
    stmt: str, block: ScopedBlock, mem_map: Optional[dict[str, MemoryDecl]] = None
) -> None:
    if not stmt:
        return
    if stmt.startswith(".pragma"):
        # Skip pragmas inside blocks
        return
    label_match = re.match(r"^(?!.*::)([^:]+):\s*(.*)$", stmt)
    if label_match:
        name, rest = label_match.group(1), label_match.group(2)
        block.body.append(parse_label(f"{name}:"))
        if rest:
            _process_statement(rest, block)
        return
    if stmt.startswith(".reg"):
        block.registers.append(parse_register_decl(stmt))
        return
    try:
        block.body.append(parse_branch(stmt))
        return
    except ValueError:
        pass
    block.body.append(parse_instruction(stmt, mem_map=mem_map))


def parse_scoped_block(
    text: str, mem_map: Optional[dict[str, MemoryDecl]] = None
) -> ScopedBlock:
    """
    Parse a scoped block text (which may contain nested braces) into a ScopedBlock tree.
    Braces can appear inline with other statements.
    """
    tokens = _brace_tokens(text.splitlines())

    root = ScopedBlock(registers=[], body=[])
    stack: list[ScopedBlock] = [root]

    for tok in tokens:
        current = stack[-1]
        if tok == "{":
            new_block = ScopedBlock(registers=[], body=[])
            current.body.append(new_block)
            stack.append(new_block)
            continue
        if tok == "}":
            if len(stack) == 1:
                raise ValueError("Unmatched closing brace")
            stack.pop()
            continue
        # Otherwise process statements within this text segment; handle multiple ';' in one segment
        for stmt in tok.split(";"):
            cleaned = stmt.strip()
            if cleaned:
                _process_statement(cleaned, current, mem_map=mem_map)

    if len(stack) != 1:
        raise ValueError("Unclosed block: missing closing brace(s)")

    return root


def parse_entry_directive(
    text: str, mem_map: Optional[dict[str, MemoryDecl]] = None
) -> EntryDirective:
    """
    Parse an .entry directive (including its parameter list and scoped body).
    """
    m = re.search(
        r"\.entry\s+(?P<name>\S+)\s*\((?P<params>.*?)\)\s*(?P<body>.*)",
        text,
        flags=re.DOTALL,
    )
    if not m:
        raise ValueError("Could not parse .entry directive")

    name = m.group("name")
    params_block = m.group("params")
    body_text = m.group("body")

    params: list[MemoryDecl] = []
    for line in params_block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = stripped.rstrip(",;")
        params.append(parse_param_directive(stripped))

    brace_idx = body_text.find("{")
    if brace_idx == -1:
        raise ValueError("Entry missing body braces")

    directives_text = body_text[:brace_idx]
    body_block_text = body_text[brace_idx:]

    directives: list[Opaque] = []
    for line in directives_text.splitlines():
        cleaned = line.split("//", 1)[0].strip()
        if not cleaned:
            continue
        directives.append(Opaque(content=cleaned))

    body = parse_scoped_block(body_block_text, mem_map=mem_map)

    return EntryDirective(name=name, params=params, directives=directives, body=body)


def parse_module(text: str) -> Module:
    """
    Parse a PTX module text into a Module of statements (MemoryDecls and EntryDirectives).
    """
    statements: list[Statement] = []
    mem_map: dict[str, MemoryDecl] = {}
    in_entry = False
    entry_lines: list[str] = []
    brace_balance = 0
    seen_body = False

    for line in text.splitlines():
        stripped = line.split("//", 1)[0].strip()
        if not in_entry:
            if stripped.startswith(".entry") or stripped.startswith(".visible .entry"):
                in_entry = True
                entry_lines = [line]
                brace_balance = line.count("{") - line.count("}")
                seen_body = line.count("{") > 0
                if seen_body and brace_balance == 0:
                    statements.append(
                        parse_entry_directive("\n".join(entry_lines), mem_map=mem_map)
                    )
                    in_entry = False
                continue

            if not stripped:
                continue
            if (
                stripped.startswith(".global")
                or stripped.startswith(".shared")
                or stripped.startswith(".extern")
            ):
                try:
                    md = parse_memory_directive(stripped)
                    statements.append(md)
                    mem_map[md.name] = md
                    continue
                except ValueError:
                    pass
            statements.append(Opaque(content=stripped))
        else:
            entry_lines.append(line)
            brace_balance += line.count("{") - line.count("}")
            if line.count("{") > 0:
                seen_body = True
            if seen_body and brace_balance == 0:
                statements.append(
                    parse_entry_directive("\n".join(entry_lines), mem_map=mem_map)
                )
                in_entry = False

    if in_entry:
        raise ValueError("Unclosed entry block")

    return Module(statements=statements)
