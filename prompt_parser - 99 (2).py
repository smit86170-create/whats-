from __future__ import annotations
import re
import lark
import random
import logging
from functools import lru_cache
import hashlib
from collections import namedtuple
import torch
import os

# Configure logging with configurable level
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - [ERROR-%(error_code)s] - %(message)s')

# Configurable cache size
CACHE_SIZE = int(os.getenv('PROMPT_PARSER_CACHE_SIZE', 1024))

# Grammar for parsing complex prompts
schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*

prompt: prompt_core

prompt_core: (scheduled | emphasized | grouped | alternate | alternate_distinct | alternate1 | alternate2 | sequence | nested_sequence | top_level_sequence | compound | numbered | and_rule | plain | WHITESPACE)*

reverse_flag: "reverse" | "r"
step_range_list: step_range ("," step_range)*
step_range: NUMBER_Q "-" NUMBER_Q | NUMBER_Q "%-" NUMBER_Q "%"

!emphasized: "(" prompt ")" (":" NUMBER_Q)? | "[" prompt "]" | "(" prompt ":" prompt ")"
!scheduled: "[" prompt (":" prompt)* ":" NUMBER_Q (step_range_list | reverse_flag | reverse_flag step_range_list)? "]"
!alternate: "[" prompt "|" prompt ("|" prompt)* "]"
!alternate_distinct: "[" prompt ("|" prompt)* "]!"
!alternate1: "(" prompt "|" prompt+ ")"
!alternate2: compound ("|" compound)+

!top_level_sequence: prompt "::" sequence ("!!" | "::" sequence)*
!sequence: prompt "::" prompt ("," | WHITESPACE)* nested_sequence* "!"  
!nested_sequence: "::" prompt ("," | WHITESPACE)* ("~" | "!")



!grouped: "{" (prompt_core | NUMBER_Q) ("," | "|")* "}"
!compound: /[a-zA-Z0-9]+(_[a-zA-Z0-9]+)+/
!numbered: NUMBER_Q "q" (grouped | compound | alternate | alternate_distinct | alternate1 | alternate2)
!and_rule: (plain | compound) ("&" (plain | compound))+

plain: /([^\\\[\]()&]|\\.)+/
WHITESPACE: /\s+/

NUMBER_Q: /[0-9]+/
NUMBER: SIGNED_NUMBER | PERCENTAGE
PERCENTAGE: SIGNED_NUMBER "%"
SIGNED_NUMBER: /[-+]?\d*\.?\d+/

""", start="start", parser="earley", propagate_positions=True)

# Define DictWithShape class for compatibility with tensor operations
class DictWithShape(dict):
    def __init__(self, data, shape):
        super().__init__(data)
        self.shape = shape

# Custom class to mark unparsed prompts
class PromptParsingFailed:
    def __init__(self, prompt: str, reason: str, error_code: str):
        self.prompt = prompt
        self.reason = reason
        self.error_code = error_code
        self.is_unparsed = True

    def __str__(self):
        return self.prompt

def validate_prompt(prompt: str) -> bool:
    stack = []
    for i, char in enumerate(prompt):
        if char in '([':
            stack.append((char, i))
        elif char == ')':
            if not stack or stack[-1][0] != '(':
                snippet = prompt[max(0, i-10):i+10]
                logging.warning(f"Unbalanced parenthesis at position {i}: ...{snippet}...", extra={'error_code': 'VAL_PAREN'})
                return True
            stack.pop()
        elif char == ']':
            if not stack or stack[-1][0] != '[':
                snippet = prompt[max(0, i-10):i+10]
                logging.warning(f"Unbalanced bracket at position {i}: ...{snippet}...", extra={'error_code': 'VAL_BRACK'})
                return True
            stack.pop()
    if stack:
        char, pos = stack[-1]
        snippet = prompt[max(0, pos-10):pos+10]
        logging.warning(f"Unclosed {char} at position {pos}: ...{snippet}...", extra={'error_code': 'VAL_UNCLOSED'})

    if '::' in prompt and not any(prompt.endswith(s) for s in ('!', ';', '!!')) and ':::' not in prompt:
        snippet = re.search(r'.{0,20}::.{0,20}', prompt)
        if snippet:
            logging.warning(f"Unterminated sequence in: ...{snippet.group(0)}...", extra={'error_code': 'VAL_SEQ'})

    scheduled_matches = re.finditer(r'\[([^\[\]]*?):([^\[\]]*?)\]', prompt)
    for match in scheduled_matches:
        value = match.group(2)
        try:
            weight = float(value)
            if weight < 0:
                logging.warning(f"Negative weight '{value}' in scheduled prompt: ...{match.group(0)}...", extra={'error_code': 'VAL_WEIGHT_NEG'})
        except ValueError:
            logging.warning(f"Invalid weight '{value}' in scheduled prompt: ...{match.group(0)}...", extra={'error_code': 'VAL_WEIGHT_INV'})

    emphasized_matches = re.finditer(r'\(([^():]*):([-+]?\d*\.?\d+)\)', prompt)
    for match in emphasized_matches:
        value = match.group(2)
        try:
            weight = float(value)
            if weight < 0:
                logging.warning(f"Negative weight '{value}' in emphasized prompt: ...{match.group(0)}...", extra={'error_code': 'VAL_EMPH_NEG'})
        except ValueError:
            logging.warning(f"Invalid weight '{value}' in emphasized prompt: ...{match.group(0)}...", extra={'error_code': 'VAL_EMPH_INV'})

    if re.search(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', prompt):
        logging.warning(f"Prompt contains invalid control characters near: ...{prompt}...", extra={'error_code': 'VAL_CTRL'})

    if " AND " in prompt:
        subprompts = re.split(r"\s*AND\s*", prompt)
        for subprompt in subprompts:
            validate_prompt(subprompt.strip())

    if re.search(r'\[\s*\|\s*\]', prompt) or re.search(r'\[\s*(\|\s*)+]', prompt):
        match = re.search(r'\[[^\[\]]*?\|[^\[\]]*?]', prompt)
        if match:
            logging.warning(f"Empty or invalid alternate group found: ...{match.group(0)}...", extra={'error_code': 'VAL_ALT'})

    return True

def resolve_tree(tree: lark.Tree | lark.Token, keep_spacing: bool = False) -> str:
    """Recursively resolve a parse tree node to a string, optionally preserving spacing."""
    if isinstance(tree, lark.Tree):
        children = []
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                if keep_spacing:
                    children.append(" ")
                continue
            if isinstance(child, lark.Token) and child.type in ("NUMBER", "COLON"):
                continue
            children.append(resolve_tree(child, keep_spacing))
        result = "".join(str(c) for c in children if c)
        return re.sub(r"[\s\u2028\u2029]+", " ", result).strip() if keep_spacing else result.strip()
    return str(tree).strip()

def hash_tree(tree: lark.Tree | lark.Token) -> str:
    """Generate a stable hash for a Lark tree or token for caching."""
    if isinstance(tree, lark.Tree):
        children_hashes = ''.join(hash_tree(c) for c in sorted(tree.children, key=lambda x: str(x)))
        return hashlib.md5((tree.data + children_hashes).encode()).hexdigest()
    return hashlib.md5(str(tree).strip().encode()).hexdigest()

@lru_cache(maxsize=CACHE_SIZE)
def collect_steps(steps: int, tree: lark.Tree | lark.Token, prefix: str = "", suffix: str = "", depth: int = 0, verbose: bool = False, use_scheduling: bool = True, seed: int | None = 42, use_legacy: bool = False) -> list[list[int | str]]:
    """Collect schedules for a prompt parse tree, optimized with caching."""
    random_gen = random.Random(seed) if seed is not None and not use_legacy else random.Random(seed)
    schedules = []
    indent = "  " * depth
    if verbose:
        logging.info(f"{indent}Processing: {tree}")

    if isinstance(tree, str):
        schedules = [[steps, prefix + tree + suffix]]
    elif isinstance(tree, lark.Tree):
        transformer = ScheduleTransformer(steps, 1, seed, use_legacy)
        schedules = transformer.collect_steps(steps, tree, use_scheduling, prefix, suffix, depth)
    
    if verbose:
        logging.info(f"{indent}Schedules: {schedules}")
    return schedules

class ScheduleTransformer(lark.Transformer):
    """Transformer for converting parse trees into prompt schedules."""
    def __init__(self, total_steps: int, current_step: int = 1, seed: int | None = 42, use_legacy: bool = False):
        super().__init__()
        self.total_steps = total_steps
        self.current_step = current_step
        self.random_gen = random.Random(seed) if not use_legacy else random.Random(seed)
        self.use_legacy = use_legacy
        self.logger = logging.getLogger(__name__)

    def collect_steps(self, steps: int, tree: lark.Tree | lark.Token, use_scheduling: bool, prefix: str = "", suffix: str = "", depth: int = 0) -> list[list[int | str]]:
        self.total_steps = steps
        self.current_step = 1
        return self._process_tree(tree, steps, prefix, suffix, depth, use_scheduling)

    def _process_tree(self, tree: lark.Tree | lark.Token, steps: int, prefix: str, suffix: str, depth: int, use_scheduling: bool) -> list[list[int | str]]:
        if isinstance(tree, str):
            return [[steps, prefix + tree + suffix]]
        elif isinstance(tree, lark.Tree):
            return self._process_tree_node(tree, steps, prefix, suffix, depth, use_scheduling)
        return [[steps, prefix + str(tree).strip() + suffix]]

    def _process_tree_node(self, tree: lark.Tree, steps: int, prefix: str, suffix: str, depth: int, use_scheduling: bool) -> list[list[int | str]]:
        schedules = []
        data = tree.data
        children = tree.children

        if data == "plain":
            schedules = [[steps, prefix + self._resolve_tree(tree, keep_spacing=True) + suffix]]
        elif data == "scheduled" and use_scheduling:
            schedules = self._process_scheduled(tree, steps, prefix, suffix, depth, use_scheduling)
        elif data == "alternate":
            options = [self._resolve_tree(c, keep_spacing=True) for c in children if self._resolve_tree(c) or self._resolve_tree(c) == ""]
            if self.use_legacy:
                option = options[0] if options else "empty_prompt"
                schedules = [[steps, prefix + option + suffix]]
            else:
                schedules = [[step, prefix + (options[(step - 1) % len(options)] if options else "empty_prompt") + suffix] for step in range(1, steps + 1) if options]
        elif data == "alternate_distinct":
            options = [self._resolve_tree(c, keep_spacing=True) for c in children if self._resolve_tree(c)]
            if not options:
                self.logger.warning("Empty alternate_distinct group. Using 'empty_prompt'.", extra={'error_code': 'ALT_DIST002'})
                schedules = [[steps, prefix + "empty_prompt" + suffix]]
            else:
                option = options[0] if self.use_legacy else self.random_gen.choice(options)
                schedules = [[steps, prefix + option + suffix]]
        elif data == "alternate1":
            options = [self._resolve_tree(c, keep_spacing=True) for c in children if self._resolve_tree(c)]
            if not options:
                self.logger.warning("Empty alternate1 group. Using 'empty_prompt'.", extra={'error_code': 'ALT1_002'})
                schedules = [[steps, prefix + "empty_prompt" + suffix]]
            else:
                option = options[0] if self.use_legacy else self.random_gen.choice(options)
                schedules = [[steps, prefix + option + suffix]]
        elif data == "alternate2":
            options = [self._resolve_tree(c, keep_spacing=True) for c in children if self._resolve_tree(c)]
            combined_options = []
            for option in options:
                if "_" in option:
                    combined_options.append(option)
                else:
                    suffix_opt = options[0].split("_")[-1] if "_" in options[0] else ""
                    combined_options.append(f"{option}_{suffix_opt}" if suffix_opt else option)
            if not combined_options:
                self.logger.warning("Empty alternate2 group. Using 'empty_prompt'.", extra={'error_code': 'ALT2_002'})
                schedules = [[steps, prefix + "empty_prompt" + suffix]]
            else:
                option = combined_options[0] if self.use_legacy else self.random_gen.choice(combined_options)
                schedules = [[steps, prefix + option + suffix]]
        elif data == "grouped":
            options = []
            for child in children:
                resolved = self._resolve_tree(child, keep_spacing=True).strip(" ,|")
                if resolved:
                    if "|" in resolved:
                        options.extend(resolved.split("|"))
                    else:
                        options.append(resolved)
            text = ", ".join(options)
            schedules = [[steps, prefix + text + suffix]]
        elif data == "sequence":
            text = self.transform(tree)
            schedules = [[steps, prefix + text + suffix]]
        elif data == "nested_sequence":
            elements = [self._resolve_tree(c, keep_spacing=True).strip(" ,~!;") for c in children if self._resolve_tree(c).strip(" ,~!;")]
            if self.use_legacy:
                text = elements[0] if elements else "empty_prompt"
            else:
                text = self.random_gen.choice(elements) if any("~" in str(c) for c in children) and elements else f"[{' | '.join(elements)}]"
            schedules = [[steps, prefix + text + suffix]]
        elif data == "top_level_sequence":
            if self.use_legacy:
                owner = self._resolve_tree(children[0], keep_spacing=True).strip()
                schedules = [[steps, prefix + owner + suffix]]
            else:
                text = self.transform(tree)
                schedules = [[steps, prefix + text + suffix]]
        elif data == "numbered":
            schedules = self._process_numbered(tree, steps, prefix, suffix)
        elif data == "and_rule":
            text = " and ".join(self._resolve_tree(c, keep_spacing=True) for c in children if self._resolve_tree(c))
            schedules = [[steps, prefix + text + suffix]]
        else:
            for i, child in enumerate(children):
                if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                    continue
                pre = "".join(self._resolve_tree(c, keep_spacing=True) for j, c in enumerate(children) if j < i and not (isinstance(c, lark.Token) and c.type == "WHITESPACE"))
                post = "".join(self._resolve_tree(c, keep_spacing=True) for j, c in enumerate(children) if j > i and not (isinstance(c, lark.Token) and c.type == "WHITESPACE"))
                schedules.extend(self.collect_steps(steps, child, use_scheduling, prefix + pre, post + suffix, depth + 1))

        return schedules or [[steps, prefix + self._resolve_tree(tree, keep_spacing=True) + suffix]]

    def _process_scheduled(self, tree: lark.Tree, steps: int, prefix: str, suffix: str, depth: int, use_scheduling: bool) -> list[list[int | str]]:
        prompts = [p for p in tree.children if not (isinstance(p, lark.Token) and p.type in ("NUMBER", "reverse_flag")) and not (isinstance(p, lark.Tree) and p.data == "step_range_list")]
        number_node = next((p for p in tree.children if isinstance(p, lark.Token) and p.type == "NUMBER"), None)
        step_range_list = next((p for p in tree.children if isinstance(p, lark.Tree) and p.data == "step_range_list"), None)
        reverse_flag = any(p for p in tree.children if isinstance(p, lark.Token) and p.type == "reverse_flag")

        try:
            weight = float(number_node) if number_node else 1.0
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid weight '{number_node}' in scheduled prompt. Defaulting to 1.0.", extra={'error_code': 'WGT001'})
            weight = 1.0

        boundary = int(weight * steps) if weight <= 1.0 else int(weight)
        boundary = max(1, min(boundary, steps))
        schedules = []

        if step_range_list:
            ranges = []
            for sr in step_range_list.children:
                if isinstance(sr, lark.Tree) and sr.data == "step_range":
                    start, end = sr.children
                    start_val = float(start.value.rstrip('%'))
                    end_val = float(end.value.rstrip('%'))
                    if '%' in str(start) or '%' in str(end):
                        start_val = start_val / 100.0 * steps
                        end_val = end_val / 100.0 * steps
                    if start_val >= end_val or start_val < 0 or end_val <= 0:
                        self.logger.warning(f"Invalid step range {start_val}-{end_val} in scheduled prompt. Skipping.", extra={'error_code': 'RNG001'})
                        continue
                    ranges.append((int(start_val), int(end_val)))
            if reverse_flag:
                ranges = [(steps - end, steps - start) for start, end in reversed(ranges)]
            for start, end in ranges:
                end = min(end, steps)
                if start < end:
                    for prompt in prompts:
                        right = self._resolve_tree(prompt, keep_spacing=True).strip()
                        schedules.append([end, prefix + right + suffix])
        else:
            step_size = boundary / max(1, len(prompts))
            current_step = 0
            for i, prompt in enumerate(prompts):
                next_boundary = int((i + 1) * step_size) if i < len(prompts) - 1 else steps
                if next_boundary > current_step:
                    right = self._resolve_tree(prompt, keep_spacing=True).strip()
                    schedules.append([next_boundary, prefix + right + suffix])
                    current_step = next_boundary

        if reverse_flag and not step_range_list:
            schedules = [[steps - s[0], s[1]] for s in reversed(schedules) if s[0] <= steps and s[0] > 0]

        return schedules

    def _process_numbered(self, tree: lark.Tree, steps: int, prefix: str, suffix: str) -> list[list[int | str]]:
        quantity = int(tree.children[0])
        distinct = tree.children[1].value == "!" if len(tree.children) > 1 and isinstance(tree.children[1], lark.Token) else False
        target = tree.children[-1]
        options = []
        if isinstance(target, lark.Tree) and target.data in ("alternate", "alternate_distinct", "alternate1", "alternate2"):
            options = [self._resolve_tree(c, keep_spacing=True) for c in target.children if self._resolve_tree(c)]
        else:
            options = [self._resolve_tree(target, keep_spacing=True)] if isinstance(target, lark.Token) else [self._resolve_tree(c, keep_spacing=True) for c in target.children]

        if distinct and quantity > len(options):
            self.logger.warning(f"Quantity {quantity} exceeds available options {len(options)} in distinct alternate. Using all options.", extra={'error_code': 'ALT_DIST001'})
            quantity = len(options)

        if self.use_legacy:
            selected = [options[0]] * quantity if options else ["empty_prompt"] * quantity
        else:
            if distinct:
                selected = self.random_gen.sample(options, min(quantity, len(options)))
            else:
                selected = self.random_gen.choices(options, k=quantity)

        return [[steps, prefix + ", ".join(selected) + suffix]]

    def _resolve_tree(self, tree: lark.Tree | lark.Token, keep_spacing: bool = False) -> str:
        if isinstance(tree, lark.Tree):
            children = []
            for child in tree.children:
                if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                    if keep_spacing:
                        children.append(" ")
                    continue
                if isinstance(child, lark.Token) and child.type in ("NUMBER", "COLON"):
                    continue
                children.append(self._resolve_tree(child, keep_spacing))
            result = "".join(str(c) for c in children if c)
            return re.sub(r"[\s\u2028\u2029]+", " ", result).strip() if keep_spacing else result.strip()
        return str(tree).strip()

    def start(self, args):
        return "".join(str(arg) for arg in args if arg)

    def prompt(self, args):
        return "".join(str(arg) for arg in args if arg)

    def plain(self, args):
        return args[0].value

    def compound(self, args):
        return "_".join(str(arg) for arg in args)

    def and_rule(self, args):
        return " and ".join(self._resolve_tree(arg, keep_spacing=True) for arg in args if self._resolve_tree(arg))

    def grouped(self, args):
        options = []
        for arg in args:
            resolved = self._resolve_tree(arg, keep_spacing=True).strip(" ,|")
            if resolved:
                if "|" in resolved:
                    options.extend(resolved.split("|"))
                else:
                    options.append(resolved)
        return ", ".join(options)

    def alternate(self, args):
        args = [self._resolve_tree(arg, keep_spacing=True) for arg in args if self._resolve_tree(arg) or self._resolve_tree(arg) == ""]
        if self.use_legacy:
            return args[0] if args else "empty_prompt"
        return args[(self.current_step - 1) % len(args)] if args else "empty_prompt"

    def alternate_distinct(self, args):
        options = [self._resolve_tree(arg, keep_spacing=True) for arg in args if self._resolve_tree(arg)]
        if not options:
            self.logger.warning("Empty alternate_distinct group. Using 'empty_prompt'.", extra={'error_code': 'ALT_DIST002'})
            return "empty_prompt"
        return options[0] if self.use_legacy else self.random_gen.choice(options)

    def alternate1(self, args):
        options = [self._resolve_tree(arg, keep_spacing=True) for arg in args if self._resolve_tree(arg)]
        if not options:
            self.logger.warning("Empty alternate1 group. Using 'empty_prompt'.", extra={'error_code': 'ALT1_002'})
            return "empty_prompt"
        return options[0] if self.use_legacy else self.random_gen.choice(options)

    def alternate2(self, args):
        options = [self._resolve_tree(c, keep_spacing=True) for c in args if self._resolve_tree(c)]
        combined_options = []
        for option in options:
            if "_" in option:
                combined_options.append(option)
            else:
                suffix_opt = options[0].split("_")[-1] if "_" in options[0] else ""
                combined_options.append(f"{option}_{suffix_opt}" if suffix_opt else option)
        if not combined_options:
            self.logger.warning("Empty alternate2 group. Using 'empty_prompt'.", extra={'error_code': 'ALT2_002'})
            return "empty_prompt"
        return combined_options[0] if self.use_legacy else self.random_gen.choice(combined_options)

    def numbered(self, args):
        quantity = int(args[0])
        distinct = args[1].value == "!" if len(args) > 1 and isinstance(args[1], lark.Token) else False
        target = args[-1]
        options = []
        if isinstance(target, lark.Tree) and target.data in ("alternate", "alternate_distinct", "alternate1", "alternate2"):
            options = [self._resolve_tree(c, keep_spacing=True) for c in target.children if self._resolve_tree(c)]
        else:
            options = [self._resolve_tree(target, keep_spacing=True)] if isinstance(target, lark.Token) else [self._resolve_tree(c, keep_spacing=True) for c in target.children]
        if distinct and quantity > len(options):
            self.logger.warning(f"Quantity {quantity} exceeds options {len(options)}. Using all options.", extra={'error_code': 'NUM001'})
            quantity = len(options)
        if self.use_legacy:
            selected = [options[0]] * quantity if options else ["empty_prompt"] * quantity
        else:
            if distinct:
                selected = self.random_gen.sample(options, min(quantity, len(options)))
            else:
                selected = self.random_gen.choices(options, k=quantity)
        return ", ".join(selected)

    def sequence(self, args, parent=None):
        owner = self._resolve_tree(args[0], keep_spacing=True) if parent is None else parent
        descriptors = [self._resolve_tree(arg, keep_spacing=True).strip(" ,~!;") for arg in args[1:] if self._resolve_tree(arg).strip(" ,~!;")]
        return f"{owner}: {', '.join(descriptors)}"

    def nested_sequence(self, args):
        elements = [self._resolve_tree(arg, keep_spacing=True).strip(" ,~!;") for arg in args if self._resolve_tree(arg).strip(" ,~!;")]
        if self.use_legacy:
            return elements[0] if elements else "empty_prompt"
        if any("~" in str(arg) for arg in args):
            return self.random_gen.choice(elements) if elements else "empty_prompt"
        return f"[{' | '.join(elements)}]"

    def top_level_sequence(self, args):
        if isinstance(args, lark.Tree) and args.data == "top_level_sequence":
            args = args.children

        if not args:
            return ""

        if self.use_legacy:
            return self._resolve_tree(args[0], keep_spacing=True).strip()

        owner = self._resolve_tree(args[0], keep_spacing=True).strip()
        sequences = []
        trailing_text = []

        for child in args[1:]:
            if isinstance(child, lark.Tree) and child.data == "sequence":
                sequences.append(self.sequence(child.children, owner))
            elif isinstance(child, str) and child.strip() == "!!":
                continue
            else:
                t = self._resolve_tree(child, keep_spacing=True).strip(" ,")
                if t:
                    trailing_text.append(t)

        text = f"{owner} -> {', '.join(sequences)}"
        if trailing_text:
            text += f", {', '.join(trailing_text)}"
        return text

    def emphasized(self, args):
        prompt = self._resolve_tree(args[0], keep_spacing=True)
        try:
            if len(args) > 1 and isinstance(args[1], lark.Token) and args[1].type == "NUMBER":
                weight = float(args[1])
                if weight < 0:
                    self.logger.warning(f"Negative weight in emphasized prompt: {weight}. Defaulting to 1.0.", extra={'error_code': 'WGT002'})
                    weight = 1.0
            else:
                weight = 1 / 1.1 if len(args) == 1 and isinstance(args[0], lark.Tree) and args[0].data != "scheduled" else 1.1
        except ValueError:
            self.logger.warning(f"Invalid weight in emphasized prompt: {args[1] if len(args) > 1 else 'None'}. Defaulting to 1.0.", extra={'error_code': 'WGT002'})
            weight = 1.0
        return f"({prompt}:{weight})"

def get_learned_conditioning_prompt_schedules(prompts: list[str], base_steps: int, hires_steps: int | None = None, use_old_scheduling: bool = False, seed: int | None = 42, verbose: bool = False, telemetry: bool = False, log_level: str = 'WARNING') -> list[list[list[int | str | PromptParsingFailed]]]:
    logging.getLogger().setLevel(log_level)
    if seed is not None:
        random.seed(seed)

    steps = hires_steps if hires_steps is not None and not use_old_scheduling else base_steps
    use_scheduling = not hires_steps or use_old_scheduling

    @lru_cache(maxsize=CACHE_SIZE)
    def get_schedule(prompt: str, steps: int, use_scheduling: bool, seed: int | None, use_legacy: bool) -> list[list[int | str | PromptParsingFailed]]:
        if seed is not None:
            random.seed(seed)
        if " AND " in prompt:
            subprompts = re.split(r"\s*AND\s*", prompt)
            result = []
            for subprompt in subprompts:
                subprompt = subprompt.strip()
                try:
                    validate_prompt(subprompt)
                    tree = schedule_parser.parse(subprompt)
                    schedules = collect_steps(steps, tree, "", "", 0, verbose, use_scheduling, seed, use_old_scheduling)
                    result.extend(schedules)
                except lark.exceptions.UnexpectedInput as e:
                    error_code = 'PRS001'
                    reason = f"Syntax error at position {e.pos_in_stream}: {str(e)}"
                    suggestion = "Ensure balanced brackets/parentheses and valid syntax."
                    logging.warning(
                        f"Returning subprompt '{subprompt}' as-is due to parsing error. {reason}\nSuggestion: {suggestion}",
                        extra={'error_code': error_code}
                    )
                    result.append([steps, PromptParsingFailed(subprompt, reason, error_code)])
                except lark.exceptions.LarkError as e:
                    error_code = 'PRS002'
                    reason = f"General parsing error: {str(e)}"
                    suggestion = "Verify prompt syntax."
                    logging.warning(
                        f"Returning subprompt '{subprompt}' as-is due to parsing error. {reason}\nSuggestion: {suggestion}",
                        extra={'error_code': error_code}
                    )
                    result.append([steps, PromptParsingFailed(subprompt, reason, error_code)])
            seen = set()
            final_result = []
            for step, text in sorted(result, key=lambda x: x[0]):
                text_str = str(text).strip()
                text_str = re.sub(r",\s*$", "", text_str)
                text_out = PromptParsingFailed(text_str, "Aggregated from subprompts", "AGG001") if isinstance(text, PromptParsingFailed) else text_str
                if step <= steps and step > 0 and (step, text_str) not in seen:
                    final_result.append([step, text_out])
                    seen.add((step, text_str))
            return final_result

        try:
            validate_prompt(prompt)
            tree = schedule_parser.parse(prompt)
            if verbose:
                logging.info(f"Parsed prompt: {prompt}\nParse tree:\n{tree.pretty()}")
            schedules = collect_steps(steps, tree, "", "", 0, verbose, use_scheduling, seed, use_old_scheduling)
            if telemetry:
                logging.info(f"Prompt: {prompt}, Schedule: {schedules}")
            seen = set()
            result = []
            for step, text in sorted(schedules, key=lambda x: x[0]):
                text = re.sub(r"\s+", " ", text.strip())
                if step <= steps and (step, text) not in seen:
                    result.append([step, text])
                    seen.add((step, text))
            return result
        except lark.exceptions.UnexpectedInput as e:
            error_code = 'PRS003'
            reason = f"Syntax error at position {e.pos_in_stream}: {str(e)}"
            suggestion = "Ensure balanced brackets/parentheses and valid syntax."
            logging.warning(
                f"Returning prompt '{prompt}' as-is due to parsing error. {reason}\nSuggestion: {suggestion}",
                extra={'error_code': error_code}
            )
            return [[steps, PromptParsingFailed(prompt.strip(), reason, error_code)]]
        except lark.exceptions.LarkError as e:
            error_code = 'PRS004'
            reason = f"General parsing error: {str(e)}"
            suggestion = "Verify prompt syntax."
            logging.warning(
                f"Returning prompt '{prompt}' as-is due to parsing error. {reason}\nSuggestion: {suggestion}",
                extra={'error_code': error_code}
            )
            return [[steps, PromptParsingFailed(prompt.strip(), reason, error_code)]]
        except Exception as e:
            error_code = 'UNK001'
            reason = f"Unexpected error: {str(e)}"
            suggestion = "Check for invalid characters or complex syntax issues."
            logging.error(
                f"Unexpected error parsing prompt '{prompt}'. Returning as-is. {reason}\nSuggestion: {suggestion}",
                extra={'error_code': error_code}
            )
            return [[steps, PromptParsingFailed(prompt.strip(), reason, error_code)]]

    return [get_schedule(prompt, steps, use_scheduling, seed, use_old_scheduling) for prompt in set(prompts)]

# Rest of the original file unchanged
ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])

class SdConditioning(list):
    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None, copy_from=None):
        super().__init__()
        self.extend(prompts)
        if copy_from is None:
            copy_from = prompts
        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, 'is_negative_prompt', False)
        self.width = width or getattr(copy_from, 'width', None)
        self.height = height or getattr(copy_from, 'height', None)

def get_learned_conditioning(model, prompts: SdConditioning | list[str], steps, hires_steps=None, use_old_scheduling=False, seed: int | None = 42):
    res = []
    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps, hires_steps, use_old_scheduling, seed=seed)
    cache = {}
    for prompt, prompt_schedule in zip(prompts, prompt_schedules):
        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue
        texts = SdConditioning([str(x[1]) for x in prompt_schedule], copy_from=prompts)
        conds = model.get_learned_conditioning(texts)
        cond_schedule = []
        for i, (end_at_step, text) in enumerate(prompt_schedule):
            if isinstance(text, PromptParsingFailed):
                logging.warning(
                    f"Using unparsed prompt '{text.prompt}' in conditioning due to {text.reason}",
                    extra={'error_code': text.error_code}
                )
            if isinstance(conds, dict):
                cond = {k: v[i] for k, v in conds.items()}
            else:
                cond = conds[i]
            cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))
        cache[prompt] = cond_schedule
        res.append(cond_schedule)
    return res

def get_multicond_learned_conditioning(model, prompts: SdConditioning | list[str], steps, hires_steps=None, use_old_scheduling=False, seed: int | None = 42):
    if seed is not None:
        random.seed(seed)
    
    steps = hires_steps if hires_steps is not None and not use_old_scheduling else steps
    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps, hires_steps, use_old_scheduling, seed=seed)
    conds_list, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)
    
    schedule_steps = set()
    for schedule in prompt_schedules:
        schedule_steps.update(step for step, _ in schedule)
    if len(schedule_steps) > 1 and not all(len(schedule) == len(prompt_schedules[0]) for schedule in prompt_schedules):
        logging.error(
            "Incompatible schedules: subprompts have different step boundaries.",
            extra={'error_code': 'SCH001'}
        )
        raise ValueError("Incompatible schedules: subprompts have different step boundaries.")
    
    cache = {}
    res = []
    for prompt, schedule, conds in zip(prompts, prompt_schedules, conds_list):
        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue
        
        texts = SdConditioning([str(x[1]) for x in schedule], copy_from=prompts)
        try:
            model_conds = model.get_learned_conditioning(texts)
        except Exception as e:
            logging.error(
                f"Failed to condition prompt '{prompt}': {str(e)}",
                extra={'error_code': 'CND001'}
            )
            raise ValueError(f"Conditioning failed for prompt '{prompt}'")
        
        prompt_schedule = []
        for index, weight in conds:
            cond_schedule = []
            for i, (end_at_step, text) in enumerate(schedule):
                try:
                    end_at_step = int(end_at_step)
                except ValueError:
                    logging.error(
                        f"Invalid end_at_step '{end_at_step}' for prompt '{prompt}'",
                        extra={'error_code': 'STP001'}
                    )
                    raise ValueError(f"Invalid step boundary in schedule for prompt '{prompt}'")
                
                if isinstance(text, PromptParsingFailed):
                    logging.warning(
                        f"Using unparsed prompt '{text.prompt}' in conditioning due to {text.reason}",
                        extra={'error_code': text.error_code}
                    )
                
                if isinstance(model_conds, dict):
                    cond = {k: v[i] for k, v in model_conds.items()}
                else:
                    cond = model_conds[i]
                cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))
            prompt_schedule.append(ComposableScheduledPromptConditioning(cond_schedule, weight))
        
        shape = model_conds.shape if not isinstance(model_conds, dict) else model_conds['crossattn'].shape
        cache[prompt] = MulticondLearnedConditioning(shape, [prompt_schedule])
        res.append(cache[prompt])
    
    shapes = {r.shape for r in res}
    if len(shapes) > 1:
        logging.error(
            "Inconsistent tensor shapes across subprompts.",
            extra={'error_code': 'TSH001'}
        )
        raise ValueError("Inconsistent tensor shapes across subprompts.")
    
    return res[0] if len(res) == 1 else MulticondLearnedConditioning(shapes.pop(), res)

re_AND = re.compile(r"\bAND\b")
re_weight = re.compile(r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")

def get_multicond_prompt_list(prompts: SdConditioning | list[str]):
    res_indexes = []
    prompt_indexes = {}
    prompt_flat_list = SdConditioning(prompts)
    prompt_flat_list.clear()
    for prompt in prompts:
        subprompts = re_AND.split(prompt)
        indexes = []
        for subprompt in subprompts:
            match = re_weight.search(subprompt)
            text, weight = match.groups() if match is not None else (subprompt, 1.0)
            try:
                weight = float(weight) if weight is not None else 1.0
            except ValueError:
                logging.error(
                    f"Invalid weight '{weight}' in prompt '{subprompt}'. Defaulting to 1.0.",
                    extra={'error_code': 'WGT004'}
                )
                weight = 1.0
            index = prompt_indexes.get(text, None)
            if index is None:
                index = len(prompt_flat_list)
                prompt_flat_list.append(text)
                prompt_indexes[text] = index
            indexes.append((index, weight))
        res_indexes.append(indexes)
    return res_indexes, prompt_flat_list, prompt_indexes

class ComposableScheduledPromptConditioning:
    def __init__(self, schedules, weight=1.0):
        self.schedules: list[ScheduledPromptConditioning] = schedules
        self.weight: float = weight

class MulticondLearnedConditioning:
    def __init__(self, shape, batch):
        self.shape: tuple = shape
        self.batch: list[list[ComposableScheduledPromptConditioning]] = batch

def reconstruct_cond_batch(c: list[list[ScheduledPromptConditioning]], current_step):
    param = c[0][0].cond
    is_dict = isinstance(param, dict)
    if is_dict:
        dict_cond = param
        res = {k: torch.zeros((len(c),) + param.shape, device=param.device or 'cpu', dtype=param.dtype or torch.float32) for k, param in dict_cond.items()}
        res = DictWithShape(res, (len(c),) + dict_cond['crossattn'].shape)
    else:
        res = torch.zeros((len(c),) + param.shape, device=param.device or 'cpu', dtype=param.dtype or torch.float32)
    for i, cond_schedule in enumerate(c):
        target_index = 0
        for current, entry in enumerate(cond_schedule):
            if current_step <= entry.end_at_step:
                target_index = current
                break
        if is_dict:
            for k, param in cond_schedule[target_index].cond.items():
                res[k][i] = param
        else:
            res[i] = cond_schedule[target_index].cond
    return res

def stack_conds(tensors):
    token_count = max([x.shape[0] for x in tensors])
    for i in range(len(tensors)):
        if tensors[i].shape[0] != token_count:
            last_vector = tensors[i][-1:]
            last_vector_repeated = last_vector.repeat([token_count - tensors[i].shape[0], 1])
            tensors[i] = torch.vstack([tensors[i], last_vector_repeated])
    return torch.stack(tensors)

def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step):
    param = c.batch[0][0].schedules[0].cond
    tensors = []
    conds_list = []
    for composable_prompts in c.batch:
        conds_for_batch = []
        for composable_prompt in composable_prompts:
            target_index = 0
            for current, entry in enumerate(composable_prompt.schedules):
                try:
                    end_at_step = int(entry.end_at_step)
                except ValueError:
                    logging.error(
                        f"Invalid end_at_step value: {entry.end_at_step}, expected an integer.",
                        extra={'error_code': 'STP002'}
                    )
                    end_at_step = current_step
                if current_step <= end_at_step:
                    target_index = current
                    break
            conds_for_batch.append((len(tensors), composable_prompt.weight))
            tensors.append(composable_prompt.schedules[target_index].cond)
        conds_list.append(conds_for_batch)
    if isinstance(tensors[0], dict):
        keys = list(tensors[0].keys())
        stacked = {k: stack_conds([x[k] for x in tensors]) for k in keys}
        stacked = DictWithShape(stacked, stacked['crossattn'].shape)
    else:
        stacked = stack_conds(tensors).to(device=param.device or 'cpu', dtype=param.dtype or torch.float32)
    return conds_list, stacked

re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([-+]?\d*\.?\d+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

def parse_prompt_attention(text: str | PromptParsingFailed) -> list[list[str | float]]:
    if isinstance(text, PromptParsingFailed):
        logging.warning(f"Received unparsed prompt: {text.prompt}, reason: {text.reason}", extra={'error_code': text.error_code})
        return [[text.prompt, 1.0]]
    
    res = []
    round_brackets = []
    square_brackets = []
    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position: int, multiplier: float):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)
        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            try:
                weight_value = float(weight)
            except ValueError:
                logging.error(
                    f"Invalid weight '{weight}' in prompt '{text}'. Defaulting to 1.0.",
                    extra={'error_code': 'WGT005'}
                )
                weight_value = 1.0
            multiply_range(round_brackets.pop(), weight_value)
        elif text == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part.strip(), 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)
    
    if not res:
        res = [["", 1.0]]
    
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    return res

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
else:
    import torch