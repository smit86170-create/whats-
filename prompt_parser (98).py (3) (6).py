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
!start: (prompt | /[][():|]/+)*
prompt: (scheduled | emphasized | grouped | alternate | alternate1 | alternate2 | top_level_sequence | sequence | compound | numbered | and_rule | plain | WHITESPACE)*

!emphasized: "(" prompt ")" 
           | "(" prompt ":" NUMBER ")" 
scheduled: "[" prompt (":" prompt)* ":" NUMBER "]" (step_range_list | reverse_flag | step_range_list reverse_flag)?
reverse_flag: "reverse" | "r"
step_range_list: step_range ("," step_range)*
step_range: NUMBER "-" NUMBER | NUMBER "%" "-" NUMBER "%"

alternate: "[" prompt ("|" prompt)* "]"
!alternate_distinct: "[" prompt ("|" prompt)* "]!"
alternate1: (prompt) "|" (prompt)+
alternate2: (plain | compound) ("|" (plain | compound))+

grouped: "{" ((NUMBER_Q | prompt | sequence | grouped) ("," | "|")?)+ "}"
top_level_sequence: prompt ("::" sequence)+ "!!" ("," plain)?
sequence: prompt "::" prompt ("," | WHITESPACE)* ("!" | ";")
nested_sequence: "::" prompt ("," | WHITESPACE)* ("!" | ";")

compound: /[a-zA-Z0-9]+(_[a-zA-Z0-9]+)*/
numbered: NUMBER_Q ("!" | "_")? (grouped | sequence | compound | and_rule | plain | alternate1 | alternate2)
and_rule: (plain | compound) ("&" (plain | compound))+
WHITESPACE: /\s+/
plain: /([^\\\[\]()&]|\\.)+/

%import common.SIGNED_NUMBER -> NUMBER
%import common.INT -> NUMBER_Q
""", start="start")

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
             pass #validate_prompt(subprompt.strip())

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
                    children.append(" ")  # Preserve single space
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
        return hashlib.md5((tree.data + ''.join(hash_tree(c) for c in tree.children)).encode()).hexdigest()
    return hashlib.md5(str(tree).encode()).hexdigest()

@lru_cache(maxsize=CACHE_SIZE)
def collect_steps(steps: int, tree: lark.Tree | lark.Token, prefix: str = "", suffix: str = "", depth: int = 0, verbose: bool = False, use_scheduling: bool = True, seed: int | None = 42) -> list[list[int | str]]:
    """Collect schedules for a prompt parse tree, optimized with caching."""
    if seed is not None:
        random.seed(seed)
    
    schedules = []
    indent = "  " * depth
    if verbose:
        logging.info(f"{indent}Processing: {tree}")

    if isinstance(tree, str):
        schedules = [[steps, prefix + tree + suffix]]
    elif isinstance(tree, lark.Tree):
        schedules = _process_tree(tree, steps, prefix, suffix, depth, verbose, use_scheduling, seed)
    
    if verbose:
        logging.info(f"{indent}Schedules: {schedules}")
    return schedules

def _process_tree(tree: lark.Tree, steps: int, prefix: str, suffix: str, depth: int, verbose: bool, use_scheduling: bool, seed: int | None) -> list[list[int | str]]:
    """Helper function to process a parse tree node."""
    schedules = []
    data = tree.data
    children = tree.children

    if data == "plain":
        schedules = [[steps, prefix + resolve_tree(tree, keep_spacing=True) + suffix]]
    elif data == "top_level_sequence":
        transformer = ScheduleTransformer(steps, 1, seed)
        text = transformer.transform(tree)
        schedules = [[steps, prefix + text + suffix]]
    elif data == "scheduled" and use_scheduling:
        schedules = _process_scheduled(tree, steps, prefix, suffix, depth, verbose, use_scheduling, seed)
    elif data == "alternate":
        options = [resolve_tree(c, keep_spacing=True) for c in children if resolve_tree(c) or resolve_tree(c) == ""]
        schedules = [[step, prefix + (options[(step - 1) % len(options)] if options else "empty_prompt") + suffix] for step in range(1, steps + 1) if options]
    elif data == "alternate_distinct":
        options = [resolve_tree(c, keep_spacing=True) for c in children if resolve_tree(c)]
        if not options:
            # logging.warning("Empty alternate_distinct group. Using 'empty_prompt'.", extra={'error_code': 'ALT_DIST002'})
            schedules = [[steps, prefix + "empty_prompt" + suffix]]
        else:
            random.seed(seed)
            option = random.choice(options)
            schedules = [[steps, prefix + option + suffix]]
    elif data in ("alternate1", "alternate2"):
        options = [resolve_tree(c, keep_spacing=True) for c in children if resolve_tree(c)]
        random.seed(seed)
        option = random.choice(options) if options else "empty_prompt"
        schedules = [[steps, prefix + option + suffix]]
    elif data == "grouped":
        options = [resolve_tree(c, keep_spacing=True).strip(" ,|") for c in children if resolve_tree(c).strip(" ,|")]
        text = ", ".join(options)
        schedules = [[steps, prefix + text + suffix]]
    elif data == "sequence":
        transformer = ScheduleTransformer(steps, 1, seed)
        text = transformer.transform(tree)
        schedules = [[steps, prefix + text + suffix]]
    elif data == "nested_sequence":
        elements = [resolve_tree(c, keep_spacing=True).strip(" ,~!;") for c in children if resolve_tree(c).strip(" ,~!;")]
        random.seed(seed)
        text = random.choice(elements) if any("~" in str(c) for c in children) and elements else f"[{' | '.join(elements)}]"
        schedules = [[steps, prefix + text + suffix]]
    elif data == "numbered":
        schedules = _process_numbered(tree, steps, prefix, suffix, seed)
    elif data == "and_rule":
        text = " and ".join(resolve_tree(c, keep_spacing=True) for c in children if resolve_tree(c))
        schedules = [[steps, prefix + text + suffix]]
    else:
        for i, child in enumerate(children):
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                continue
            pre = "".join(resolve_tree(c, keep_spacing=True) for j, c in enumerate(children) if j < i and not (isinstance(c, lark.Token) and c.type == "WHITESPACE"))
            post = "".join(resolve_tree(c, keep_spacing=True) for j, c in enumerate(children) if j > i and not (isinstance(c, lark.Token) and c.type == "WHITESPACE"))
            schedules.extend(collect_steps(steps, child, prefix + pre, post + suffix, depth + 1, verbose, use_scheduling, seed))
    
    return schedules or [[steps, prefix + resolve_tree(tree, keep_spacing=True) + suffix]]

def _process_scheduled(tree: lark.Tree, steps: int, prefix: str, suffix: str, depth: int, verbose: bool, use_scheduling: bool, seed: int | None) -> list[list[int | str]]:
    """Helper function to process scheduled prompts."""
    prompts = [p for p in tree.children if not (isinstance(p, lark.Token) and p.type in ("NUMBER", "reverse_flag")) and not (isinstance(p, lark.Tree) and p.data == "step_range_list")]
    number_node = next((p for p in tree.children if isinstance(p, lark.Token) and p.type == "NUMBER"), None)
    step_range_list = next((p for p in tree.children if isinstance(p, lark.Tree) and p.data == "step_range_list"), None)
    reverse_flag = any(p for p in tree.children if isinstance(p, lark.Token) and p.type == "reverse_flag")

    try:
        weight = float(number_node) if number_node else 1.0
    except (ValueError, TypeError):
        logging.error(f"Invalid weight '{number_node}' in scheduled prompt. Defaulting to 1.0.", extra={'error_code': 'WGT001'})
        weight = 1.0
    
    boundary = int(weight * steps) if weight <= 1.0 else int(weight)
    boundary = max(1, min(boundary, steps))
    schedules = []

    if step_range_list:
        ranges = []
        for sr in step_range_list.children:
            if isinstance(sr, lark.Tree) and sr.data == "step_range":
                start, end = sr.children
                start_val = float(start.value)
                end_val = float(end.value.rstrip('%')) if isinstance(end, lark.Token) and end.type == "NUMBER" else float(end.value.rstrip('%'))
                if '%' in str(end):
                    start_val = start_val / 100.0 * steps
                    end_val = end_val / 100.0 * steps
                ranges.append((int(start_val), int(end_val)))
        if reverse_flag:
            ranges = [(steps - end, steps - start) for start, end in reversed(ranges)]
        for start, end in ranges:
            end = min(end, steps)
            if start < end:
                for prompt in prompts:
                    right = resolve_tree(prompt, keep_spacing=True).strip()
                    schedules.append([end, prefix + right + suffix])
    else:
        step_size = boundary / max(1, len(prompts))  # Distribute steps across prompts
        current_step = 0
        for i, prompt in enumerate(prompts):
            next_boundary = int((i + 1) * step_size) if i < len(prompts) - 1 else steps
            if next_boundary > current_step:
                right = resolve_tree(prompt, keep_spacing=True).strip()
                schedules.append([next_boundary, prefix + right + suffix])
                current_step = next_boundary

    if reverse_flag and not step_range_list:
        schedules = [[steps - s[0], s[1]] for s in reversed(schedules) if s[0] <= steps]
    
    return schedules

def _process_numbered(tree: lark.Tree, steps: int, prefix: str, suffix: str, seed: int | None) -> list[list[int | str]]:
    """Helper function to process numbered prompts."""
    quantity = int(tree.children[0])
    distinct = tree.children[1] == "!" if len(tree.children) > 1 and isinstance(tree.children[1], str) else False
    target = tree.children[-1]
    random.seed(seed)

    options = []
    if isinstance(target, lark.Tree) and target.data in ("alternate1", "alternate2"):
        options = [resolve_tree(c, keep_spacing=True) for c in target.children if resolve_tree(c)]
    else:
        if isinstance(target, lark.Token):
            options = [resolve_tree(target, keep_spacing=True)]
        elif isinstance(target, lark.Tree):
            options = [resolve_tree(c, keep_spacing=True) for c in target.children if resolve_tree(c)]

    if not options:
        logging.warning("Empty numbered group. Using 'empty_prompt'.", extra={'error_code': 'NUM001'})
        return [[steps, prefix + "empty_prompt" + suffix]]

    if distinct:
        if quantity > len(options):
            selected = random.sample(options, len(options)) + random.choices(options, k=quantity - len(options))
        else:
            selected = random.sample(options, quantity)
    else:
        selected = random.choices(options, k=quantity)

    return [[steps, prefix + ", ".join(selected) + suffix]]

class ScheduleTransformer(lark.Transformer):
    """Transformer for converting parse trees into prompt schedules."""
    def __init__(self, total_steps: int, current_step: int = 1, seed: int | None = 42):
        super().__init__()
        self.total_steps = total_steps
        self.current_step = current_step
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def start(self, args):
        return "".join(str(arg) for arg in args if arg)

    def prompt(self, args):
        return "".join(str(arg) for arg in args if arg)

    def plain(self, args):
        return args[0].value

    def compound(self, args):
        return "_".join(str(arg) for arg in args)

    def and_rule(self, args):
        return " and ".join(resolve_tree(arg, keep_spacing=True) for arg in args if resolve_tree(arg))

    def grouped(self, args):
        return ", ".join(resolve_tree(arg, keep_spacing=True) for arg in args if resolve_tree(arg).strip(" ,|"))

    def alternate(self, args):
        args = [resolve_tree(arg, keep_spacing=True) for arg in args if resolve_tree(arg) or resolve_tree(arg) == ""]
        return args[(self.current_step - 1) % len(args)] if args else "empty_prompt"

    def alternate1(self, args):
        options = [resolve_tree(arg, keep_spacing=True) for arg in args if resolve_tree(arg)]
        random.seed(self.seed)
        return random.choice(options) if options else "empty_prompt"

    def alternate2(self, args):
        options = [resolve_tree(arg, keep_spacing=True) for arg in args if resolve_tree(arg)]
        random.seed(self.seed)
        return random.choice(options) if options else "empty_prompt"

    def numbered(self, args):
        quantity = int(args[0])
        distinct = args[1] == "!" if len(args) > 1 and isinstance(args[1], str) else False
        target = args[-1]
        random.seed(self.seed)
        options = []
        if isinstance(target, lark.Tree) and target.data in ("alternate1", "alternate2"):
            options = [resolve_tree(c, keep_spacing=True) for c in target.children if resolve_tree(c)]
        else:
            options = [resolve_tree(target, keep_spacing=True)] if isinstance(target, lark.Token) else [resolve_tree(c, keep_spacing=True) for c in target.children]
        if distinct:
            if quantity > len(options):
                selected = random.sample(options, len(options)) + random.choices(options, k=quantity - len(options))
            else:
                selected = random.sample(options, quantity)
        else:
            selected = random.choices(options, k=quantity)
        return ", ".join(selected)

    def sequence(self, args, parent=None):
        owner = resolve_tree(args[0], keep_spacing=True) if parent is None else parent
        descriptors = [resolve_tree(arg, keep_spacing=True).strip(" ,~!;") for arg in args[1:] if resolve_tree(arg).strip(" ,~!;")]
        return f"{owner}: {', '.join(descriptors)}"

    def top_level_sequence(self, args):
        owner = resolve_tree(args[0], keep_spacing=True).strip()
        sequences = []
        trailing_text = []
        for child in args[1:]:
            if isinstance(child, lark.Tree) and child.data == "sequence":
                sequences.append(self.sequence(child.children, owner))
            elif isinstance(child, str) and child.strip() == "!!":
                continue
            else:
                t = resolve_tree(child, keep_spacing=True).strip(" ,")
                if t:
                    trailing_text.append(t)
        text = f"{owner} -> {', '.join(sequences)}"
        if trailing_text:
            text += f", {', '.join(trailing_text)}"
        return text

    def nested_sequence(self, args):
        elements = [resolve_tree(arg, keep_spacing=True).strip(" ,~!;") for arg in args if resolve_tree(arg).strip(" ,~!;")]
        random.seed(self.seed)
        if any("~" in str(arg) for arg in args):
            return random.choice(elements) if elements else "empty_prompt"
        return f"[{' | '.join(elements)}]"

    def emphasized(self, args):
        prompt = resolve_tree(args[0], keep_spacing=True)
        try:
            weight = float(args[1]) if len(args) > 1 and isinstance(args[1], lark.Token) and args[1].type == "NUMBER" else 1.1
        except ValueError:
            logging.error(f"Invalid weight in emphasized prompt: {args[1]}. Defaulting to 1.0.", extra={'error_code': 'WGT002'})
            weight = 1.0
        return f"({prompt}:{weight})"

    def scheduled(self, args):
        prompts = [arg for arg in args[:-1] if not isinstance(arg, lark.Token) or arg.type != "NUMBER"]
        number_node = args[-1]
        if isinstance(number_node, lark.Tree):
            number_node = resolve_tree(number_node, keep_spacing=True)
        try:
            weight = float(number_node)
        except ValueError:
            logging.error(f"Invalid weight '{number_node}' in scheduled prompt. Defaulting to 1.0.", extra={'error_code': 'WGT003'})
            weight = 1.0

        boundary = int(weight * self.total_steps) if weight <= 1.0 else int(weight)
        boundary = max(1, min(boundary, self.total_steps))

        if not prompts:
            return "empty_prompt"
        if len(prompts) == 1:
            return f"({resolve_tree(prompts[0], keep_spacing=True)}:{weight})" if self.current_step >= boundary else ""
        step_increment = boundary / max(1, len(prompts))
        for i, prompt in enumerate(prompts):
            step = min(self.total_steps, int(i * step_increment)) if i < len(prompts) - 1 else self.total_steps
            if self.current_step <= step:
                return f"({resolve_tree(prompt, keep_spacing=True)}:{weight})"
        return f"({resolve_tree(prompts[-1], keep_spacing=True)}:{weight})"

def get_learned_conditioning_prompt_schedules(
    prompts: list[str],
    base_steps: int,
    hires_steps: int | None = None,
    use_old_scheduling: bool = False,
    seed: int | None = 42,
    verbose: bool = False,
    telemetry: bool = False,
    log_level: str = 'WARNING',
    strict: bool = False
) -> list[list[list[int | str | PromptParsingFailed]]]:
    import logging
    logging.getLogger().setLevel(log_level)
    if seed is not None:
        random.seed(seed)

    steps = hires_steps if hires_steps is not None and not use_old_scheduling else base_steps
    use_scheduling = not hires_steps or use_old_scheduling

    _schedule_cache = {}

    # 👇 вложенная функция с правильным отступом
    @lru_cache(maxsize=CACHE_SIZE)
    def get_schedule(prompt: str, steps: int, use_scheduling: bool, seed: int | None) -> list[list[int | str]]:
        if seed is not None:
            random.seed(seed)

        if (prompt, steps, use_scheduling, seed) in _schedule_cache:
            return _schedule_cache[(prompt, steps, use_scheduling, seed)]

        # Если есть "AND"
        if " AND " in prompt:
            subprompts = re.split(r"\s*AND\s*", prompt)
            if len(subprompts) > 1:
                result = []
                for subprompt in subprompts:
                    try:
                        tree = schedule_parser.parse(subprompt.strip())
                        schedules = collect_steps(steps, tree, "", "", 0, verbose, use_scheduling, seed)
                        result.extend(schedules)
                    except Exception as e:
                        logging.warning(
                            f"Subprompt parse error: {e}. Using raw prompt: '{prompt}'",
                            extra={'error_code': 'PRS_RAW'}
                        )
                        return [[steps, prompt]]

                seen = set()
                final_result = []
                for step, text in sorted(result, key=lambda x: x[0]):
                    text_str = str(text).strip()
                    text_str = re.sub(r",\s*$", "", text_str)
                    if step <= steps and (step, text_str) not in seen:
                        final_result.append([step, text_str])
                        seen.add((step, text_str))

                _schedule_cache[(prompt, steps, use_scheduling, seed)] = final_result
                return final_result

        # Обычный парсинг
        try:
            tree = schedule_parser.parse(prompt)
            if verbose:
                logging.info(f"Parsed prompt: {prompt}\nParse tree:\n{tree.pretty()}")
            schedules = collect_steps(steps, tree, "", "", 0, verbose, use_scheduling, seed)
            if telemetry:
                logging.info(f"Prompt: {prompt}, Schedule: {schedules}")

            seen = set()
            result = []
            for step, text in sorted(schedules, key=lambda x: x[0]):
                text_str = re.sub(r"\s+", " ", str(text).strip())
                if step <= steps and (step, text_str) not in seen:
                    result.append([step, text_str])
                    seen.add((step, text_str))

            _schedule_cache[(prompt, steps, use_scheduling, seed)] = result
            return result

        except Exception as e:
            logging.warning(
                f"Prompt parse error: {e}. Using raw prompt: '{prompt}'",
                extra={'error_code': 'PRS_RAW'}
            )
            return [[steps, prompt]]

    # 👇 основной цикл по промптам
    result = []
    for prompt in prompts:
        schedule = get_schedule(prompt, steps, use_scheduling, seed)

        if strict and any(isinstance(s, list) and isinstance(s[1], str) and s[1] == prompt for s in schedule):
            raise ValueError(f"Failed to parse prompt strictly: '{prompt}'")

        result.append(schedule)

    return result

    
    

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

    if prompt_schedules is None:
        raise ValueError("Prompt schedules unexpectedly returned None")

    if not isinstance(prompt_schedules, list):
        raise TypeError(f"Expected prompt_schedules to be a list, got {type(prompt_schedules)}")

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
    """
    Converts a list of prompts into multi-conditioning schedules for Stable Diffusion.
    
    Args:
        model: The model used to generate conditioning tensors.
        prompts: List of prompt strings to process.
        steps: Total number of steps for scheduling.
        hires_steps: Optional high-resolution steps; overrides steps if provided.
        use_old_scheduling: If True, use older scheduling logic.
        seed: Random seed for reproducibility.
    
    Returns:
        MulticondLearnedConditioning: Object containing shape and batch of composable prompt schedules.
    
    Raises:
        ValueError: If schedules are incompatible or tensor shapes cannot be aligned.
    """
    if seed is not None:
        random.seed(seed)
    
    steps = hires_steps if hires_steps is not None and not use_old_scheduling else steps
    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps, hires_steps, use_old_scheduling, seed=seed)
    conds_list, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)
    
    # Validate schedule compatibility
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
        
        # Collect all texts for conditioning
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
    
    # Ensure consistent tensor shapes
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
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    If the input is a PromptParsingFailed object, returns the unparsed prompt with a default weight of 1.0.
    """
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