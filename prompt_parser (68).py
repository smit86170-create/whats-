from __future__ import annotations
import re
import lark
import random
import logging
from functools import lru_cache
import hashlib
from collections import namedtuple
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Updated grammar to fix compound and support all A1111 syntax
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
alternate1: (prompt) "|" (prompt)+
alternate2: (plain | compound) ("|" (plain | compound))+

grouped: "{" ((NUMBER_Q | prompt | sequence | grouped) ("," | "|")?)+ "}"
top_level_sequence: prompt ("::" sequence)+ "!!" ("," plain)?
sequence: prompt "::" prompt ("," | WHITESPACE)* ("!" | ";")
nested_sequence: "::" prompt ("," | WHITESPACE)* ("~" | "!")

compound: /[a-zA-Z0-9]+(_[a-zA-Z0-9]+)*/
numbered: NUMBER_Q ("!" | "_")? (grouped | sequence | compound | and_rule | plain)
and_rule: (plain | compound) ("&" (plain | compound))+
WHITESPACE: /\s+/
plain: /([^\\\[\]():|!_&]|\\.)+/

%import common.SIGNED_NUMBER -> NUMBER
%import common.INT -> NUMBER_Q
""", start="start")

# Validate prompt syntax
def validate_prompt(prompt):
    """Validate prompt syntax for critical errors like unbalanced brackets."""
    stack = []
    for char in prompt:
        if char in '([':
            stack.append(char)
        elif char == ')':
            if not stack or stack[-1] != '(':
                raise ValueError(f"Unbalanced parentheses in prompt: {prompt}")
            stack.pop()
        elif char == ']':
            if not stack or stack[-1] != '[':
                raise ValueError(f"Unbalanced brackets in prompt: {prompt}")
            stack.pop()
    if stack:
        raise ValueError(f"Unbalanced brackets in prompt: {prompt}")
    if '::' in prompt and not any(prompt.endswith(s) for s in ('!', '~', ';', '!!')) and ':::' not in prompt:
        logging.warning(f"Sequence in prompt '{prompt}' may not be properly terminated. Proceeding with parsing.")
    return True

# Resolve tree to string, minimizing extra spaces
def resolve_tree(tree):
    """Recursively resolve a tree node to its string representation, minimizing extra spaces."""
    if isinstance(tree, lark.Tree):
        children = []
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                continue
            elif isinstance(child, lark.Token) and child.type in ("NUMBER", "COLON"):
                continue
            else:
                children.append(resolve_tree(child))
        result = "".join(str(c) for c in children if c)
        result = re.sub(r"\b\s+\b", "", result)
        return result.strip()
    return str(tree).strip()

# Robust tree hashing for caching
def hash_tree(tree):
    """Generate a stable hash for a Lark tree or token."""
    if isinstance(tree, lark.Tree):
        return hashlib.md5((tree.data + ''.join(hash_tree(c) for c in tree.children)).encode()).hexdigest()
    return hashlib.md5(str(tree).encode()).hexdigest()

#@lru_cache(maxsize=1024)  # Keep commented as per request
def collect_steps(steps, tree, prefix, suffix, depth=0, verbose=False, use_scheduling=True, seed=None):
    """Collect schedules for a prompt parse tree with caching."""
    schedules = []
    indent = "  " * depth

    if verbose:
        logging.info(f"{indent}Processing: {tree}")

    if isinstance(tree, str):
        schedules = [[steps, prefix + tree + suffix]]
    elif isinstance(tree, lark.Tree):
        data = tree.data
        children = tree.children

        if data == "plain":
            schedules = [[steps, prefix + resolve_tree(tree) + suffix]]
        elif data == "top_level_sequence":
            transformer = ScheduleTransformer(steps, 1)
            text = transformer.transform(tree)
            schedules = [[steps, prefix + text + suffix]]
        elif data == "scheduled" and use_scheduling:
            prompts = [p for p in children if not (isinstance(p, lark.Token) and p.type in ("NUMBER", "reverse_flag")) and not (isinstance(p, lark.Tree) and p.data == "step_range_list")]
            number_node = next((p for p in children if isinstance(p, lark.Token) and p.type == "NUMBER"), None)
            step_range_list = next((p for p in children if isinstance(p, lark.Tree) and p.data == "step_range_list"), None)
            reverse_flag = any(p for p in children if isinstance(p, lark.Token) and p.type == "reverse_flag")

            try:
                weight = float(number_node) if number_node else 1.0
            except (ValueError, TypeError):
                logging.warning(f"Invalid weight '{number_node}' in scheduled prompt. Defaulting to 1.0.")
                weight = 1.0
            boundary = int(weight * steps) if weight <= 1.0 else int(weight)
            boundary = max(1, min(boundary, steps))

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
                        for i, prompt in enumerate(prompts):
                            if i == 0 and start > 0:
                                schedules.append([start, prefix + suffix])
                            child_schedules = collect_steps(steps, prompt, prefix, suffix, depth + 1, verbose, use_scheduling, seed)
                            for step, text in child_schedules:
                                if start <= step <= end:
                                    schedules.append([min(step, end), text])
            else:
                if len(prompts) == 1:
                    right = resolve_tree(prompts[0]).lstrip()
                    sep = "" if prefix.rstrip() and right and prefix.rstrip()[-1].isalnum() and right[0].isalnum() else " "
                    schedules.append([boundary, prefix + sep + suffix])
                    child_schedules = collect_steps(steps, prompts[0], prefix, suffix, depth + 1, verbose, use_scheduling, seed)
                    for step, text in child_schedules:
                        if step > boundary:
                            schedules.append([step, text])
                else:
                    step_size = boundary / max(1, len(prompts) - 1)
                    current_step = 0
                    for i, prompt in enumerate(prompts):
                        next_boundary = int((i + 1) * step_size) if i < len(prompts) - 1 else steps
                        if next_boundary > current_step:
                            child_schedules = collect_steps(steps, prompt, prefix, suffix, depth + 1, verbose, use_scheduling, seed)
                            for step, text in child_schedules:
                                if current_step < step <= next_boundary:
                                    schedules.append([min(step, next_boundary), text])
                            current_step = next_boundary
                if reverse_flag and not step_range_list:
                    schedules = [[steps - s[0], s[1]] for s in reversed(schedules) if s[0] <= steps]
        elif data == "alternate":
            options = [resolve_tree(c) for c in children if resolve_tree(c) or resolve_tree(c) == ""]
            if not options:
                schedules = [[steps, prefix + suffix]]
            else:
                for step in range(1, steps + 1):
                    option = options[(step - 1) % len(options)]
                    schedules.append([step, prefix + option + suffix])
                # Remove duplicates and limit to steps
                seen = set()
                unique_schedules = []
                for s in sorted(schedules, key=lambda x: x[0]):
                    if s[0] <= steps and (s[0], s[1]) not in seen:
                        unique_schedules.append(s)
                        seen.add((s[0], s[1]))
                schedules = unique_schedules
        elif data in ("alternate1", "alternate2"):
            options = [resolve_tree(c) for c in children if resolve_tree(c)]
            if not options:
                schedules = [[steps, prefix + suffix]]
            else:
                if seed is not None:
                    random.seed(seed)
                option = random.choice(options)
                schedules = [[steps, prefix + option + suffix]]
        elif data == "grouped":
            options = [resolve_tree(c).strip(" ,|") for c in children if resolve_tree(c).strip(" ,|")]
            text = ", ".join(options)
            schedules = [[steps, prefix + text + suffix]]
        elif data == "sequence":
            described = resolve_tree(children[0]).strip()
            descriptors = [resolve_tree(c).strip(" ,~!;") for c in children[1:] if resolve_tree(c).strip(" ,~!;")]
            text = f"{described}: {', '.join(descriptors)}"
            child_schedules = []
            for child in children:
                child_schedules.extend(collect_steps(steps, child, prefix, suffix, depth + 1, verbose, use_scheduling, seed))
            if child_schedules:
                schedules = child_schedules
            else:
                schedules = [[steps, prefix + text + suffix]]
        elif data == "nested_sequence":
            elements = [resolve_tree(c).strip(" ,~!;") for c in children if resolve_tree(c).strip(" ,~!;")]
            if any("~" in str(c) for c in children):
                if seed is not None:
                    random.seed(seed)
                text = random.choice(elements) if elements else ""
            else:
                text = f"[{' | '.join(elements)}]"
            schedules = [[steps, prefix + text + suffix]]
        elif data == "numbered":
            quantity = int(children[0])
            distinct = children[1] == "!" if len(children) > 1 and isinstance(children[1], str) else False
            target = children[-1]
            options = [resolve_tree(target)] if isinstance(target, lark.Token) else [resolve_tree(c) for c in target.children]
            if distinct:
                if quantity > len(options):
                    selected = random.sample(options, len(options)) + random.choices(options, k=quantity - len(options))
                else:
                    selected = random.sample(options, quantity)
            else:
                selected = random.choices(options, k=quantity)
            text = ", ".join(selected)
            schedules = [[steps, prefix + text + suffix]]
        elif data == "and_rule":
            text = " and ".join(resolve_tree(c) for c in children if resolve_tree(c))
            schedules = [[steps, prefix + text + suffix]]
        else:
            for i, child in enumerate(children):
                if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                    continue
                pre = "".join(resolve_tree(c) for j, c in enumerate(children) if j < i and not (isinstance(c, lark.Token) and c.type == "WHITESPACE"))
                post = "".join(resolve_tree(c) for j, c in enumerate(children) if j > i and not (isinstance(c, lark.Token) and c.type == "WHITESPACE"))
                child_schedules = collect_steps(steps, child, prefix + pre, post + suffix, depth + 1, verbose, use_scheduling, seed)
                schedules.extend(child_schedules)
            if not schedules:
                schedules.append([steps, prefix + resolve_tree(tree) + suffix])
    else:
        schedules = [[steps, prefix + resolve_tree(tree) + suffix]]

    if verbose:
        logging.info(f"{indent}Schedules: {schedules}")
    return schedules

def get_learned_conditioning_prompt_schedules(prompts, base_steps, hires_steps=None, use_old_scheduling=False, seed=None, verbose=False, telemetry=False):
    """
    Converts a list of prompts into schedules for Stable Diffusion.

    Supports complex scheduling, alternates, grouping, sequences, and multi-conditioning (e.g., 'a AND b').

    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10, seed=42)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a'], [10, 'ab']]
    >>> g("a [b:.5] c")
    [[5, 'ac'], [10, 'abc']]
    >>> g("a [b:c:1] d")
    [[1, 'abd'], [10, 'acd']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'abe'], [2, 'ace'], [10, 'ade']]
    >>> g("[fe|]male")
    [[1, 'female'], [2, 'male'], [3, 'female'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'female'], [8, 'male'], [9, 'female'], [10, 'male']]
    >>> g("character:::outfit::red_dress!, accessories::diamond_necklace!, appearance::ponytail, blonde hair, green eyes!!, dark background")
    [[10, 'character -> outfit: red_dress, accessories: diamond_necklace, appearance: ponytail, blonde hair, green eyes, dark background']]
    >>> g("3!_text1|text2")
    [[10, 'text1, text2, text1']]
    >>> g("text1 & text2")
    [[10, 'text1 and text2']]
    >>> g("a AND b")
    [[10, 'a'], [10, 'b']]
    """
    if seed is not None:
        random.seed(seed)

    if hires_steps is None or use_old_scheduling:
        steps = base_steps
        use_scheduling = True
    else:
        steps = hires_steps
        use_scheduling = False

    class ScheduleTransformer(lark.Transformer):
        def __init__(self, total_steps, current_step=1):
            super().__init__()
            self.total_steps = total_steps
            self.current_step = current_step

        def start(self, args):
            return "".join(str(arg) for arg in args if arg)

        def prompt(self, args):
            return "".join(str(arg) for arg in args if arg)

        def plain(self, args):
            return args[0].value

        def compound(self, args):
            return "_".join(str(arg) for arg in args)

        def and_rule(self, args):
            return " and ".join(resolve_tree(arg) for arg in args if resolve_tree(arg))

        def grouped(self, args):
            return ", ".join(resolve_tree(arg) for arg in args if resolve_tree(arg).strip(" ,|"))

        def alternate(self, args):
            args = [resolve_tree(arg) for arg in args if resolve_tree(arg) or resolve_tree(arg) == ""]
            return args[(self.current_step - 1) % len(args)] if args else ""

        def alternate1(self, args):
            options = [resolve_tree(arg) for arg in args if resolve_tree(arg)]
            if seed is not None:
                random.seed(seed)
            return random.choice(options) if options else ""

        def alternate2(self, args):
            options = [resolve_tree(arg) for arg in args if resolve_tree(arg)]
            if seed is not None:
                random.seed(seed)
            return random.choice(options) if options else ""

        def numbered(self, args):
            quantity = int(args[0])
            distinct = args[1] == "!" if len(args) > 1 and isinstance(args[1], str) else False
            target = args[-1]
            options = [resolve_tree(target)] if isinstance(target, lark.Token) else [resolve_tree(c) for c in target.children]
            if distinct:
                if quantity > len(options):
                    selected = random.sample(options, len(options)) + random.choices(options, k=quantity - len(options))
                else:
                    selected = random.sample(options, quantity)
            else:
                selected = random.choices(options, k=quantity)
            return ", ".join(selected)

        def sequence(self, args, parent=None):
            owner = resolve_tree(args[0]) if parent is None else parent
            descriptors = [resolve_tree(arg).strip(" ,~!;") for arg in args[1:] if resolve_tree(arg).strip(" ,~!;")]
            return f"{owner}: {', '.join(descriptors)}"

        def top_level_sequence(self, args):
            owner = resolve_tree(args[0]).strip()
            sequences = []
            trailing_text = []
            for child in args[1:]:
                if isinstance(child, lark.Tree) and child.data == "sequence":
                    sequences.append(self.sequence(child.children, owner))
                elif isinstance(child, str) and child.strip() == "!!":
                    continue
                else:
                    t = resolve_tree(child).strip(" ,")
                    if t:
                        trailing_text.append(t)
            text = f"{owner} -> {', '.join(sequences)}"
            if trailing_text:
                text += f", {', '.join(trailing_text)}"
            return text

        def nested_sequence(self, args):
            elements = [resolve_tree(arg).strip(" ,~!;") for arg in args if resolve_tree(arg).strip(" ,~!;")]
            if any("~" in str(arg) for arg in args):
                if seed is not None:
                    random.seed(seed)
                return random.choice(elements) if elements else ""
            return f"[{' | '.join(elements)}]"

        def emphasized(self, args):
            prompt = resolve_tree(args[0])
            weight = float(args[1]) if len(args) > 1 and isinstance(args[1], lark.Token) and args[1].type == "NUMBER" else 1.1
            return f"({prompt}:{weight})"

        def scheduled(self, args):
            prompts = [arg for arg in args[:-1] if not isinstance(arg, lark.Token) or arg.type != "NUMBER"]
            number_node = args[-1]
            if isinstance(number_node, lark.Tree):
                number_node = resolve_tree(number_node)
            try:
                weight = float(number_node)
            except ValueError:
                logging.warning(f"Invalid weight '{number_node}' in scheduled prompt. Defaulting to 1.0.")
                weight = 1.0

            boundary = int(weight * self.total_steps) if weight <= 1.0 else int(weight)
            boundary = max(1, min(boundary, self.total_steps))

            if not prompts:
                return ""
            if len(prompts) == 1:
                return f"({resolve_tree(prompts[0])}:{weight})" if self.current_step >= boundary else ""
            step_increment = boundary / max(1, len(prompts) - 1)
            for i, prompt in enumerate(prompts):
                step = min(self.total_steps, int(i * step_increment)) if i < len(prompts) - 1 else self.total_steps
                if self.current_step <= step:
                    return f"({resolve_tree(prompt)}:{weight})"
            return f"({resolve_tree(prompts[-1])}:{weight})"

    _schedule_cache = {}

    def get_schedule(prompt, steps, use_scheduling):
        if (prompt, steps, use_scheduling) in _schedule_cache:
            return _schedule_cache[(prompt, steps, use_scheduling)]

        try:
            validate_prompt(prompt)
            tree = schedule_parser.parse(prompt)
            if verbose:
                logging.info(f"Parsed prompt: {prompt}\nParse tree:\n{tree.pretty()}")
            schedules = collect_steps(steps, tree, "", "", 0, verbose, use_scheduling, seed)
            if telemetry:
                logging.info(f"Prompt: {prompt}, Schedule: {schedules}")
            # Remove duplicates and sort
            result = []
            seen = set()
            for step, text in sorted(schedules, key=lambda x: x[0]):
                text = re.sub(r"\s+", " ", text.strip())
                if step <= steps and (step, text) not in seen:
                    result.append([step, text])
                    seen.add((step, text))
            _schedule_cache[(prompt, steps, use_scheduling)] = result
            return result
        except lark.exceptions.LarkError as e:
            logging.warning(f"Parsing error for prompt '{prompt}': {str(e)}. Returning as-is.")
            return [[steps, re.sub(r",\s*$", "", prompt.strip())]]
        except Exception as e:
            logging.error(f"Unexpected error parsing prompt '{prompt}': {str(e)}")
            raise

    promptdict = {prompt: get_schedule(prompt, steps, use_scheduling) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]

# Rest of the original file unchanged (included for completeness)
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

def get_learned_conditioning(model, prompts: SdConditioning | list[str], steps, hires_steps=None, use_old_scheduling=False):
    res = []
    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps, hires_steps, use_old_scheduling)
    cache = {}
    for prompt, prompt_schedule in zip(prompts, prompt_schedules):
        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue
        texts = SdConditioning([x[1] for x in prompt_schedule], copy_from=prompts)
        conds = model.get_learned_conditioning(texts)
        cond_schedule = []
        for i, (end_at_step, _) in enumerate(prompt_schedule):
            if isinstance(conds, dict):
                cond = {k: v[i] for k, v in conds.items()}
            else:
                cond = conds[i]
            cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))
        cache[prompt] = cond_schedule
        res.append(cond_schedule)
    return res

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
            weight = float(weight) if weight is not None else 1.0
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

def get_multicond_learned_conditioning(model, prompts, steps, hires_steps=None, use_old_scheduling=False):
    res_indexes, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)
    learned_conditioning = get_learned_conditioning(model, prompt_flat_list, steps, hires_steps, use_old_scheduling)
    res = []
    for indexes in res_indexes:
        res.append([ComposableScheduledPromptConditioning(learned_conditioning[i], weight) for i, weight in indexes])
    return MulticondLearnedConditioning(shape=(len(prompts),), batch=res)

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
                    logging.error(f"Invalid end_at_step value: {entry.end_at_step}, expected an integer.")
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
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    """
    res = []
    round_brackets = []
    square_brackets = []
    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
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
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)
    if len(res) == 0:
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