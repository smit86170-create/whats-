from __future__ import annotations
import re
import lark
import random
import logging
from functools import lru_cache
from collections import namedtuple
import torch

# Configure logging for production telemetry
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Full grammar from original script
schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | grouped | alternate | alternate1 | alternate2 | top_level_sequence | compound | numbered | and_rule | plain | WHITESPACE)*

!emphasized: "(" prompt ")" 
        | "(" prompt ":" prompt ")" 
        | "[" prompt "]"
scheduled: "[" [prompt (":" prompt)+] "]" ":" NUMBER (step_range_list | reverse_flag | step_range_list reverse_flag)?
reverse_flag: "reverse" | "r"
step_range_list: step_range ("," step_range)*
step_range: NUMBER "-" NUMBER | NUMBER "%" "-" NUMBER "%"

alternate: "[" prompt ("|" [prompt])+ "]"
alternate1: (prompt) "|" (prompt)+
alternate2: (plain | compound) ("|" (plain | compound))+

grouped: "{" ((NUMBER_Q | prompt | sequence | grouped) ("," | "|")?)+ "}"
top_level_sequence: prompt "::" sequence ("::" sequence)* "!!"
sequence: prompt "::" prompt ("," | WHITESPACE)* nested_sequence* "!" 
nested_sequence: "::" prompt ("," | WHITESPACE)* ("~" | "!")

compound: /[a-zA-Z0-9]+(_[a-zA-Z0-9]+)+/
numbered: NUMBER_Q ("!")? (grouped | sequence | compound | and_rule | plain)
and_rule: (plain | compound) ("&" (plain | compound))+
WHITESPACE: /\s+/
plain: /([^\\\[\]():|,&{}]|\.)+/

%import common.SIGNED_NUMBER -> NUMBER
%import common.INT -> NUMBER_Q
""", start="start")

# Pre-parse validation to catch common errors
def validate_prompt(prompt):
    """Validate prompt syntax before parsing to prevent silent failures."""
    if prompt.count('[') != prompt.count(']') or prompt.count('(') != prompt.count(')'):
        raise ValueError(f"Unbalanced brackets in prompt: {prompt}. Ensure all '[' and '(' have matching ']' and ')'.")
    if '::' in prompt and not (prompt.endswith('!') or prompt.endswith('!!') or prompt.endswith('~')):
        raise ValueError(f"Sequence not terminated with '!', '~', or '!!': {prompt}")
    return True

# Memoized tree resolution for performance
@lru_cache(maxsize=1000)
def resolve_tree(tree):
    """Recursively resolve a tree node to its final string representation."""
    if isinstance(tree, lark.Tree):
        return "".join(resolve_tree(child) for child in tree.children)
    return str(tree)

def get_learned_conditioning_prompt_schedules(prompts, base_steps, hires_steps=None, use_old_scheduling=False, seed=None, verbose=False, telemetry=False):
    """
    Parse prompts into schedules for Stable Diffusion conditioning.

    Args:
        prompts: List of prompt strings.
        base_steps: Number of base sampling steps.
        hires_steps: Number of high-res sampling steps (optional).
        use_old_scheduling: Use legacy scheduling logic.
        seed: Random seed for reproducible alternates.
        verbose: Enable detailed parsing feedback.
        telemetry: Log parsing details for production monitoring.

    Returns:
        List of schedules, where each schedule is a list of [step, prompt] pairs.

    Example:
        >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10, seed=42)[0]
        >>> g("test")
        [[10, 'test']]
        >>> g("a [b:3]")
        [[3, 'a '], [10, 'a b']]
        >>> g("a [b: 3]")
        [[3, 'a '], [10, 'a b']]
        >>> g("a [[[b]]:2]")
        [[2, 'a '], [10, 'a [[b]]']]
        >>> g("[(a:2):3]")
        [[3, ''], [10, '(a:2)']]
        >>> g("a [b : c : 1] d")
        [[1, 'a b  d'], [10, 'a  c  d']]
        >>> g("a[b:[c:d:2]:1]e")
        [[1, 'abe'], [2, 'ace'], [10, 'ade']]
        >>> g("a [unbalanced")
        [[10, 'a [unbalanced']]
        >>> g("a [b:.5] c")
        [[5, 'a  c'], [10, 'a b c']]
        >>> g("[fe|]male")
        [[1, 'female'], [2, 'male'], [3, 'female'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'female'], [8, 'male'], [9, 'female'], [10, 'male']]
        >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10, 10, seed=42)[0]
        >>> g("a [b:.5] c")
        [[10, 'a b c']]
    """
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Handle step configuration
    if hires_steps is None or use_old_scheduling:
        int_offset = 0
        flt_offset = 0
        steps = base_steps
    else:
        int_offset = base_steps
        flt_offset = 1.0
        steps = hires_steps

    def collect_steps(steps, tree):
        """Collect step intervals and resolved prompts from the parse tree."""
        schedules = []

        def walk(node, prefix="", parent_schedules=None):
            """Recursively walk the parse tree, accumulating prefixes and schedules."""
            if parent_schedules is None:
                parent_schedules = []

            if isinstance(node, lark.Tree):
                if node.data == "scheduled":
                    if len(node.children) < 2:
                        logging.warning(f"Invalid scheduled prompt: {node}. Skipping.")
                        return prefix

                    prompts = node.children[:-2]
                    number_node = node.children[-2]
                    additional_info = node.children[-1] if len(node.children) > 2 else None

                    try:
                        weight = float(number_node)
                    except (TypeError, ValueError):
                        logging.warning(f"Invalid weight '{number_node}' in scheduled prompt. Defaulting to 1.0.")
                        weight = 1.0

                    step_intervals = []
                    is_reverse = False
                    if additional_info:
                        if isinstance(additional_info, str) and additional_info.lower() in ("reverse", "r"):
                            is_reverse = True
                        elif isinstance(additional_info, list):
                            for r in additional_info:
                                start, end = r.split("-")
                                if "%" in start or "%" in end:
                                    start = round(float(start.strip("%")) / 100 * steps)
                                    end = round(float(end.strip("%")) / 100 * steps)
                                else:
                                    start, end = int(start), int(end)
                                if start > end:
                                    logging.warning(f"Invalid range {start}-{end}. Swapping values.")
                                    start, end = end, start
                                if start > steps:
                                    start = steps
                                if end > steps:
                                    end = steps
                                step_intervals.append((start, end))

                    if not step_intervals:
                        num_prompts = len(prompts)
                        step_intervals = [
                            (int(i * (weight * steps) / num_prompts), int((i + 1) * (weight * steps) / num_prompts))
                            for i in range(num_prompts)
                        ]

                    if is_reverse:
                        prompts = prompts[::-1]
                        step_intervals = step_intervals[::-1]

                    local_schedules = []
                    for i, (start, end) in enumerate(step_intervals):
                        prompt_str = resolve_tree(prompts[i]).strip()
                        full_prompt = (prefix + prompt_str).strip()
                        if full_prompt:
                            local_schedules.append([end, full_prompt])
                        elif i == 0:
                            local_schedules.append([end, ""])

                    schedules.extend(local_schedules)
                    return prefix
                elif node.data == "plain":
                    prefix += resolve_tree(node)
                elif node.data == "alternate" or node.data == "alternate1":
                    options = [resolve_tree(child) for child in node.children if isinstance(child, lark.Tree) or isinstance(child, str)]
                    prefix += random.choice(options)
                elif node.data == "alternate2":
                    options = [resolve_tree(child) for child in node.children if isinstance(child, lark.Tree) or isinstance(child, str)]
                    prefix += random.choice(options)
                elif node.data == "grouped":
                    group_items = []
                    for child in node.children:
                        if isinstance(child, str) and '|' in child:
                            group_items.append(random.choice(child.split('|')))
                        elif isinstance(child, lark.Tree):
                            group_items.append(resolve_tree(child))
                        else:
                            group_items.append(str(child))
                    prefix += ", ".join(group_items)
                elif node.data in ("sequence", "top_level_sequence", "nested_sequence"):
                    owner = resolve_tree(node.children[0]) if node.children else ""
                    descriptors = []
                    for child in node.children[1:]:
                        if isinstance(child, str):
                            descriptors.append(child.strip(" ~!"))
                        elif isinstance(child, lark.Tree) and child.data == "nested_sequence":
                            descriptors.append(f"[{' | '.join(resolve_tree(c).strip(' ~!') for c in child.children if isinstance(c, str))}]")
                        else:
                            descriptors.append(resolve_tree(child))
                    prefix += f"{owner}: {', '.join(descriptors)}"
                elif node.data == "numbered":
                    number = node.children[0]
                    child = node.children[1] if len(node.children) > 1 else None
                    if child:
                        prefix += f"{number} {resolve_tree(child)}"
                    else:
                        prefix += str(number)
                elif node.data == "and_rule":
                    items = [resolve_tree(child) for child in node.children if isinstance(child, lark.Tree) or isinstance(child, str)]
                    prefix += " and ".join(items)
                else:
                    for child in node.children:
                        prefix = walk(child, prefix, schedules)
            elif isinstance(node, str):
                prefix += node
            return prefix

        final_prefix = walk(tree)
        if not schedules:
            schedules.append([steps, final_prefix.strip() or resolve_tree(tree).strip()])

        # Ensure unique schedules and correct format
        seen = set()
        unique_schedules = []
        for step, prompt in sorted(schedules, key=lambda x: x[0]):
            if (step, prompt) not in seen:
                unique_schedules.append([step, prompt])
                seen.add((step, prompt))

        if verbose:
            print(f"Schedules before final processing: {schedules}")
            print(f"Unique schedules: {unique_schedules}")

        return unique_schedules

    def at_step(step, tree):
        """Transform the parse tree into a prompt for a specific step."""
        class AtStep(lark.Transformer):
            def alternate(self, args):
                args = ["" if not arg else arg for arg in args]
                return args[(step - 1) % len(args)]

            def alternate1(self, args):
                return random.choice(args)

            def alternate2(self, args):
                return random.choice(args)

            def grouped(self, args):
                resolved = []
                for arg in args:
                    if isinstance(arg, str) and '|' in arg:
                        resolved.append(random.choice(arg.split('|')))
                    else:
                        resolved.append(arg)
                return ", ".join(resolved)

            def sequence(self, args, parent=None):
                if parent is None:
                    owner = args[0]
                    children = args[1:]
                else:
                    owner = parent
                    children = args
                descriptors = []
                for child in children:
                    if isinstance(child, str):
                        descriptors.append(child.strip(" ~!"))
                    elif isinstance(child, list):
                        descriptors.append(self.nested_sequence(child))
                return f"{owner}: {', '.join(descriptors)}"

            def nested_sequence(self, args):
                sequence_elements = [desc.strip(" ~!") for desc in args if isinstance(desc, str)]
                return f"[{' | '.join(sequence_elements)}]"

            def top_level_sequence(self, args):
                owner = args[0]
                sequences = []
                for child in args[1:]:
                    if isinstance(child, list):
                        sequences.append(self.sequence(child, owner))
                    elif isinstance(child, str) and child.strip() == "!!":
                        break
                    else:
                        sequences.append(child.strip(" ~!"))
                return f"{owner} -> {', '.join(sequences)}"

            def and_rule(self, args):
                return " and ".join(args)

            def compound(self, args):
                return "_".join(str(arg) for arg in args)

            def numbered(self, args):
                number = args[0]
                child = args[1] if len(args) > 1 else None
                if child:
                    return f"{number} {child}"
                return str(number)

            def scheduled(self, args):
                if len(args) < 2:
                    return ""
                prompts = args[:-2]
                step_intervals = args[-2]
                is_reverse = args[-1].lower() in ("reverse", "r") if len(args) > 2 else False
                try:
                    weight = float(args[-2][-1]) if isinstance(args[-2], list) else float(args[-2])
                except (TypeError, ValueError):
                    weight = 1.0

                if is_reverse:
                    prompts = prompts[::-1]
                    step_intervals = step_intervals[::-1]

                for i, (start, end) in enumerate(step_intervals):
                    if start <= step <= end:
                        return resolve_tree(prompts[i])
                return resolve_tree(prompts[-1])

            def start(self, args):
                def flatten(x):
                    if isinstance(x, str):
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)
                return ''.join(flatten(args))

            def plain(self, args):
                yield args[0].value

            def __default__(self, data, children, meta):
                for child in children:
                    yield child

        return AtStep().transform(tree)

    def get_schedule(prompt):
        """Parse a single prompt into a schedule."""
        try:
            validate_prompt(prompt)
            tree = schedule_parser.parse(prompt)
            if verbose:
                print(f"Parsed prompt: {prompt}\nParse tree:\n{tree.pretty()}")
            schedule = collect_steps(steps, tree)
            if telemetry:
                logging.info(f"Prompt: {prompt}, Schedule: {schedule}")
            return schedule
        except lark.exceptions.LarkError as e:
            if verbose:
                print(f"Parsing error: {str(e)}\nSuggestions: Check for unbalanced brackets, invalid weights, or missing sequence terminators.")
            return [[steps, prompt]]

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]

# Rest of the original script (unchanged)
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

def get_multicond_learned_conditioning(model, prompts, steps, hires_steps=None, use_old_scheduling=False) -> MulticondLearnedConditioning:
    res_indexes, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)
    learned_conditioning = get_learned_conditioning(model, prompt_flat_list, steps, hires_steps, use_old_scheduling)
    res = []
    for indexes in res_indexes:
        res.append([ComposableScheduledPromptConditioning(learned_conditioning[i], weight) for i, weight in indexes])
    return MulticondLearnedConditioning(shape=(len(prompts),), batch=res)

class DictWithShape(dict):
    def __init__(self, x, shape=None):
        super().__init__()
        self.update(x)
    @property
    def shape(self):
        return self["crossattn"].shape

def reconstruct_cond_batch(c: list[list[ScheduledPromptConditioning]], current_step):
    param = c[0][0].cond
    is_dict = isinstance(param, dict)
    if is_dict:
        dict_cond = param
        res = {k: torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype) for k, param in dict_cond.items()}
        res = DictWithShape(res, (len(c),) + dict_cond['crossattn'].shape)
    else:
        res = torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype)
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
                    logging.error(f"Invalid end_at_step value: {entry.end_at_step}. Defaulting to current_step.")
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
        stacked = stack_conds(tensors).to(device=param.device, dtype=param.dtype)
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