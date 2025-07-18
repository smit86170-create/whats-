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

# Grammar updated to handle commas correctly and support all features
schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | grouped | alternate | alternate1 | alternate2 | top_level_sequence | sequence | compound | numbered | and_rule | plain | WHITESPACE)*

!emphasized: "(" prompt ")" 
        | "(" prompt ":" prompt ")" 
        | "[" prompt "]"
scheduled: "[" [prompt (":" prompt)+] "]" ":" NUMBER (step_range_list | reverse_flag | step_range_list reverse_flag)?
reverse_flag: "reverse" | "r"
step_range_list: step_range ("," step_range)*
step_range: NUMBER "-" NUMBER | NUMBER "%" "-" NUMBER "%"

alternate: "[" prompt ("|" [prompt])* "]"
alternate1: (prompt) "|" (prompt)+
alternate2: (plain | compound) ("|" (plain | compound))+

grouped: "{" ((NUMBER_Q | prompt | sequence | grouped) ("," | "|")?)+ "}"
top_level_sequence: prompt "::" sequence ("::" sequence)* "!!"
sequence: prompt "::" prompt ("," | WHITESPACE)* nested_sequence* ("!" | ";")
nested_sequence: "::" prompt ("," | WHITESPACE)* ("~" | "!")

compound: /[a-zA-Z0-9]+(_[a-zA-Z0-9]+)+/

numbered: NUMBER_Q ("!" | "_")? (grouped | sequence | compound | and_rule | plain)
and_rule: (plain | compound) ("&" (plain | compound))+
WHITESPACE: /\s+/
plain: /([^\\\[\]()]|\\.)+/


%import common.SIGNED_NUMBER -> NUMBER
%import common.INT -> NUMBER_Q
""", start="start")

# Strengthened validation
def validate_prompt(prompt):
    """Validate prompt syntax before parsing."""
    stack = []
    for char in prompt:
        if char in '([':
            stack.append(char)
        elif char == ')':
            if not stack or stack[-1] != '(':
                raise ValueError(f"Unbalanced brackets in prompt: {prompt}. Ensure all '[' and '(' have matching ']' and ')'.")
            stack.pop()
        elif char == ']':
            if not stack or stack[-1] != '[':
                raise ValueError(f"Unbalanced brackets in prompt: {prompt}. Ensure all '[' and '(' have matching ']' and ')'.")
            stack.pop()
    if stack:
        raise ValueError(f"Unbalanced brackets in prompt: {prompt}. Ensure all '[' and '(' have matching ']' and ')'.")
    # Allow text after top_level_sequence terminated with !!
    if '::' in prompt:
        # Find the last !! to check if sequence is properly terminated
        parts = prompt.split('!!')
        if len(parts) > 1:
            sequence_part = parts[0] + '!!'
            if not (sequence_part.endswith('!') or sequence_part.endswith('~') or sequence_part.endswith(';') or sequence_part.endswith('!!')):
                raise ValueError(f"Sequence not terminated with '!', '~', ';', or '!!': {sequence_part}")
    return True

# Memoized tree resolution
@lru_cache(maxsize=1000)
def resolve_tree(tree):
    """Recursively resolve a tree node to its final string representation, removing extra whitespace."""
    if isinstance(tree, lark.Tree):
        return " ".join(resolve_tree(child) for child in tree.children if not isinstance(child, lark.Token) or child.type != "WHITESPACE").strip()
    return str(tree).strip()

def get_learned_conditioning_prompt_schedules(prompts, base_steps, hires_steps=None, use_old_scheduling=False, seed=None, verbose=False, telemetry=False):
    """
    Converts a list of prompts into schedules for Stable Diffusion.

    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10, seed=42)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a'], [10, 'a b']]
    >>> g("a [b:.5] c")
    [[5, 'a c'], [10, 'a b c']]
    >>> g("a [b:c:1] d")
    [[1, 'a b d'], [10, 'a c d']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'a b e'], [2, 'a c e'], [10, 'a d e']]
    >>> g("[fe|]male")
    [[1, 'female'], [2, 'male'], [3, 'female'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'female'], [8, 'male'], [9, 'female'], [10, 'male']]
    >>> g("lowres, worst aesthetic, bad quality")
    [[10, 'lowres, worst aesthetic, bad quality']]
    >>> g("character:::outfit::red_dress!, accessories::diamond_necklace!, appearance::ponytail, blonde hair, green eyes!!, dark background")
    [[10, 'character -> outfit: red_dress, accessories: diamond_necklace, appearance: ponytail, blonde hair, green eyes, dark background']]
    """
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Handle step configuration
    if hires_steps is None or use_old_scheduling:
        steps = base_steps
        use_scheduling = True
    else:
        steps = hires_steps
        use_scheduling = False

    def collect_steps(steps, tree, prefix="", suffix="", depth=0):
        """Collect step intervals and prompts from the parse tree."""
        schedules = []
        indent = "  " * depth
        if verbose:
            logging.info(f"{indent}Processing node: {tree.data if isinstance(tree, lark.Tree) else tree}")

        # Base case: string token
        if isinstance(tree, str):
            if verbose:
                logging.info(f"{indent}String: {tree}")
            return [[steps, prefix + tree + suffix]]

        # Base case: plain node
        if isinstance(tree, lark.Tree) and tree.data == "plain":
            text = resolve_tree(tree)
            if verbose:
                logging.info(f"{indent}Plain text: {text}")
            return [[steps, prefix + text + suffix]]

        # Scheduled node: [prompt1:prompt2:...:weight]
        if isinstance(tree, lark.Tree) and tree.data == "scheduled" and use_scheduling:
            if verbose:
                logging.info(f"{indent}Found scheduled node: {tree}")
            children = tree.children
            weight_token = children[-1]
            prompts = [child for child in children[:-1] if isinstance(child, lark.Tree) and child.data == "prompt"]
            try:
                weight = float(resolve_tree(weight_token))
            except (TypeError, ValueError):
                logging.warning(f"Invalid weight '{weight_token}' in scheduled prompt. Defaulting to 0.5.")
                weight = 0.5

            boundary = int(weight * steps) if weight <= 1.0 else int(weight)
            boundary = max(0, min(boundary, steps))

            if len(prompts) == 1:
                prompt_str = resolve_tree(prompts[0]).strip()
                schedules.append([boundary, prefix + suffix])
                schedules.append([steps, prefix + prompt_str + suffix])
            else:
                step_increment = boundary / max(1, len(prompts) - 1)
                for i, prompt in enumerate(prompts):
                    step = min(steps, int(i * step_increment)) if i < len(prompts) - 1 else steps
                    prompt_str = resolve_tree(prompt).strip()
                    schedules.append([step, prefix + prompt_str + suffix])
            if verbose:
                logging.info(f"{indent}Scheduled prompts: {[resolve_tree(p) for p in prompts]}")
                logging.info(f"{indent}Step boundary: {boundary}")
                logging.info(f"{indent}Local schedules: {schedules}")
            return schedules

        # Alternate node: [prompt1|prompt2|...]
        if isinstance(tree, lark.Tree) and tree.data == "alternate":
            if verbose:
                logging.info(f"{indent}Found alternate node: {tree}")
            options = []
            for child in tree.children:
                if isinstance(child, lark.Tree) and child.data == "prompt":
                    opt = resolve_tree(child).strip()
                    options.append(opt if opt else "")
                elif isinstance(child, lark.Token) and child.type == "plain":
                    options.append(child.value.strip())
            if not options:
                if verbose:
                    logging.info(f"{indent}No valid alternate options, skipping.")
                return [[steps, prefix + suffix]]
            for step in range(1, steps + 1):
                option = options[(step - 1) % len(options)]
                schedules.append([step, prefix + option + suffix])
            if verbose:
                logging.info(f"{indent}Alternate options: {options}")
                logging.info(f"{indent}Local schedules: {schedules}")
            return schedules

        # Alternate1 and Alternate2: Random choice
        if isinstance(tree, lark.Tree) and tree.data in ("alternate1", "alternate2"):
            if verbose:
                logging.info(f"{indent}Found {tree.data} node: {tree}")
            options = []
            for child in tree.children:
                if isinstance(child, lark.Tree):
                    opt = resolve_tree(child).strip()
                    options.append(opt if opt else "")
                elif isinstance(child, lark.Token) and child.type == "plain":
                    options.append(child.value.strip())
            if not options:
                if verbose:
                    logging.info(f"{indent}No valid options, skipping.")
                return [[steps, prefix + suffix]]
            option = random.choice(options)
            schedules.append([steps, prefix + option + suffix])
            if verbose:
                logging.info(f"{indent}Selected option: {option}")
            return schedules

        # Grouped node: {a, b, c}
        if isinstance(tree, lark.Tree) and tree.data == "grouped":
            if verbose:
                logging.info(f"{indent}Found grouped node: {tree}")
            options = [resolve_tree(child).strip() for child in tree.children if resolve_tree(child).strip()]
            text = ", ".join(options)
            if verbose:
                logging.info(f"{indent}Grouped text: {text}")
            return [[steps, prefix + text + suffix]]

        # Sequence node: для obj::attr1, attr2
        if isinstance(tree, lark.Tree) and tree.data == "sequence":
            # Первый child — объект (например, 'outfit'), остальные — описания (например, 'red_dress')
            described_object = resolve_tree(tree.children[0]).strip()
            descriptors = []
            for child in tree.children[1:]:
                # Пропускаем пустые и служебные символы
                t = resolve_tree(child).strip(" ;~!,")
                if t:
                    descriptors.append(t)
            # Особый случай: если descriptors уже содержит запятые — не добавлять двойных
            text = f"{described_object}: {', '.join(descriptors)}"
            return [[steps, prefix + text + suffix]]

        # Top-level sequence node: для owner::obj1::attr1, obj2::attr2!!, ...
        if isinstance(tree, lark.Tree) and tree.data == "top_level_sequence":
            # Первый child — owner (например, 'character'), остальные — sequence
            owner = resolve_tree(tree.children[0]).strip()
            sequences = []
            trailing_text = []
            for child in tree.children[1:]:
                # Собираем только sequence — вложенные
                if isinstance(child, lark.Tree) and child.data == "sequence":
                    # Важно: collect_steps вернёт [[steps, text]], берем text!
                    seq_schedule = collect_steps(steps, child)
                    if seq_schedule and len(seq_schedule[0]) > 1:
                        sequences.append(seq_schedule[0][1])
                elif isinstance(child, lark.Token) and child.value.strip() == "!!":
                    continue
                else:
                    t = resolve_tree(child).strip(" ,")
                    if t:
                        trailing_text.append(t)
            text = f"{owner} -> {', '.join(sequences)}"
            if trailing_text:
                text += f", {', '.join(trailing_text)}"
            # Вернуть только ОДИН результат (единый prompt)
            return [[steps, prefix + text + suffix]]

        # Nested sequence node
        if isinstance(tree, lark.Tree) and tree.data == "nested_sequence":
            if verbose:
                logging.info(f"{indent}Found nested sequence node: {tree}")
            sequence_elements = [resolve_tree(child).strip(" ;~!") for child in tree.children if resolve_tree(child).strip()]
            text = f"[{' | '.join(sequence_elements)}]"
            if verbose:
                logging.info(f"{indent}Nested sequence text: {text}")
            return [[steps, prefix + text + suffix]]

        # Other nodes
        if isinstance(tree, lark.Tree):
            if verbose:
                logging.info(f"{indent}Processing children of {tree.data}")
            for i, child in enumerate(tree.children):
                if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                    continue
                pre_scheduled = ""
                post_scheduled = ""
                for j, sibling in enumerate(tree.children):
                    if j < i:
                        pre_scheduled += resolve_tree(sibling)
                    elif j > i:
                        post_scheduled += resolve_tree(sibling)
                new_prefix = prefix + pre_scheduled
                new_suffix = post_scheduled + suffix
                child_schedules = collect_steps(steps, child, new_prefix, new_suffix, depth + 1)
                schedules.extend(child_schedules)
            if not schedules:
                schedules.append([steps, prefix + resolve_tree(tree) + suffix])
            if verbose:
                logging.info(f"{indent}Schedules for {tree.data}: {schedules}")
            return schedules

        # Fallback
        if verbose:
            logging.info(f"{indent}Fallback: {resolve_tree(tree)}")
        return [[steps, prefix + resolve_tree(tree) + suffix]]

    def get_schedule(prompt):
        """Parse a single prompt into a schedule."""
        try:
            validate_prompt(prompt)
            tree = schedule_parser.parse(prompt)
            if verbose:
                logging.info(f"Parsed prompt: {prompt}\nParse tree:\n{tree.pretty()}")
            schedules = collect_steps(steps, tree)
            if telemetry:
                logging.info(f"Prompt: {prompt}, Schedule: {schedules}")
            return sorted(schedules, key=lambda x: x[0])
        except lark.exceptions.LarkError as e:
            logging.warning(f"Parsing error for prompt '{prompt}': {str(e)}. Returning as-is.")
            return [[steps, re.sub(r',\s*$', '', prompt.strip())]]
        except Exception as e:
            logging.error(f"Unexpected error parsing prompt '{prompt}': {str(e)}")
            raise

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]

# Rest of the script (unchanged)
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