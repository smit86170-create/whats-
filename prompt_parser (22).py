from __future__ import annotations

import re
import logging
from collections import namedtuple
import lark
import random
import torch

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | grouped | alternate | top_level_sequence | compound | numbered | and_rule | plain | WHITESPACE)*

!emphasized: "(" prompt ")" 
        | "(" prompt ":" NUMBER ")"  // Support (prompt:weight)
        | "[" prompt "]"
scheduled: "[" prompt (":" prompt)* "]" ":" NUMBER (step_range_list | reverse_flag | step_range_list reverse_flag)?  // Allow single and multi-prompt schedules
reverse_flag: "reverse" | "r"
step_range_list: step_range ("," step_range)*
step_range: NUMBER "-" NUMBER | NUMBER "%" "-" NUMBER "%"
   
alternate: (prompt | plain | compound) ("|" (prompt | plain | compound))+
grouped: "{" ((NUMBER_Q | prompt | sequence | grouped) (","| "|")?)+ "}"
top_level_sequence: prompt ("::" | ":::") prompt ("::" prompt)* "!"?  // Simplified to handle ::: and ::
sequence: prompt ("::" | ":::") prompt ("," | WHITESPACE)* nested_sequence* "!"?  
nested_sequence: ("::" | ":::") prompt ("," | WHITESPACE)* "!"?

compound: /[a-zA-Z0-9]+(_[a-zA-Z0-9]+)+/
numbered: NUMBER_Q ("!")? (grouped | sequence | compound | and_rule | plain)
and_rule: (plain | compound) ("&" (plain | compound))+
WHITESPACE: /\s+/
plain: /([^\\\[\]()]|\\.)+/

%import common.SIGNED_NUMBER -> NUMBER
%import common.INT -> NUMBER_Q
""")

def get_learned_conditioning_prompt_schedules(prompts, base_steps, hires_steps=None, use_old_scheduling=False):
    """
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [b:c:0.5]")
    [[5, 'a b'], [10, 'a c']]
    >>> g("a [b:c:]")  # Invalid weight
    [[10, 'a [b:c:]']]
    >>> g("character:::outfit::red_dress")  # Unclosed sequence
    [[10, 'character: outfit: red_dress']]
    >>> g("[a:b:-0.5]")  # Negative weight
    [[10, 'a']]
    >>> g("{cloudy|sunny,rainy}")  # Grouped alternate
    [[10, 'cloudy, rainy']] or [[10, 'sunny, rainy']]
    """
    if hires_steps is None or use_old_scheduling:
        int_offset = 0
        flt_offset = 0
        steps = base_steps
    else:
        int_offset = base_steps
        flt_offset = 1.0
        steps = hires_steps

    def collect_steps(steps, tree):
        res = []
        context = []  # Track context for prompts like "a [b:3]"

        def resolve_tree(tree):
            if isinstance(tree, lark.Tree):
                return "".join(resolve_tree(child) for child in tree.children)
            return str(tree)

        class CollectSteps(lark.Visitor):
            def alternate(self, tree):
                options = [resolve_tree(child) for child in tree.children]
                res.append((steps, random.choice(options)))

            def compound(self, tree):
                res.append((steps, "_".join(tree.children)))

            def top_level_sequence(self, tree):
                parts = []
                for child in tree.children:
                    if isinstance(child, str) and child.strip() in ("!", "::", ":::"):
                        continue
                    parts.append(resolve_tree(child))
                res.append((steps, ": ".join(parts)))

            def sequence(self, tree, parent=None):
                parts = []
                for child in tree.children:
                    if isinstance(child, str) and child.strip() in ("!", "::", ":::"):
                        continue
                    if isinstance(child, lark.Tree) and child.data == "nested_sequence":
                        parts.append(self.nested_sequence(child))
                    else:
                        parts.append(resolve_tree(child))
                return ": ".join(parts)

            def nested_sequence(self, tree):
                sequence_elements = []
                for child in tree.children:
                    if isinstance(child, str) and child.strip() not in ("!", "::", ":::"):
                        sequence_elements.append(child.strip())
                    elif isinstance(child, lark.Tree):
                        sequence_elements.append(resolve_tree(child))
                return f"[{' | '.join(sequence_elements)}]"

            def grouped(self, tree):
                group_descriptions = []
                for child in tree.children:
                    if isinstance(child, lark.Tree):
                        if child.data == "alternate":
                            options = [resolve_tree(c) for c in child.children]
                            group_descriptions.append(random.choice(options))
                        else:
                            group_descriptions.append(resolve_tree(child))
                    else:
                        group_descriptions.append(str(child).strip())
                res.append((steps, ", ".join(group_descriptions)))

            def scheduled(self, tree):
                if not hasattr(tree, "children") or len(tree.children) < 2:
                    logger.warning(f"Invalid scheduled structure: {tree}")
                    res.append((steps, resolve_tree(tree)))
                    return

                prompts = []
                for child in tree.children[:-1]:
                    if isinstance(child, lark.Tree):
                        prompts.append(resolve_tree(child))
                    else:
                        prompts.append(str(child))

                number_node = tree.children[-1]
                additional_info = tree.children[-2] if len(tree.children) > 2 else None

                try:
                    v = float(number_node)
                    if v < 0:
                        logger.warning(f"Negative weight {v} in scheduled prompt, defaulting to first prompt")
                        res.append((steps, context[-1] if context else prompts[0] if prompts else ""))
                        return
                    if v > 1 and len(prompts) > 1:
                        logger.warning(f"Weight {v} exceeds 1, clamping to 1")
                        v = min(1.0, v)
                except ValueError:
                    logger.warning(f"Invalid number format in scheduled prompt: {number_node}")
                    res.append((steps, context[-1] + " " + "".join(prompts) if context else "".join(prompts)))
                    return

                is_reverse = False
                step_intervals = []

                if additional_info:
                    if isinstance(additional_info, str) and additional_info.lower() in ("reverse", "r"):
                        is_reverse = True
                    elif isinstance(additional_info, list):
                        for r in additional_info:
                            try:
                                start, end = r.split("-")
                                if "%" in start or "%" in end:
                                    start = round(float(start.strip("%")) / 100 * steps)
                                    end = round(float(end.strip("%")) / 100 * steps)
                                else:
                                    start, end = int(start), int(end)
                                if start < 0 or end < 0:
                                    logger.warning(f"Negative step range {start}-{end}, clamping to 0")
                                    start, end = max(0, start), max(0, end)
                                if start > end:
                                    logger.warning(f"Invalid range {start}-{end}, skipping")
                                    continue
                                if start > steps or end > steps:
                                    logger.warning(f"Range {start}-{end} exceeds steps {steps}, clamping")
                                    start, end = min(start, steps), min(end, steps)
                                step_intervals.append((start, end))
                            except ValueError:
                                logger.warning(f"Invalid step range format: {r}")
                                continue

                if not step_intervals:
                    num_prompts = len(prompts)
                    if num_prompts <= 1:
                        step = min(int(v) if v >= 1 else int(v * steps), steps)
                        step_intervals = [(0, step)]
                    else:
                        step_intervals = [
                            (int(i * (v * steps) / num_prompts), int((i + 1) * (v * steps) / num_prompts))
                            for i in range(num_prompts)
                        ]

                if is_reverse:
                    prompts = prompts[::-1]
                    step_intervals = step_intervals[::-1]

                tree.children[-2 if additional_info else -1] = (prompts, step_intervals, context[-1] if context else "")
                for _, end in step_intervals:
                    res.append((end, ""))

            def prompt(self, tree):
                context.append(resolve_tree(tree))

        CollectSteps().visit(tree)
        return sorted(set([r[0] for r in res if isinstance(r, tuple)]))

    def at_step(step, tree):
        class AtStep(lark.Transformer):
            def and_rule(self, args):
                resolved_items = [self._resolve_tree(arg) if isinstance(arg, lark.Tree) else str(arg) for arg in args]
                return " and ".join(resolved_items)

            def compound(self, args):
                return "_".join(str(arg) for arg in args)

            def top_level_sequence(self, args):
                parts = []
                for child in args:
                    if isinstance(child, str) and child.strip() in ("!", "::", ":::"):
                        continue
                    parts.append(self._resolve_tree(child) if isinstance(child, lark.Tree) else str(child))
                return ": ".join(parts)

            def sequence(self, args, parent=None):
                parts = []
                for child in args:
                    if isinstance(child, str) and child.strip() in ("!", "::", ":::"):
                        continue
                    if isinstance(child, lark.Tree) and child.data == "nested_sequence":
                        parts.append(self.nested_sequence(child))
                    else:
                        parts.append(self._resolve_tree(child) if isinstance(child, lark.Tree) else str(child))
                return ": ".join(parts)

            def nested_sequence(self, args):
                sequence_elements = []
                for desc in args:
                    if isinstance(desc, str) and desc.strip() not in ("!", "::", ":::"):
                        sequence_elements.append(desc.strip())
                    elif isinstance(desc, lark.Tree):
                        sequence_elements.append(self._resolve_tree(desc))
                return f"[{' | '.join(sequence_elements)}]"

            def alternate(self, args):
                resolved_options = [self._resolve_tree(arg) if isinstance(arg, lark.Tree) else str(arg) for arg in args]
                return random.choice(resolved_options)

            def scheduled(self, args):
                if not args or len(args) < 2:
                    logger.warning(f"Invalid scheduled args: {args}")
                    return ""

                prompts = []
                for arg in args[:-1]:
                    if isinstance(arg, lark.Tree):
                        prompts.append(self._resolve_tree(arg))
                    else:
                        prompts.append(str(arg))

                number_node = args[-1]
                additional_info = args[-2] if len(args) > 2 else None

                try:
                    weight = float(number_node)
                    if weight < 0:
                        logger.warning(f"Negative weight {weight}, using first prompt")
                        return prompts[0] if prompts else ""
                except ValueError:
                    logger.warning(f"Invalid weight {number_node}, using concatenated prompts")
                    return "".join(prompts)

                prompts_and_intervals = args[-2 if additional_info else -1]
                if not isinstance(prompts_and_intervals, tuple):
                    return "".join(prompts)

                prompts, step_intervals, context = prompts_and_intervals
                is_reverse = additional_info.lower() in ("reverse", "r") if isinstance(additional_info, str) else False

                if is_reverse:
                    prompts = prompts[::-1]
                    step_intervals = step_intervals[::-1]

                for i, (start, end) in enumerate(step_intervals):
                    if start <= step <= end:
                        return (context + " " + prompts[i]).strip() if context else prompts[i]
                return context if context else prompts[0] if prompts else ""

            def grouped(self, args):
                group_descriptions = []
                for arg in args:
                    if isinstance(arg, lark.Tree) and arg.data == "alternate":
                        options = [self._resolve_tree(c) for c in arg.children]
                        group_descriptions.append(random.choice(options))
                    elif isinstance(arg, lark.Tree):
                        group_descriptions.append(self._resolve_tree(arg))
                    else:
                        group_descriptions.append(str(arg).strip())
                return ", ".join(group_descriptions)

            def start(self, args):
                result = []
                def flatten(x):
                    if isinstance(x, str):
                        result.append(x)
                    elif isinstance(x, list):
                        for gen in x:
                            flatten(gen)
                    else:
                        result.append(x)
                flatten(args)
                return ''.join(str(r) for r in result)

            def plain(self, args):
                return args[0].value

            def __default__(self, data, children, meta):
                return "".join(str(child) for child in children)

            def _resolve_tree(self, tree):
                if isinstance(tree, lark.Tree):
                    return "".join(self._resolve_tree(child) for child in tree.children)
                return str(tree)

        return AtStep().transform(tree)

    def get_schedule(prompt):
        try:
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError as e:
            logger.warning(f"Failed to parse prompt '{prompt}': {str(e)}")
            return [[steps, prompt]]
        schedule = []
        prev_prompt = ""
        for t in collect_steps(steps, tree):
            prompt_at_step = ''.join(str(p) for p in at_step(t, tree)).strip()
            if prompt_at_step and prompt_at_step != prev_prompt:
                schedule.append([t, prompt_at_step])
                prev_prompt = prompt_at_step
        return schedule or [[steps, prompt]]

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]

ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])


class SdConditioning(list):
    """
    A list with prompts for stable diffusion's conditioner model.
    Can also specify width and height of created image - SDXL needs it.
    """
    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None, copy_from=None):
        super().__init__()
        self.extend(prompts)

        if copy_from is None:
            copy_from = prompts

        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, 'is_negative_prompt', False)
        self.width = width or getattr(copy_from, 'width', None)
        self.height = height or getattr(copy_from, 'height', None)



def get_learned_conditioning(model, prompts: SdConditioning | list[str], steps, hires_steps=None, use_old_scheduling=False):
    """converts a list of prompts into a list of prompt schedules - each schedule is a list of ScheduledPromptConditioning, specifying the comdition (cond),
    and the sampling step at which this condition is to be replaced by the next one.

    Input:
    (model, ['a red crown', 'a [blue:green:5] jeweled crown'], 20)

    Output:
    [
        [
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0523,  ..., -0.4901, -0.3066,  0.0674], ..., [ 0.3317, -0.5102, -0.4066,  ...,  0.4119, -0.7647, -1.0160]], device='cuda:0'))
        ],
        [
            ScheduledPromptConditioning(end_at_step=5, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.0192,  0.3867, -0.4644,  ...,  0.1135, -0.3696, -0.4625]], device='cuda:0')),
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.7352, -0.4356, -0.7888,  ...,  0.6994, -0.4312, -1.2593]], device='cuda:0'))
        ]
    ]
    """
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
        self.shape: tuple = shape  # the shape field is needed to send this object to DDIM/PLMS
        self.batch: list[list[ComposableScheduledPromptConditioning]] = batch


def get_multicond_learned_conditioning(model, prompts, steps, hires_steps=None, use_old_scheduling=False) -> MulticondLearnedConditioning:
    """same as get_learned_conditioning, but returns a list of ScheduledPromptConditioning along with the weight objects for each prompt.
    For each prompt, the list is obtained by splitting the prompt using the AND separator.

    https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/
    """

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
    # if prompts have wildly different lengths above the limit we'll get tensors of different shapes
    # and won't be able to torch.stack them. So this fixes that.
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
        #print(f"Conds_for_batch: {conds_for_batch}")

        for composable_prompt in composable_prompts:
            target_index = 0
            
            for current, entry in enumerate(composable_prompt.schedules):
                try:
                    end_at_step = int(entry.end_at_step)                    
                except ValueError:
                    raise TypeError(f"Invalid end_at_step value: {entry.end_at_step}, expected an integer.")
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

System: ]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

def parse_prompt_attention(text):
    """
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0], ['house', 1.5730000000000004], [' ', 1.1], ['on', 1.0], [' a ', 1.1], ['hill', 0.55], [', sun, ', 1.1], ['sky', 1.4641000000000006], ['.', 1.1]]
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
        text_segment = m.group(0)
        weight = m.group(1)

        if text_segment.startswith('\\'):
            res.append([text_segment[1:], 1.0])
        elif text_segment == '(':
            round_brackets.append(len(res))
        elif text_segment == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text_segment == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text_segment == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        elif text_segment == ':':
            continue
        else:
            parts = re.split(re_break, text_segment)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                if part:
                    res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if not res:
        res = [["", 1.0]]

    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1] and res[i][0] != "BREAK" and res[i + 1][0] != "BREAK":
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)