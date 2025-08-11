from __future__ import annotations
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

import re
from collections import namedtuple
import lark
import random
from functools import lru_cache
import hashlib
import torch
import os
from itertools import product

# Конфигурация размера кэша
#CACHE_SIZE = int(os.getenv('PROMPT_PARSER_CACHE_SIZE', 1024))

# Грамматика Lark
schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():|]/+)*
prompt: (scheduled | emphasized | grouped | alternate | alternate1 | alternate2 | top_level_sequence | sequence | compound | numbered | and_rule | plain | WHITESPACE)* 
!emphasized: "(" prompt ")" 
        | "(" prompt ":" prompt ")"
        | "(" prompt ":" NUMBER ")"
        | "[" prompt "]"
scheduled: "[" [prompt (":" prompt)+] "]" ":" NUMBER (step_range_list | reverse_flag | step_range_list reverse_flag)?
reverse_flag: "reverse" | "r"
step_range_list: step_range ("," step_range)*
step_range: NUMBER "-" NUMBER | NUMBER "%" "-" NUMBER "%"
alternate: "[" prompt ("|" prompt)* "]"
!alternate_distinct: "[" prompt ("|" prompt)* "]!"
alternate1: (prompt) "|" (prompt)+
alternate2: (plain | compound) ("|" (plain | compound))+
grouped: "{" ((NUMBER_Q | prompt | sequence | grouped) ("," | "|")?)+ "}"
top_level_sequence: prompt ("::" sequence)+ "!!" ("," plain)?
sequence: prompt "::" prompt ("," | WHITESPACE)* nested_sequence* ("!" | ";")
nested_sequence: "::" prompt ("," | WHITESPACE)* ("!" | ";" | "~")
compound: /[a-zA-Z0-9]+(_[a-zA-Z0-9]+)+/
numbered: NUMBER_Q ("!" | "_")? (grouped | sequence | compound | and_rule | plain | alternate1 | alternate2)
and_rule: (plain | compound) ("&" (plain | compound))+
WHITESPACE: /\s+/
plain: /([^\\\[\]()&]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
%import common.INT -> NUMBER_Q
""", start="start")

class DictWithShape(dict):
    def __init__(self, x, shape=None):
        super().__init__()
        self.update(x)

    @property
    def shape(self):
        return self["crossattn"].shape

#@lru_cache(maxsize=CACHE_SIZE)
def hash_tree(tree: lark.Tree | lark.Token) -> str:
    if isinstance(tree, lark.Tree):
        return hashlib.md5((tree.data + ''.join(hash_tree(c) for c in tree.children)).encode()).hexdigest()
    return hashlib.md5(str(tree).encode()).hexdigest()

def resolve_tree(tree: lark.Tree | lark.Token, keep_spacing: bool = True) -> str:
    if isinstance(tree, lark.Tree):
        children = []
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                if keep_spacing:
                    children.append(" ")
                continue
            children.append(resolve_tree(child, keep_spacing))
        result = "".join(str(c) for c in children if c)
        return re.sub(r"[\s\u2028\u2029]+", " ", result).strip() if keep_spacing else result.strip()
    return str(tree).strip()

class ScheduleTransformer(lark.Transformer):
    def __init__(self, total_steps: int, current_step: int = 1, seed: int | None = 42):
        super().__init__()
        self.total_steps = total_steps
        self.current_step = current_step
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random


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
        return self.rng.choice(options) if options else "empty_prompt"


    def alternate2(self, args):
        options = [resolve_tree(arg, keep_spacing=True) for arg in args if resolve_tree(arg)]
        return self.rng.choice(options) if options else "empty_prompt"


    def numbered(self, args):
        quantity = int(args[0])
        distinct = args[1] == "!" if len(args) > 1 and isinstance(args[1], str) else False
        target = args[-1]

        options = []
        if isinstance(target, lark.Tree) and target.data in ("alternate", "alternate1", "alternate2"):
            options = [self.visit(child) for child in target.children if self.visit(child)]
        else:
            options = (
                [resolve_tree(target, keep_spacing=True)]
                if isinstance(target, lark.Token)
                else [self.visit(child) for child in target.children if self.visit(child)]
            )

        if not options:
            return "empty_prompt"

        if distinct:
            if quantity > len(options):
                selected = self.rng.sample(options, len(options)) + self.rng.choices(options, k=quantity - len(options))
            else:
                selected = self.rng.sample(options, quantity)
        else:
            selected = self.rng.choices(options, k=quantity)

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
        # Извлекаем элементы, исключая завершающий символ
        elements = [resolve_tree(arg, keep_spacing=True).strip(" ,~!;") for arg in args[:-1] if resolve_tree(arg).strip(" ,~!;")]
        # Проверяем завершающий символ
        terminator = args[-1] if args and isinstance(args[-1], str) else None
        if terminator == "~":
            return self.rng.choice(elements) if elements else "empty_prompt"
        return f"[{' | '.join(elements)}]"
    
    def emphasized(self, args):
        prompt = resolve_tree(args[0], keep_spacing=True)
        try:
            weight = float(args[1]) if len(args) > 1 and isinstance(args[1], lark.Token) and args[1].type == "NUMBER" else 1.1
        except ValueError:
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

class CollectSteps(lark.Visitor):
    def __init__(self, steps, prefix="", suffix="", depth=0, use_scheduling=True, seed=None):
        super().__init__()
        self.steps = steps
        self.prefix = prefix
        self.suffix = suffix
        self.depth = depth
        self.use_scheduling = use_scheduling
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random
        self.schedules = []
        self.MAX_COMBINATIONS = 100

    def visit(self, tree):
        if isinstance(tree, lark.Tree):
            method_name = f"visit_{tree.data}"
            method = getattr(self, method_name, self._default_visit)
            return method(tree)
        elif isinstance(tree, lark.Token):
            return self._visit_token(tree)
        return []

    def _default_visit(self, tree):
        schedules = []
        for i, child in enumerate(tree.children):
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                continue
            pre = "".join(resolve_tree(c, keep_spacing=True) for j, c in enumerate(tree.children) if j < i and not (isinstance(c, lark.Token) and c.type == "WHITESPACE"))
            post = "".join(resolve_tree(c, keep_spacing=True) for j, c in enumerate(tree.children) if j > i and not (isinstance(c, lark.Token) and c.type == "WHITESPACE"))
            collector = CollectSteps(self.steps, prefix=self.prefix + pre, suffix=post + self.suffix, depth=self.depth + 1, use_scheduling=self.use_scheduling, seed=self.seed)
            child_schedules = collector.visit(child)
            schedules.extend(child_schedules)
        return schedules

    def _visit_token(self, token):
        if token.type == "WHITESPACE":
            return []
        return [[self.steps, self.prefix + str(token) + self.suffix]]

    def visit_plain(self, tree):
        text = resolve_tree(tree, keep_spacing=True)
        return [[self.steps, self.prefix + text + self.suffix]]

    def visit_top_level_sequence(self, tree):
        transformer = ScheduleTransformer(self.steps, 1, self.seed)
        text = transformer.transform(tree)
        return [[self.steps, self.prefix + text + self.suffix]]

    def visit_scheduled(self, tree):
        if not tree.children:
            return
        # Извлекаем промпты, исключая число и доп. информацию
        prompts = [p for p in tree.children if not (isinstance(p, lark.Token) and p.type in ("NUMBER", "reverse_flag")) and not (isinstance(p, lark.Tree) and p.data == "step_range_list")]
        number_node = next((p for p in tree.children if isinstance(p, lark.Token) and p.type == "NUMBER"), None)
        step_range_list = next((p for p in tree.children if isinstance(p, lark.Tree) and p.data == "step_range_list"), None)
        is_reverse = any(p for p in tree.children if isinstance(p, lark.Token) and p.type == "reverse_flag")

        try:
            weight = float(number_node) if number_node else 1.0
        except (ValueError, TypeError):
            weight = 1.0

        if not prompts:
            return [[self.steps, self.prefix + "empty_prompt" + self.suffix]]

        step_intervals = []
        if step_range_list:
            # Обработка step_range_list (например, 1-5,6-10)
            for sr in step_range_list.children:
                if isinstance(sr, lark.Tree) and sr.data == "step_range":
                    start, end = sr.children
                    try:
                        start_val = float(start.value)
                        end_val = float(end.value.rstrip('%')) if isinstance(end, lark.Token) else float(end.value.rstrip('%'))
                        if '%' in str(end):
                            start_val = start_val / 100.0 * self.steps
                            end_val = end_val / 100.0 * self.steps
                        start_val = max(0, int(start_val))
                        end_val = max(1, int(end_val))
                        if start_val < end_val:
                            step_intervals.append((start_val, end_val))
                    except (ValueError, TypeError):
                        continue
        else:
            # Равномерное деление шагов
            num_prompts = len(prompts)
            step_size = weight * self.steps / num_prompts if weight <= 1.0 else self.steps / num_prompts
            step_intervals = [
                (int(i * step_size), int((i + 1) * step_size)) for i in range(num_prompts)
            ]

        # Применяем reverse, если указан
        if is_reverse:
            prompts = prompts[::-1]
            step_intervals = step_intervals[::-1]

        # Формируем расписание с рекурсивной обработкой
        schedules = []
        for i, (start, end) in enumerate(step_intervals):
            end = min(end, self.steps)
            if start < end:
                child_schedules = self.visit(prompts[i]) if isinstance(prompts[i], lark.Tree) else [[self.steps, resolve_tree(prompts[i])]]
                for sched in child_schedules:
                    schedules.append([end, self.prefix + sched[1] + self.suffix])

        self.res.extend(schedules)

    
    def visit_alternate(self, tree):
        options = []
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                continue
            child_schedules = self.visit(child)
            child_options = [sched[1].strip(" ,|") for sched in child_schedules if sched[1].strip(" ,|")]
            options.append(child_options or [resolve_tree(child, keep_spacing=True).strip(" ,|")])
        if not options:
            return [[self.steps, self.prefix + "empty_prompt" + self.suffix]]
        schedules = []
        for step in range(1, self.steps + 1):
            option = options[(step - 1) % len(options)]
            for sched in option:
                schedules.append([step, self.prefix + sched + self.suffix])
        return schedules

    def visit_alternate_distinct(self, tree):
        options = []
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                continue
            child_schedules = self.visit(child)
            child_options = [sched[1].strip(" ,|") for sched in child_schedules if sched[1].strip(" ,|")]
            options.append(child_options or [resolve_tree(child, keep_spacing=True).strip(" ,|")])
        if not options:
            return [[self.steps, self.prefix + "empty_prompt" + self.suffix]]
        selected = self.rng.choice(options[0]) if options[0] else "empty_prompt"
        return [[self.steps, self.prefix + selected + self.suffix]]

    def visit_alternate1(self, tree):
        return self.visit_alternate_distinct(tree)

    def visit_alternate2(self, tree):
        # Добавлено для A1111: обработка составных слов с '_'
        options = [resolve_tree(c).strip() for c in tree.children]
        combined_options = []
        for option in options:
            if "_" in option:
                combined_options.append(option)
            else:
                suffix = options[0].split("_")[-1] if "_" in options[0] else ""
                combined_options.append(f"{option}_{suffix}" if suffix else option)
        self.res.append([[self.steps, self.prefix + "|".join(combined_options) + self.suffix]])


    def visit_grouped(self, tree):
        all_options = []
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                continue
            child_schedules = self.visit(child)
            child_options = [sched[1].strip(" ,|") for sched in child_schedules if sched[1].strip(" ,|")]
            all_options.append(child_options or [resolve_tree(child, keep_spacing=True).strip(" ,|")])
        # БЕЗ ОГРАНИЧЕНИЙ:
        from itertools import product
        combinations = list(product(*all_options))
        return [[self.steps, self.prefix + ", ".join(combo) + self.suffix] for combo in combinations if ", ".join(combo).strip()]

    def visit_sequence(self, tree):
        transformer = ScheduleTransformer(self.steps, 1, self.seed)
        text = transformer.transform(tree)
        return [[self.steps, self.prefix + text + self.suffix]]

    def visit_nested_sequence(self, tree):
        # Извлекаем элементы, исключая завершающий символ
        elements = [
            resolve_tree(c, keep_spacing=True).strip(" ,~!;")
            for c in tree.children[:-1]
            if resolve_tree(c, keep_spacing=True).strip(" ,~!;")
        ]
        # Проверяем завершающий символ
        terminator = tree.children[-1] if tree.children and isinstance(tree.children[-1], lark.Token) else None
        if terminator and terminator.value == "~":
            text = self.rng.choice(elements) if elements else "empty_prompt"
        else:
            text = f"[{' | '.join(elements)}]"
        return [[self.steps, self.prefix + text + self.suffix]]

    def visit_numbered(self, tree):
        quantity = int(tree.children[0])
        distinct = str(tree.children[1]) == "!" if len(tree.children) > 1 else False
        target = tree.children[-1]
        options = []
        child_schedules = self.visit(target)
        options = [sched[1].strip(" ,|") for sched in child_schedules if sched[1].strip(" ,|")]
        if not options:
            options = [resolve_tree(target, keep_spacing=True).strip(" ,|")]
        if not options:
            return [[self.steps, self.prefix + "empty_prompt" + self.suffix]]
        if distinct:
            if quantity > len(options):
                selected = self.rng.sample(options, len(options)) + self.rng.choices(options, k=quantity - len(options))
            else:
                selected = self.rng.sample(options, quantity)
        else:
            selected = self.rng.choices(options, k=quantity)
        return [[self.steps, self.prefix + ", ".join(selected) + self.suffix]]

    def visit_and_rule(self, tree):
        text = " and ".join(resolve_tree(c, keep_spacing=True) for c in tree.children if resolve_tree(c, keep_spacing=True))
        return [[self.steps, self.prefix + text + self.suffix]]

    def visit_emphasized(self, tree):
        prompt = resolve_tree(tree.children[0], keep_spacing=True)
        try:
            weight = float(tree.children[1]) if len(tree.children) > 1 and isinstance(tree.children[1], lark.Token) and tree.children[1].type == "NUMBER" else 1.1
        except (ValueError, TypeError):
            weight = 1.1
        return [[self.steps, self.prefix + f"({prompt}:{weight})" + self.suffix]]

    def __call__(self, tree):
        self.schedules = self.visit(tree)
        return self.schedules or [[self.steps, self.prefix + resolve_tree(tree, keep_spacing=True) + self.suffix]]

def at_step(step, tree):
    """
    Добавлено для A1111: итеративно возвращает промпт для текущего шага в формате A1111.
    """
    class AtStep(lark.Transformer):
        def scheduled(self, args):
            # Обработка scheduled для A1111 с (prompt:focus)
            if not args or len(args) < 2:
                return
            *prompts, when, _, is_reverse, weight = args
            step_intervals = args[-2]
            is_reverse = args[-1].lower() in ("reverse", "r") if len(args) > 2 else False
            if is_reverse:
                prompts = prompts[::-1]
                step_intervals = step_intervals[::-1]
            for i, (start, end) in enumerate(step_intervals):
                if start <= step <= end:
                    yield f"({prompts[i]}:focus)"  # Формат A1111
            yield f"({prompts[-1]}:{weight})"

        def alternate(self, args):
            args = ["" if not arg else arg for arg in args]
            yield args[(step - 1) % len(args)]

        def start(self, args):
            def flatten(x):
                if isinstance(x, str):
                    yield x
                else:
                    for gen in x:
                        yield from flatten(gen)
            return "".join(flatten(args))

        def __default__(self, data, children, meta):
            for child in children:
                yield child

    return "".join(AtStep().transform(tree))

def get_schedule(prompt):
    try:
        tree = schedule_parser.parse(prompt)
    except lark.exceptions.LarkError:
        return [[steps, prompt]]
    
    # Собираем шаги
    visitor = CollectSteps()
    visitor.visit(tree)
    return [[t[0], at_step(t[0], tree)] for t in visitor.res]
    
def get_learned_conditioning_prompt_schedules(prompts, base_steps, hires_steps=None, use_old_scheduling=False, seed: int | None = 42, use_visitor: bool = True):
    rng = random.Random(seed) if seed is not None else random
    steps = hires_steps if hires_steps is not None and not use_old_scheduling else base_steps
    use_scheduling = not hires_steps or use_old_scheduling

    #@lru_cache(maxsize=CACHE_SIZE)
    def get_schedule(prompt, steps, use_scheduling, seed):
        try:
            tree = schedule_parser.parse(prompt)
            collector = CollectSteps(steps, use_scheduling=use_scheduling, seed=seed)
            schedules = collector(tree)
            if not use_visitor:
                transformer = ScheduleTransformer(steps, 1, seed)
                schedules = [[step, transformer.transform(tree)] for step, _ in schedules]
            return schedules
        except lark.exceptions.LarkError as e:
            logger.warning("Prompt parse failed: '%s' — %s", prompt, str(e))
            return [[steps, prompt]]

    result = []
    for prompt in prompts:
        result.append(get_schedule(prompt, steps, use_scheduling, seed))
    return result

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
        if not prompt_schedule:
            raise ValueError(f"Empty schedule for prompt '{prompt}'")
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

def get_multicond_learned_conditioning(model, prompts: SdConditioning | list[str], steps, hires_steps=None, use_old_scheduling=False):
    steps = hires_steps if hires_steps is not None and not use_old_scheduling else steps
    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps, hires_steps, use_old_scheduling)
    conds_list, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)
    
    cache = {}
    res = []
    for prompt, schedule, conds in zip(prompts, prompt_schedules, conds_list):
        if not schedule:
            raise ValueError(f"Empty schedule for prompt '{prompt}'")
        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue
        
        texts = SdConditioning([str(x[1]) for x in schedule], copy_from=prompts)
        try:
            model_conds = model.get_learned_conditioning(texts)
        except Exception:
            raise ValueError(f"Conditioning failed for prompt '{prompt}'")
        
        prompt_schedule = []
        for index, weight in conds:
            cond_schedule = []
            for i, (end_at_step, _) in enumerate(schedule):
                try:
                    end_at_step = int(end_at_step)
                except ValueError:
                    raise ValueError(f"Invalid step boundary in schedule for prompt '{prompt}'")
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
        raise ValueError("Inconsistent tensor shapes across subprompts.")
    
    return res[0] if len(res) == 1 else MulticondLearnedConditioning(shapes.pop(), res)

re_AND = re.compile(r"\bAND\b(?!_PERP|_SALT|_TOPK)")
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
    if not c or not c[0]:
        raise ValueError("Empty conditioning schedule")
    param = c[0][0].cond
    if param is None:
        raise ValueError("Invalid conditioning parameter")
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
    if not c.batch or not c.batch[0]:
        raise ValueError("Empty multicond batch")
    param = c.batch[0][0].schedules[0].cond
    if param is None:
        raise ValueError("Invalid conditioning parameter")
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
        stacked = DictWithShape(stacked, None)
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
    res = []
    round_brackets = []
    square_brackets = []
    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text_match = m.group(0)
        weight = m.group(1)
        if text_match.startswith('\\'):
            res.append([text_match[1:], 1.0])
        elif text_match == '(':
            round_brackets.append(len(res))
        elif text_match == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            try:
                weight_value = float(weight)
            except ValueError:
                weight_value = 1.0
            multiply_range(round_brackets.pop(), weight_value)
        elif text_match == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text_match == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text_match)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part.strip(), 1.0])

    if round_brackets or square_brackets:
        print("Warning: Unbalanced brackets in prompt — continuing anyway")

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
    random.seed(42)
    g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
    assert g("test") == [[10, 'test']]
    assert g("a [b:3]") == [[3, 'a '], [10, 'a b']]
    assert g("[(a:2):3]") == [[3, ''], [10, '(a:2)']]
    assert g("{[cat|dog], bird}") == [[10, 'cat, bird'], [10, 'dog, bird']]
    assert g("{[cat|dog], [bird|fish]}") == [[10, 'cat, bird'], [10, 'cat, fish'], [10, 'dog, bird'], [10, 'dog, fish']]
    result = g("3{[cat|dog]}")
    expected = [[10, 'cat, cat, dog']]
    assert result == expected
    print("All integration tests passed!")