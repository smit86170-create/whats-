from __future__ import annotations
import re
from collections import namedtuple
import lark
import random
from functools import lru_cache
import hashlib
from itertools import product

import os
import logging
logger = logging.getLogger(__name__)  # не настраиваем basicConfig в библиотеке

# Фиче-флаги (можно переопределять через env):
def _env_bool(name: str, default: str = "0") -> bool:
    v = str(os.getenv(name, default)).strip().lower()
    return v not in ("0", "", "false", "no", "off")

ALLOW_EMPTY_ALTERNATE   = _env_bool("ALLOW_EMPTY_ALTERNATE", "0")
EXPAND_ALTERNATE_PER_STEP = _env_bool("EXPAND_ALTERNATE_PER_STEP", "1")
GROUP_COMBO_LIMIT       = int(os.getenv("GROUP_COMBO_LIMIT", "100"))
# Размер кэша (оставляем как у тебя, просто используем ниже)
CACHE_SIZE = int(os.getenv('PROMPT_PARSER_CACHE_SIZE', 4096))

# Грамматика Lark
# Грамматика Lark (динамически подставляем правило для alternate)
# динамическое правило для alternate
_alt_rule = r' "[" prompt ("|" prompt)+ "]" '         # минимум один '|'
if ALLOW_EMPTY_ALTERNATE:
    _alt_rule = r' "[" prompt ("|" [prompt])+ "]" '   # разрешить пустые опции

# НЕ f-строка! просто r"""...""" + конкатенация
_grammar = r"""
!start: (prompt | /[][():|,]/+)*

prompt: (scheduled | emphasized | grouped
       | alternate | alternate_distinct
       | alternate1 | alternate2
       | top_level_sequence | sequence
       | compound | numbered | and_rule
       | plain | WHITESPACE)*

round_emph: "(" prompt (":" NUMBER)? ")"
square_emph: "[" prompt "]"
!emphasized: round_emph | square_emph

scheduled: "[" [prompt (":" prompt)+] "]" ":" NUMBER (step_range_list | reverse_flag | step_range_list reverse_flag)?
reverse_flag: "reverse" | "r"
step_range_list: step_range ("," step_range)*
step_range: NUMBER "-" NUMBER | NUMBER "%" "-" NUMBER "%"

alternate: """ + _alt_rule + r"""
!alternate_distinct: "[" prompt ("|" prompt)* "]!"
alternate1: (prompt) "|" (prompt)+
alternate2: (plain | compound) ("|" (plain | compound))+

grouped: "{" ((NUMBER_Q | prompt | sequence | grouped) ("," | "|")?)+ "}"

top_level_sequence: prompt ("::" sequence)+ "!!" ("," plain)?
sequence: prompt "::" prompt ("," | WHITESPACE)* nested_sequence* ("!" | ";")
nested_sequence: "::" prompt ("," | WHITESPACE)* ("!" | ";" | "~")

compound: /[a-zA-Z0-9]+(_[a-zA-Z0-9]+)+/
numbered: NUMBER_Q ("!" | "_")? (grouped | sequence | compound | and_rule | plain | alternate | alternate_distinct | alternate1 | alternate2)
atom: round_emph | square_emph
    | grouped
    | alternate | alternate1 | alternate2 | alternate_distinct
    | sequence | top_level_sequence
    | compound | numbered
    | plain

and_rule: atom ("&" atom)+
WHITESPACE: /\s+/
plain: /([^\\\[\]():|,&{}]|\\.)+/

%import common.SIGNED_NUMBER -> NUMBER
%import common.INT -> NUMBER_Q
"""

schedule_parser = lark.Lark(_grammar, start="start")


@lru_cache(maxsize=CACHE_SIZE)
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
        parts = []
        for arg in args:
            s = resolve_tree(arg, keep_spacing=True)
            if s:
                parts.append(s)
        return " and ".join(parts)

    def grouped(self, args):
        parts = []
        for arg in args:
            if isinstance(arg, lark.Token) and getattr(arg, "value", "") in {",", "|"}:
                continue
            s = resolve_tree(arg, keep_spacing=True).strip(" ,|")
            if s:
                parts.append(s)
        return ", ".join(parts)


    def alternate(self, args):
        vals = []
        for arg in args:
            s = resolve_tree(arg, keep_spacing=True)
            if s or s == "":
                vals.append(s)
        return vals[(self.current_step - 1) % len(vals)] if vals else "empty_prompt"


    def alternate_distinct(self, args):
        options = [resolve_tree(arg, keep_spacing=True) for arg in args if resolve_tree(arg)]
        return self.rng.choice(options) if options else "empty_prompt"
    
    def alternate1(self, args):
        options = [resolve_tree(arg, keep_spacing=True) for arg in args if resolve_tree(arg)]
        return self.rng.choice(options) if options else "empty_prompt"


    def alternate2(self, args):
        options = [resolve_tree(arg, keep_spacing=True) for arg in args if resolve_tree(arg)]
        return self.rng.choice(options) if options else "empty_prompt"


    def numbered(self, args):
        # количество
        quantity = int(args[0])

        # флаг distinct: принимаем и "!" и "_", учитываем, что это может быть Token
        distinct = False
        if len(args) > 1:
            mark = str(args[1])
            distinct = mark in ("!", "_")

        # цель (список опций/узел)
        target = args[-1]

        # собираем опции ровно один раз (не вызываем self.visit дважды!)
        options = []
        if isinstance(target, lark.Tree) and getattr(target, "data", None) in ("alternate", "alternate1", "alternate2"):
            for child in target.children:
                val = self.visit(child)
                if val:
                    options.append(val)
        elif isinstance(target, lark.Token):
            options = [resolve_tree(target, keep_spacing=True)]
        else:
            for child in getattr(target, "children", []):
                val = self.visit(child)
                if val:
                    options.append(val)

        if not options:
            return "empty_prompt"

        # выбор
        if distinct:
            if quantity >= len(options):
                unique = self.rng.sample(options, len(options)) if options else []
                pad = self.rng.choices(options, k=quantity - len(unique)) if quantity > len(unique) else []
                selected = unique if quantity == len(unique) else (unique + pad)
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
            elif isinstance(child, (str, lark.Token)) and str(child).strip() == "!!":
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
        # элементы до завершающего маркера
        elements = [
            resolve_tree(arg, keep_spacing=True).strip(" ,~!;")
            for arg in args[:-1]
            if resolve_tree(arg, keep_spacing=True).strip(" ,~!;")
        ]
        # завершающий маркер может быть str или lark.Token
        terminator = args[-1] if args else None
        term_val = str(terminator) if isinstance(terminator, (str, lark.Token)) else None
        if term_val == "~":
            return self.rng.choice(elements) if elements else "empty_prompt"
        return f"[{' | '.join(elements)}]"

    def round_emph(self, args):
        prompt = resolve_tree(args[0], keep_spacing=True)
        try:
            w = (
                float(args[1])
                if len(args) > 1
                and isinstance(args[1], lark.Token)
                and args[1].type == "NUMBER"
                else 1.1
            )
        except ValueError:
            w = 1.1
        return f"({prompt}:{w})"

    def square_emph(self, args):
        prompt = resolve_tree(args[0], keep_spacing=True)
        return f"({prompt}:{1/1.1:.6f})"

    
    def emphasized(self, args):
        prompt = resolve_tree(args[0], keep_spacing=True)
        try:
            weight = float(args[1]) if len(args) > 1 and isinstance(args[1], lark.Token) and args[1].type == "NUMBER" else 1.1
        except ValueError:
            weight = 1.0
        return f"({prompt}:{weight})"

    def scheduled(self, args):
        # Разбираем детей узла: выделяем prompts / NUMBER / step_range_list / reverse
        prompts = []
        step_ranges = []  # список (start, end) в шагах
        reverse = False
        weight = 1.0

        # 1) Соберём все NUMBER; берём ПОСЛЕДНИЙ как вес
        number_tokens = [a for a in args if isinstance(a, lark.Token) and a.type == "NUMBER"]

        # 2) Пройдёмся по детям и классифицируем
        for a in args:
            if isinstance(a, lark.Token) and a.type == "NUMBER":
                continue  # учтём ниже
            if isinstance(a, lark.Tree):
                if getattr(a, "data", None) == "step_range_list":
                    # каждая child — это step_range; парсим "10-20" или "10%-40%"
                    for r in a.children:
                        s = resolve_tree(r, keep_spacing=False)
                        try:
                            start_s, end_s = s.split("-")

                            def _to_abs(x):
                                x = x.strip()
                                if x.endswith("%"):
                                    return int(round(float(x[:-1]) / 100.0 * self.total_steps))
                                return int(round(float(x)))

                            start = max(1, min(_to_abs(start_s), self.total_steps))
                            end = max(1, min(_to_abs(end_s), self.total_steps))
                            if start < end:
                                step_ranges.append((start, end))
                        except Exception:
                            logger.warning(f"Invalid step range format: {s}")
                    continue
                if getattr(a, "data", None) == "reverse_flag":
                    reverse = True
                    continue
            # Остальное считаем «промптами»
            prompts.append(a)

        # 3) Вес (последний NUMBER в children)
        if number_tokens:
            try:
                weight = float(number_tokens[-1])
            except ValueError:
                weight = 1.0

        # 4) Если явных диапазонов нет — делим boundary на равные части
        if not step_ranges:
            boundary = int(round(weight * self.total_steps)) if weight <= 1.0 else int(round(weight))
            boundary = max(1, min(boundary, self.total_steps))
            if not prompts:
                return "empty_prompt"
            if len(prompts) == 1:
                # Особый случай: один вариант в [] — до boundary пусто, после — включаем
                return f"({resolve_tree(prompts[0], keep_spacing=True)}:{weight})" if self.current_step >= boundary else ""
            # Несколько вариантов: равные сегменты в пределах boundary
            seg = max(1, len(prompts))
            step_size = max(1, boundary // seg) if boundary >= seg else 1
            step_ranges = []
            start = 1
            for i in range(seg):
                end = min(self.total_steps if i == seg - 1 else (start + step_size - 1), self.total_steps)
                if start < end:
                    step_ranges.append((start, end))
                start = end + 1

        # 5) reverse — зеркалим и промпты, и интервалы
        if reverse:
            prompts = list(reversed(prompts))
            step_ranges = list(reversed(step_ranges))

        # 6) Выбор активного текста на текущем шаге
        for i, (start, end) in enumerate(step_ranges):
            if start <= self.current_step <= end:
                return f"({resolve_tree(prompts[min(i, len(prompts)-1)], keep_spacing=True)}:{weight})"
        # До первого интервала — пусто, после последнего — последний промпт
        if step_ranges and self.current_step < step_ranges[0][0]:
            return ""
        if prompts:
            return f"({resolve_tree(prompts[-1], keep_spacing=True)}:{weight})"
        return "empty_prompt"

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
        # проглатываем верхнеуровневые разделители, добавленные в !start
        if re.fullmatch(r"[][():|,]+", str(token)):
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
            return [[self.steps, self.prefix + "empty_prompt" + self.suffix]]

        # 1) Извлечение как у тебя
        prompts = [
            p for p in tree.children
            if not (isinstance(p, lark.Token) and p.type == "NUMBER")
            and not (isinstance(p, lark.Tree) and getattr(p, "data", None) in ("step_range_list", "reverse_flag"))
        ]
        number_node = next((p for p in tree.children if isinstance(p, lark.Token) and p.type == "NUMBER"), None)
        step_range_list = next((p for p in tree.children if isinstance(p, lark.Tree) and getattr(p, "data", None) == "step_range_list"), None)
        is_reverse = any(isinstance(p, lark.Tree) and getattr(p, "data", None) == "reverse_flag" for p in tree.children)

        try:
            weight = float(number_node.value) if number_node is not None else 1.0
        except (ValueError, TypeError, AttributeError):
            weight = 1.0

        if not prompts:
            return [[self.steps, self.prefix + "empty_prompt" + self.suffix]]

        # Вспомогалка
        def _clamp_step(x: int) -> int:
            return max(1, min(x, self.steps))

        # 4) Интервалы
        step_intervals = []
        explicit_ranges = False
        if step_range_list:
            explicit_ranges = True
            for sr in step_range_list.children:
                if not (isinstance(sr, lark.Tree) and getattr(sr, "data", None) == "step_range"):
                    continue
                if len(sr.children) != 2:
                    continue
                start_txt = resolve_tree(sr.children[0], keep_spacing=False)
                end_txt = resolve_tree(sr.children[1], keep_spacing=False)

                def _to_steps(txt: str) -> int:
                    s = txt.strip()
                    if s.endswith("%"):
                        try:
                            return int(round(float(s[:-1]) / 100.0 * self.steps))
                        except ValueError:
                            return 1
                    try:
                        return int(round(float(s)))
                    except ValueError:
                        return 1

                start_step = _clamp_step(_to_steps(start_txt))
                end_step   = _clamp_step(_to_steps(end_txt))
                if start_step < end_step:
                    step_intervals.append((start_step, end_step))
        else:
            num_prompts = len(prompts)
            boundary = int(round(weight * self.steps)) if weight <= 1.0 else int(round(weight))
            boundary = _clamp_step(boundary)
            if num_prompts == 1:
                # Особый случай: один вариант внутри [] — включаем его ПОСЛЕ boundary
                before_end = boundary
                after_end  = self.steps
                schedules = []
                # BEFORE: без содержимого []
                schedules.append([before_end, self.prefix + self.suffix])
                # AFTER: с единственным текстом
                last_text = resolve_tree(prompts[0], keep_spacing=True)
                schedules.append([after_end, self.prefix + last_text + self.suffix])
                return schedules
            else:
                # Несколько вариантов: равномерное деление в пределах boundary
                if boundary < num_prompts:
                    boundary = num_prompts
                step_size = boundary / num_prompts
                for i in range(num_prompts):
                    start = int(round(i * step_size)) + 1
                    end   = int(round((i + 1) * step_size))
                    start = _clamp_step(start); end = _clamp_step(end)
                    if start < end:
                        step_intervals.append((start, end))

        # 5) reverse
        if is_reverse:
            prompts = prompts[::-1]
            step_intervals = step_intervals[::-1]

        # 6) Формируем расписания
        schedules = []

        # ДО первого интервала (для явных диапазонов/несколько промптов)
        if step_intervals and step_intervals[0][0] > 1:
            schedules.append([step_intervals[0][0] - 1, self.prefix + self.suffix])

        for i, (start, end) in enumerate(step_intervals):
            end = min(end, self.steps)
            if start < end:
                p = prompts[i]
                if isinstance(p, lark.Tree):
                    child_schedules = self.visit(p)
                else:
                    text = resolve_tree(p, keep_spacing=True)
                    child_schedules = [[self.steps, text]]

                for sched in child_schedules:
                    schedules.append([end, self.prefix + sched[1] + self.suffix])

        # ПОСЛЕ последнего интервала — берём последний prompt
        if step_intervals and step_intervals[-1][1] < self.steps:
            tail_text = resolve_tree(prompts[-1], keep_spacing=True)
            schedules.append([self.steps, self.prefix + tail_text + self.suffix])

        if not schedules:
            return [[self.steps, self.prefix + resolve_tree(tree, keep_spacing=True) + self.suffix]]

        return schedules
    
    def visit_alternate(self, tree):
        options = []
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                continue
            child_schedules = self.visit(child)
            child_options = [
                sched[1].strip(" ,|")
                for sched in child_schedules
                if sched[1].strip(" ,|")
            ]
            options.append(
                child_options
                or [resolve_tree(child, keep_spacing=True).strip(" ,|")]
            )

        if not options:
            return [[self.steps, self.prefix + "empty_prompt" + self.suffix]]

        if EXPAND_ALTERNATE_PER_STEP:
            schedules = []
            for step in range(1, self.steps + 1):
                option = options[(step - 1) % len(options)]
                for sched in option:
                    schedules.append([step, self.prefix + sched + self.suffix])
            return schedules
        else:
            # фиксируем один вариант на весь прогон
            # выбираем группу по текущему сид/рандому и элемент в группе
            group = options[self.rng.randrange(len(options))]
            choice = self.rng.choice(group) if group else "empty_prompt"
            return [[self.steps, self.prefix + choice + self.suffix]]


    def visit_alternate_distinct(self, tree):
        options = []
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                continue
            child_schedules = self.visit(child)
            child_options = [
                sched[1].strip(" ,|")
                for sched in child_schedules
                if sched[1].strip(" ,|")
            ]
            options.append(
                child_options
                or [resolve_tree(child, keep_spacing=True).strip(" ,|")]
            )

        # сплющиваем все варианты
        flat = [opt for group in options for opt in group]
        if not flat:
            return [[self.steps, self.prefix + "empty_prompt" + self.suffix]]

        selected = self.rng.choice(flat)
        return [[self.steps, self.prefix + selected + self.suffix]]

    def visit_alternate1(self, tree):
        return self.visit_alternate_distinct(tree)

    def visit_alternate2(self, tree):
        # исходные опции (plain/compound)
        options = [
            resolve_tree(c, keep_spacing=True).strip()
            for c in tree.children
            if resolve_tree(c, keep_spacing=True).strip()
        ]

        if not options:
            return [[self.steps, self.prefix + "empty_prompt" + self.suffix]]

        # твоя «склейка» суффикса `_xxx` c первой опции — сохраняем
        suffix = options[0].split("_")[-1] if "_" in options[0] else ""
        combined = [
            opt if ("_" in opt or not suffix) else f"{opt}_{suffix}"
            for opt in options
        ]

        choice = self.rng.choice(combined)
        return [[self.steps, self.prefix + choice + self.suffix]]


    def visit_grouped(self, tree):
        all_options = []
        for child in tree.children:
            if isinstance(child, lark.Token):
                if child.type == "WHITESPACE" or child.value in {",", "|"}:
                    continue
            child_schedules = self.visit(child)
            child_options = [
                sched[1].strip(" ,|")
                for sched in child_schedules
                if sched[1].strip(" ,|")
            ]
            all_options.append(
                child_options or [resolve_tree(child, keep_spacing=True).strip(" ,|")]
            )
        out = []
        for i, combo in enumerate(product(*all_options)):
            if i >= GROUP_COMBO_LIMIT:
                break
            text = ", ".join(combo).strip()
            if text:
                out.append([self.steps, self.prefix + text + self.suffix])

        return out or [[self.steps, self.prefix + "empty_prompt" + self.suffix]]

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

    def visit_round_emph(self, tree):
        prompt = resolve_tree(tree.children[0], keep_spacing=True)
        try:
            if (
                len(tree.children) > 1
                and isinstance(tree.children[1], lark.Token)
                and tree.children[1].type == "NUMBER"
            ):
                weight = float(tree.children[1])
            else:
                weight = 1.1
        except (ValueError, TypeError):
            weight = 1.1
        return [[self.steps, self.prefix + f"({prompt}:{weight})" + self.suffix]]

    def visit_square_emph(self, tree):
        prompt = resolve_tree(tree.children[0], keep_spacing=True)
        return [[self.steps, self.prefix + f"({prompt}:{1/1.1:.6f})" + self.suffix]]

    def visit_numbered(self, tree):
        quantity = int(tree.children[0])

        # флаг distinct: "!" или "_"
        distinct = False
        if len(tree.children) > 1:
            mark = str(tree.children[1])
            distinct = mark in ("!", "_")

        target = tree.children[-1]

        # варианты: сначала пробуем дочерние расписания, иначе — сырой текст
        child_schedules = self.visit(target)
        options = [
            sched[1].strip(" ,|")
            for sched in child_schedules
            if sched[1].strip(" ,|")
        ]
        if not options:
            options = [resolve_tree(target, keep_spacing=True).strip(" ,|")]

        if not options:
            return [[self.steps, self.prefix + "empty_prompt" + self.suffix]]

        # выбор
        if distinct:
            if quantity >= len(options):
                unique = self.rng.sample(options, len(options)) if options else []
                pad = self.rng.choices(options, k=quantity - len(unique)) if quantity > len(unique) else []
                selected = unique if quantity == len(unique) else (unique + pad)
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

def at_step_from_schedule(step: int, schedule: list[list[int | str]]) -> str:
    """
    schedule: [[end_step:int, text:str], ...] — по возрастанию end_step
    Возвращает text, активный на переданном step.
    """
    if not schedule:
        return ""
    for end_step, text in schedule:
        try:
            if step <= int(end_step):
                return text
        except Exception:
            # на всякий случай — если end_step внезапно не число
            continue
    return schedule[-1][1]

def at_step(step: int, prompt_or_schedule, *, steps: int | None = None,
            seed: int | None = 42, use_visitor: bool = True) -> str:
    """
    Универсальная обёртка:
    - Если передали готовый schedule -> используем его
    - Если передали строку промпта -> строим schedule и берём значение
    """
    # Готовый список [[end_step, text], ...]
    if isinstance(prompt_or_schedule, list) and prompt_or_schedule and isinstance(prompt_or_schedule[0], list):
        return at_step_from_schedule(step, prompt_or_schedule)

    # Строка промпта
    prompt = str(prompt_or_schedule)
    if steps is None:
        raise ValueError("steps is required when passing a prompt string to at_step(...)")
    sched = get_schedule(prompt, steps, use_scheduling=True, seed=seed, use_visitor=use_visitor)
    return at_step_from_schedule(step, sched)

@lru_cache(maxsize=CACHE_SIZE)
def get_schedule(prompt: str, steps: int, use_scheduling: bool, seed: int | None, use_visitor: bool = True):
    """
    Возвращает список расписаний для одного промпта в формате:
        [[end_step:int, text:str], ...]
    - use_visitor=True: используем CollectSteps, который формирует текст прямо в визиторах
    - use_visitor=False: берём только границы шагов из CollectSteps, а текст вычисляем через ScheduleTransformer
    """
    try:
        tree = schedule_parser.parse(prompt)
    except lark.exceptions.LarkError as e:
        logger.warning("Prompt parse failed: '%s' — %s", prompt, e)
        return [[steps, prompt]]

    # 1) Собираем шаги/варианты через визитор, чтобы получить конечные end_step точки
    collector = CollectSteps(steps, use_scheduling=use_scheduling, seed=seed)
    schedules = collector(tree)  # ожидается [[end_step, text], ...] когда use_visitor=True

    if not schedules:
        # fallback, если визитор ничего не вернул
        return [[steps, prompt]]

    # 2) Если нужно "чистое" преобразование текста трансформером — переcобираем текст
    if not use_visitor:
        rebuilt = []
        for end_step, _ in schedules:
            # На каждом end_step формируем финальную строку трансформером
            transformer = ScheduleTransformer(total_steps=steps, current_step=end_step, seed=seed)
            text = transformer.transform(tree)
            rebuilt.append([end_step, text])
        return rebuilt

    # По умолчанию — то, что собрал визитор
    return schedules


def get_learned_conditioning_prompt_schedules(
    prompts: list[str],
    base_steps: int,
    hires_steps: int | None = None,
    use_old_scheduling: bool = False,
    seed: int | None = 42,
    use_visitor: bool = True,
):
    """
    Возвращает список расписаний по одному на каждый промпт:
        [
          [[end_step, text], ...],   # для prompts[0]
          [[end_step, text], ...],   # для prompts[1]
          ...
        ]
    """
    # Выбор количества шагов по правилам A1111-совместимости
    steps = hires_steps if (hires_steps is not None and not use_old_scheduling) else base_steps
    # Когда включать "расписания": если нет hires_steps или явно просили старый режим
    use_scheduling = (hires_steps is None) or use_old_scheduling

    # Аккуратный проход по каждому промпту через кэшируемую get_schedule(...)
    prompt_schedules = [
        get_schedule(p, steps, use_scheduling, seed, use_visitor=use_visitor)
        for p in prompts
    ]
    return prompt_schedules


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
    
    if len(res) == 1:
        return res[0]
    # склейка батчей из нескольких MulticondLearnedConditioning
    agg_batch = []
    for mc in res:
        agg_batch.extend(mc.batch)
    return MulticondLearnedConditioning(shapes.pop(), agg_batch)

re_AND = re.compile(r"\bAND\b(?!_PERP|_SALT|_TOPK)")
re_weight = re.compile(r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?))?\s*$")

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

class DictWithShape(dict):
    def __init__(self, x, shape=None):
        super().__init__()
        self.update(x)
        self._shape = shape

    @property
    def shape(self):
        if self._shape is not None:
            return self._shape
        any_val = self.get("crossattn")
        if any_val is None and self:
            any_val = next(iter(self.values()))
        return getattr(any_val, "shape", None)
        
def reconstruct_cond_batch(c: list[list[ScheduledPromptConditioning]], current_step):
    import torch
    if not c or not c[0]:
        raise ValueError("Empty conditioning schedule")
    param = c[0][0].cond
    if param is None:
        raise ValueError("Invalid conditioning parameter")
    is_dict = isinstance(param, dict)
    if is_dict:
        dict_cond = param
        res = {k: torch.zeros((len(c),) + v.shape, device=getattr(v, "device", "cpu"), dtype=getattr(v, "dtype", torch.float32)) for k, v in dict_cond.items()}
        res = DictWithShape(res, (len(c),) + dict_cond.get('crossattn', next(iter(dict_cond.values()))).shape)
    else:
        res = torch.zeros((len(c),) + param.shape, device=getattr(param, "device", "cpu"), dtype=getattr(param, "dtype", torch.float32))

    for i, cond_schedule in enumerate(c):
        target_index = 0
        for current, entry in enumerate(cond_schedule):
            if current_step <= entry.end_at_step:
                target_index = current
                break
        if is_dict:
            for k, v in cond_schedule[target_index].cond.items():
                res[k][i] = v
        else:
            res[i] = cond_schedule[target_index].cond
    return res

def stack_conds(tensors):
    import torch
    token_count = max([x.shape[0] for x in tensors])
    for i in range(len(tensors)):
        if tensors[i].shape[0] != token_count:
            last_vector = tensors[i][-1: ]
            last_vector_repeated = last_vector.repeat([token_count - tensors[i].shape[0], 1])
            tensors[i] = torch.vstack([tensors[i], last_vector_repeated])
    return torch.stack(tensors)

def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step):
    import torch
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
        stacked = DictWithShape(stacked, stacked.get('crossattn', next(iter(stacked.values()))).shape)
    else:
        stacked = stack_conds(tensors).to(device=getattr(param, "device", "cpu"), dtype=getattr(param, "dtype", torch.float32))
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
:\s*([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?)\s*\)
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
                res.append([part, 1.0])

    #if round_brackets or square_brackets:
        #logger.warning("Unbalanced brackets in prompt — continuing anyway")

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
    import doctest, random
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
    random.seed(42)

    g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]

    # scheduled — базовые
    assert g("test") == [[10, 'test']]
    assert g("a [b:3]") == [[3, 'a '], [10, 'a b']]
    assert g("[(a:2):3]") == [[3, ''], [10, '(a:2)']]

    # grouped — лимит не нарушаем (на больших группах просто проверить длину)
    big = "{[a|b|c|d|e|f|g],[h|i|j|k|l|m|n],[o|p|q|r|s|t|u]}"
    res = g(big)
    assert len(res) <= GROUP_COMBO_LIMIT

    # numbered — distinct (! и _)
    # здесь сравни повторы/без повторов по длине множества выбранных
    # (если хочется — можно парсить строку и split(', '))

    # alternate_distinct — фиксированный выбор
    g2 = lambda p: get_learned_conditioning_prompt_schedules([p], 6)[0]
    one = g2("[cat|dog|fox]!")
    assert len(set([txt for _, txt in one])) == 1  # один и тот же вариант на всех шагах

    print("All integration tests passed!")