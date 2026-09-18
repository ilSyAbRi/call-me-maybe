"""Constrained decoder for schema-compliant token generation."""

import os

from src.llm import create_model
from src.model import FunctionDefinition, Prompt
from src.parser import load_functions


class Constrain_decoder:
    """Guide LLM token generation using structural constraints."""

    def __init__(
        self,
        func_defs: dict[str, FunctionDefinition] | str = (
            "data/input/functions_definition.json"
        ),
    ) -> None:
        """Create the model and load all function definitions.

        Args:
            func_defs: Path to definitions file or dictionary of definitions.
        """
        self.model = create_model()
        if isinstance(func_defs, dict):
            self._func_defs: dict[str, FunctionDefinition] = func_defs
        elif os.path.exists(func_defs):
            self._func_defs = load_functions(func_defs)
        else:
            self._func_defs = {}

        self.digit_tokens: dict[int, str] = {}
        self.minus_token: int = 0
        self.dot_token: int = 0
        self.stop_tokens: set[int] = set()
        self.funcs_prompt: str = ""
        self.func_candidates: dict[str, list[int]] = {}

        self._init_token_cache()
        self._init_func_cache()

    @property
    def func_defs(self) -> dict[str, FunctionDefinition]:
        """Return the dictionary of available function definitions."""
        return self._func_defs

    @func_defs.setter
    def func_defs(self, val: dict[str, FunctionDefinition]) -> None:
        """Update available function definitions and refresh cache."""
        self._func_defs = val
        self._init_func_cache()

    def _init_token_cache(self) -> None:
        """Pre-index special tokens for numeric and string decoding."""
        self.digit_tokens = {
            int(self.model.encode(str(d))[0].tolist()[-1]): str(d)
            for d in range(10)
        }
        self.minus_token = int(self.model.encode("-")[0].tolist()[-1])
        self.dot_token = int(self.model.encode(".")[0].tolist()[-1])

        comma_token = int(self.model.encode(",")[0].tolist()[-1])
        brace_token = int(self.model.encode("}")[0].tolist()[-1])
        newline_token = int(self.model.encode("\n")[0].tolist()[-1])

        self.stop_tokens = {comma_token, brace_token, newline_token}

    def _init_func_cache(self) -> None:
        """Cache prompts and token sequences for available functions."""
        self.funcs_prompt = "\n".join(
            [str(func) for func in self._func_defs.values()]
        )
        self.func_candidates = {
            name: [int(t) for t in self.model.encode(name + "\n")[0].tolist()]
            for name in self._func_defs
        }

    def get_func_name(self, prompt: Prompt) -> FunctionDefinition:
        """Select the function that matches the user's request.

        Uses a prefix trie constrained decoder with terminator tokens
        to avoid prefix collisions between function names.

        Args:
            prompt: User request.

        Returns:
            The selected FunctionDefinition.
        """
        system_prompt = (
            "Task: Select the best matching function.\n\n"
            "Available functions:\n"
            f"{self.funcs_prompt}\n\n"
            "Instructions:\n"
            "1. Read the user request carefully.\n"
            "2. Compare it with every available function.\n"
            "3. Select the function that performs the requested operation.\n"
            "4. Return ONLY the exact function name.\n\n"
            f"User request: {prompt.prompt}\n\n"
            "Function: "
        )

        prompt_ids = [
            int(t) for t in self.model.encode(system_prompt)[0].tolist()
        ]
        respond: list[int] = []
        active: dict[str, list[int]] = dict(self.func_candidates)
        step = 0

        while len(active) > 1:
            logits = self.model.get_logits_from_input_ids(prompt_ids + respond)
            valid_tokens = {
                tokens[step]
                for tokens in active.values()
                if step < len(tokens)
            }

            next_token = max(valid_tokens, key=lambda t: logits[t])
            respond.append(next_token)

            active = {
                k: v
                for k, v in active.items()
                if step < len(v) and v[step] == next_token
            }
            step += 1

        function_name = list(active.keys())[0]
        return self._func_defs[function_name]

    def select_number_value(
        self,
        input_ids: list[int],
        is_integer: bool = False,
        max_digits: int = 20,
    ) -> str:
        """Generate a valid number using a finite state machine.

        Args:
            input_ids: Input prompt token IDs.
            is_integer: Whether to constrain output strictly to an integer.
            max_digits: Maximum number of numeric tokens to generate.

        Returns:
            The decoded number as a string.
        """
        state = 0  # 0: start, 1: digits pre-dot, 2: dot seen, 3: post-dot
        current_ids = list(input_ids)
        result_tokens: list[int] = []

        for _ in range(max_digits):
            logits = self.model.get_logits_from_input_ids(current_ids)
            allowed: set[int] = set()

            if state == 0:
                allowed.update(self.digit_tokens.keys())
                allowed.add(self.minus_token)
            elif state == 1:
                allowed.update(self.digit_tokens.keys())
                if not is_integer:
                    allowed.add(self.dot_token)
                allowed.update(self.stop_tokens)
            elif state == 2:
                allowed.update(self.digit_tokens.keys())
            elif state == 3:
                allowed.update(self.digit_tokens.keys())
                allowed.update(self.stop_tokens)

            next_token = max(allowed, key=lambda t: logits[t])

            if next_token in self.stop_tokens:
                break

            if next_token == self.dot_token:
                state = 2
            elif next_token in self.digit_tokens:
                state = 1 if state in (0, 1) else 3
            elif next_token == self.minus_token:
                state = 0

            result_tokens.append(next_token)
            current_ids.append(next_token)

        decoded = self.model.decode(result_tokens)
        return str(decoded).strip()

    def select_boolean_value(
        self,
        input_ids: list[int],
    ) -> bool:
        """Choose true or false based on the model's scores.

        Args:
            input_ids: Input prompt token IDs.

        Returns:
            True or False boolean value.
        """
        scores = self.model.get_logits_from_input_ids(input_ids)

        true_id = int(self.model.encode("true")[0].tolist()[-1])
        false_id = int(self.model.encode("false")[0].tolist()[-1])

        return bool(scores[true_id] > scores[false_id])

    def take_string_value(
        self,
        input_ids: list[int],
        max_tokens: int = 35,
    ) -> str:
        """Generate a string value from the model.

        Stops on closing quote delimiter or newline to prevent runaway
        prose generation.

        Args:
            input_ids: Token IDs of the prompt ending with opening quote.
            max_tokens: Maximum tokens to generate for this string argument.

        Returns:
            The generated string argument.
        """
        current_ids = list(input_ids)
        result_chars: list[str] = []

        for _ in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(current_ids)
            next_token = max(range(len(logits)), key=lambda t: logits[t])
            tok_str = str(self.model.decode([next_token]))

            if '"' in tok_str:
                idx = tok_str.index('"')
                result_chars.append(tok_str[:idx])
                break

            if "\n" in tok_str or "\r" in tok_str:
                break

            result_chars.append(tok_str)
            current_ids.append(next_token)

        return "".join(result_chars)