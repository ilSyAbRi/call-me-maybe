from src.parser import load_functions
from src.llm import create_model
from src.model import Prompt, FunctionDefinition
import math
import json

class Constrain_decoder:
    def __init__(self):
        """Create the model and load all function definitions."""
        self.model = create_model()
        self.func_defs: dict = load_functions(
            "data/input/functions_definition.json"
        )

    def get_func_name(self, prompt: Prompt) -> FunctionDefinition:
        """Select the function that matches the user's request."""

        funcs_names_ids = [
            self.model.encode(func).int().tolist()[0]
            for func in self.func_defs
        ]

        funcs_prompt = "\n".join(
            [str(func) for func in self.func_defs.values()]
        )

        system_prompt = (
            "Task: Select the best matching function.\n\n"
            "Available functions:\n"
            f"{funcs_prompt}\n\n"
            "Instructions:\n"
            "1. Read the user request carefully.\n"
            "2. Compare it with every available function.\n"
            "3. Select the function that performs the requested operation.\n"
            "4. Return ONLY the exact function name.\n\n"
            f"User request: {prompt.prompt}\n\n"
            "Function: "
        )

        prompt_ids = self.model.encode(system_prompt).int().tolist()[0]
        respond: list[int] = []

        for i in range(max(len(func) for func in funcs_names_ids)):
            if len(funcs_names_ids) == 1:
                respond = funcs_names_ids[0]
                break

            logits = self.model.get_logits_from_input_ids(
                prompt_ids + respond
            )

            valid_tokens = [
                func[i]
                for func in funcs_names_ids
                if i < len(func)
            ]

            next_token = max(
                valid_tokens,
                key=lambda token: logits[token]
            )

            respond.append(next_token)

            funcs_names_ids = [
                func_ids
                for func_ids in funcs_names_ids
                if i < len(func_ids) and next_token == func_ids[i]
            ]

        function_name = self.model.decode(respond)

        return self.func_defs[function_name]

    def select_number_value(
        self,
        input_ids: list[int],
        max_digits: int = 20,
    ) -> str:
        """Generate one number from the model."""

        current_ids = list(input_ids)
        result_tokens: list[int] = []

        allowed_ids: set[int] = set()
        stop_ids: set[int] = set()

        for character in "0123456789-.":
            encoded = self.model.encode(character)
            token_id = encoded[0].tolist()[-1]
            allowed_ids.add(token_id)

        for character in ",}":
            encoded = self.model.encode(character)
            token_id = encoded[0].tolist()[-1]
            stop_ids.add(token_id)

        valid_ids = allowed_ids | stop_ids

        for _ in range(max_digits):
            scores = self.model.get_logits_from_input_ids(current_ids)

            filtered_scores = [-math.inf] * len(scores)

            for token_id in valid_ids:
                filtered_scores[token_id] = scores[token_id]

            next_token = filtered_scores.index(max(filtered_scores))

            if next_token in stop_ids:
                break

            result_tokens.append(next_token)
            current_ids.append(next_token)

        return self.model.decode(result_tokens)

    def select_boolean_value(
        self,
        input_ids: list[int],
    ) -> bool:
        """Choose true or false based on the model's scores."""

        scores = self.model.get_logits_from_input_ids(input_ids)

        true_id = self.model.encode("true")[0].tolist()[-1]
        false_id = self.model.encode("false")[0].tolist()[-1]

        if scores[true_id] > scores[false_id]:
            return True

        return False

    def bann_string_token(self):
        banned_str_token: set[int] = set()
        forbidden_characters = {'"', "\n", "\r", "”", "“", "‘", "’"}
        vocab_path = self.model.get_path_to_vocab_file()

        with open(vocab_path, "r") as file:
            vocabulary = json.load(file)

        for token_id in vocabulary.values():
            token_text = self.model.decode([token_id])

            for character in forbidden_characters:
                if character in token_text:
                    banned_str_token.add(token_id)
                    break
        return banned_str_token

    def may_repeat_token(
        self,
        tokens: list[int],
        next_token: int,
    ) -> bool:
        """Check if adding the next token creates a repetition."""

        tokens_with_next = tokens + [next_token]

        if len(tokens_with_next) >= 2:
            last_token = tokens_with_next[-1]
            previous_token = tokens_with_next[-2]

            if last_token == previous_token:
                return True

        if len(tokens_with_next) >= 4:
            last_two_tokens = tokens_with_next[-2:]
            previous_two_tokens = tokens_with_next[-4:-2]

            if last_two_tokens == previous_two_tokens:
                return True

        if len(tokens_with_next) >= 6:
            last_three_tokens = tokens_with_next[-3:]
            previous_three_tokens = tokens_with_next[-6:-3]

            if last_three_tokens == previous_three_tokens:
                return True

        return False

    def get_forbidden_tokens(self) -> set[int]:
        """Get token IDs that can break the JSON structure."""

        forbidden_tokens: set[int] = set()

        forbidden_text = [
            "'",
            "regex",
            "replacement",
            "source_string",
            "{",
            "}",
            ".",
            "\\",
            "\n",
        ]

        vocabulary_path = self.model.get_path_to_vocab_file()

        with open(vocabulary_path, "r") as vocabulary_file:
            vocabulary = json.load(vocabulary_file)

        for token_id in vocabulary.values():
            token_text = self.model.decode([token_id])

            for forbidden_text_item in forbidden_text:
                if forbidden_text_item in token_text:
                    forbidden_tokens.add(token_id)
                    break

        return forbidden_tokens

    def take_string_value(self, input_ids: list[int], max_tokens: int = 30,) -> str:
        """Generate a string value from the model."""

        # Token for the character: "
        close_quote = self.model.encode('"')[0].tolist()[-1]

        # Tokens that we do not want inside the string.
        banned_tokens = self.bann_string_token()

        # Tokens that can break our JSON.
        forbidden_tokens = self.get_forbidden_tokens()

        # We want to allow " because it closes the string.
        banned_tokens.discard(close_quote)
        forbidden_tokens.discard(close_quote)

        # Tokens that we already have.
        current_ids = list(input_ids)

        # Tokens that belong to our new string.
        result_tokens = []

        # Generate one token at a time.
        for _ in range(max_tokens):

            # Ask the model: "What token should come next?"
            scores = self.model.get_logits_from_input_ids(current_ids)

            # Copy the scores.
            new_scores = list(scores)

            # Remove forbidden tokens.
            for token_id in banned_tokens:
                new_scores[token_id] = -math.inf

            # Take the token with the highest score.
            next_token = new_scores.index(max(new_scores))

            # " means that the string is finished.
            if next_token == close_quote:
                if len(result_tokens) > 0:
                    break

            # Stop if the model starts repeating itself.
            if self.may_repeat_token(result_tokens, next_token):
                break

            # Stop if the token can break the JSON structure.
            if next_token in forbidden_tokens:
                if len(result_tokens) > 0:
                    break

            # Give the new token to the model.
            current_ids.append(next_token)

            # Save the new token in our result.
            result_tokens.append(next_token)

        # Convert tokens back into a normal string.
        return self.model.decode(result_tokens)