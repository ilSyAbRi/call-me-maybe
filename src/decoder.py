from src.parser import load_functions
from src.llm import create_model
from src.model import Prompt, FunctionDefinition
import math


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