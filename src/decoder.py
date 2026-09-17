from src.parser import load_prompts, load_functions
from src.llm import create_model
from src.model import Prompt, FunctionDefinition

from rich import print

class Constrain_decoder:
    def __init__(self):
        self.model = create_model()
        self.func_defs: dict = load_functions("data/input/functions_definition.json")

    def get_func_name(self, prompt: Prompt) -> FunctionDefinition:
        funcs_names_ids = [self.model.encode(func).int().tolist()[0] for func in self.func_defs]
        funcs_prompt = '\n'.join([str(func) for func in self.func_defs.values()])
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

        for i in range(max([len(func) for func in funcs_names_ids])):
            # return if single function remains
            if len(funcs_names_ids) == 1:
                respond = funcs_names_ids[0]
                break

            # generating the logits
            logits = self.model.get_logits_from_input_ids(prompt_ids + respond)

            # getting the comparision tokens
            valide_tokens = [func[i] for func in funcs_names_ids if i < len(func)]

            # getting the next token
            next_token = max(valide_tokens, key=lambda token: logits[token])
            respond.append(next_token)

            # filtering
            funcs_names_ids = list(filter(lambda func_ids: next_token == func_ids[i], funcs_names_ids))

        return self.func_defs[self.model.decode(respond)]


"""

i = 0

# funcs_ids
[
    [32, 12, 9, 19],
    [10, 23, 3],
    [10, 17, 2, 15],
]


"""