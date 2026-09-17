import json
from src.model import Prompt, FunctionDefinition

def load_json(path: str):
    try:
        with open(path, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")

def load_prompts(path: str):
    data = load_json(path)
    prompts = []

    for prompt_data in data:
        prompt = Prompt(prompt=prompt_data.get("prompt"))
        prompts.append(prompt)
    return prompts

def load_functions(path: str) -> dict[str, FunctionDefinition]:
    data = load_json(path)
    functions = dict()

    for function_data in data:
        function = FunctionDefinition(
            name=function_data.get("name"),
            description=function_data.get("description"),
            parameters=function_data.get("parameters"),
            returns=function_data.get("returns"),
        )
        functions[function.name] = function
    return functions