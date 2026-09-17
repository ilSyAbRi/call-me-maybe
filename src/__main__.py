from src.parser import load_prompts
from src.decoder import Constrain_decoder

from src.model import Prompt

from rich import print


{
    "prompt": "whats 1 + 1?",
    "name": "ft_add_numbers",
    "parameters":
    {
        "a": 1, 
        "b": 1
    }
}

def main():
    prompts = load_prompts("data/input/function_calling_tests.json")
    constrain_decoder = Constrain_decoder()

    for prompt in prompts:
        print("\n===", prompt.prompt, "===")
        print(constrain_decoder.get_func_name(prompt).name)
        print("====" + '='*len(prompt.prompt) + "====")


if __name__ == "__main__":
    main()
