from src.parser import load_prompts, load_functions

def main():
    prompts = load_prompts("data/input/function_calling_tests.json")
    functions = load_functions("data/input/functions_definition.json")
    
    for prompt in prompts:
        print(prompt)
    
    for function in functions:
        print(function)


if __name__ == "__main__":
    main()