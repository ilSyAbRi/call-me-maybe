from src.parser import load_prompts, load_functions

def main():
    try:
        prompts = load_prompts("data/input/function_calling_tests.json")
        functions = load_functions("data/input/functions_definition.json")
    
        for prompt in prompts:
            print(prompt)
    
        for function in functions:
            print(function)
    except Exception as e:
        print("Error :",e)


if __name__ == "__main__":
    main()