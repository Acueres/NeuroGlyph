from generator.code_generator import Gemma3CodeGenerator


def main():
    generator = Gemma3CodeGenerator()
    code = generator.generate(
        """Task: Generate a function 'add'.
        The function 'add' adds two reals that come as parameters, and returns a real.
        Use the 'println' function to print the result in the main function.
        Start functions with keyword def, not func."""
    )
    print("Model reply:\n")
    print(code)


if __name__ == "__main__":
    main()
