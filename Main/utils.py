from colorama import Fore, Style

def print_colored(text, color):
    color_code = getattr(Fore, color.upper())
    print(color_code + text, Style.RESET_ALL)
