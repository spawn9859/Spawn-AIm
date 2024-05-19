from colorama import Fore, Style

def print_colored(text, color):
    color_mapping = {
        "BLACK": Fore.BLACK,
        "RED": Fore.RED,
        "GREEN": Fore.GREEN,
        "YELLOW": Fore.YELLOW,
        "BLUE": Fore.BLUE,
        "MAGENTA": Fore.MAGENTA,
        "CYAN": Fore.CYAN,
        "WHITE": Fore.WHITE,
        "RESET": Fore.RESET,
        "PURPLE": Fore.MAGENTA  # Custom mapping for purple
    }
    color_code = color_mapping.get(color.upper(), Fore.RESET)
    print(color_code + text, Style.RESET_ALL)
