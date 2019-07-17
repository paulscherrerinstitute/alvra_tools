from .termcolor import cprint

def print_red(txt, *args, **kwargs):
    return cprint(txt, "red", attrs=["bold"], *args, **kwargs)


def print_lines(seq):
    print("\n".join(seq) + "\n")

def print_dict(d):
    key_length = max(len(k) for k in d.keys())
    for k, v in sorted(d.items()):
        print(k.ljust(key_length), v)



