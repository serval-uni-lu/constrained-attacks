def percent_format(x: float) -> str:
    x = x * 100
    return f"{x:.1f}"


def percent_format_diff(x: float) -> str:
    x = x * 100
    return "\\tiny{$" + f"{x:+.1f}" + "$}"


def time_format(x: float) -> str:
    return f"{x:.0f}"
