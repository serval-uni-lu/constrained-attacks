def beautify_cline(file_path):
    # Read the contents of the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find the index of the last occurrence of '\cline'
    last_cline_index = None
    for i in reversed(range(len(lines))):
        if lines[i].lstrip().startswith("\\cline"):
            last_cline_index = i
            break

    # Remove the last '\cline' if found
    if last_cline_index is not None:
        lines.pop(last_cline_index)
        # Write the modified content back to the file
        with open(file_path, "w") as file:
            file.writelines(lines)


def center_latex(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    lines[0] = lines[0].replace("\\begin{table}", "\\begin{table*}")
    lines[-1] = lines[-1].replace("\\end{table}", "\\end{table*}")
    
    with open(file_path, "w") as file:
            file.writelines(lines)
    
def beautify_latex(path: str, center=True) -> None:
    # return
    beautify_cline(path)
    if center:
        center_latex(path)
