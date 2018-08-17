import webbrowser

def backround_color(value):
    if value < 0.05:
        return "green"
    elif value < 0.1:
        return "#4ca64c"
    elif value < 0.15:
        return "#99cc99"
    elif value < 0.2:
        return "yellow"
    elif value < 0.3:
        return "orange"
    elif value < 0.4:
        return "red; color: white"
    elif value >= 0.4:
        return "darkred; color: white"

def write_html(file, summary_lines, parameters_lines, comparison):
    atomic_types_table = "<tr><th>" + "</th><th>\n".join(comparison.atomic_types_headers) + "</th></tr>"
    for atomic_type in comparison.atomic_types_data:
        atomic_types_table += "\n<tr style=\"background-color: {};\"><td>".format(backround_color(atomic_type[1])) + \
                              "</td><td>".join([str(item) for item in atomic_type]) + "</td></tr>"
    lines = """
<!DOCTYPE html>
<html>
<body>

<h1>
{}
</h1>

</br>
<h2>Parameters:</h2>
<p>
{}
</p>

</br>
<h2>Atoms:</h2>
<table border=1>
{}
</table>
        
</br>
<h2>Molecules:</h2>
<table border=1>
{}
</table>    

<img src=\"all_atoms.png\" width=\"1600\">

<h2>Atomic types:</h2>
<table border=1>
{}
</table> 

<tbody>
</table>
<table border=\"1\" style=\"margin-top: 0.5em\">
<tr>
<td><strong>Legend:</strong> RMSD </td>
<td style=\"background-color: green; color: white;\">&lt; 0.05</td>
<td style=\"background-color: #4ca64c; \">&lt; 0.1</td>
<td style=\"background-color: #99cc99;\">&lt; 0.15</td>
<td style=\"background-color: yellow;\">&lt; 0.2</td>
<td style=\"background-color: orange;\">&lt; 0.3</td>
<td style=\"background-color: red; color: white;\">&lt; 0.4</td>
<td style=\"background-color: darkred; color: white;\">&gt;= 0.4</td>
</tr>
</table>

{}

</body>
</html>
""".format("</br>\n".join(summary_lines),
           "</br>\n".join([line.replace("<", "&lt;").replace(">", "&gt;") for line in parameters_lines]),
           "<tr><th>" + "</th><th>\n".join(comparison.all_atoms_headers) + "</th></tr>\n<tr><td>" +
           "</td><td>\n".join([str(item) for item in comparison.all_atoms_data]) + "</td></tr>",
           "<tr><th>" + "</th><th>\n".join(comparison.molecules_headers) + "</th></tr>\n<tr><td>" +
           "</td><td>\n".join([str(item) for item in comparison.molecules_data]) + "</td></tr>",
           atomic_types_table,
           "\n".join(["<img src=\"{}.png\" style=\"float: left; width: 800px;\">".format(data[0]) for data in comparison.atomic_types_data]))
    with open(file, "w") as html_file:
        html_file.write(lines)
    webbrowser.open(file)
