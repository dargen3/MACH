from .control_existing import control_existing_files
from .set_of_molecules import SetOfMolecules, SetOfMoleculesFromChargesFile
from .control_order_of_molecules import control_order_of_molecules
from termcolor import colored
from numpy import sqrt, mean, max, min, sum, corrcoef
from scipy.stats import pearsonr
from tabulate import tabulate
from matplotlib import pyplot as plt, cm
from sys import exit
from os import path, mkdir
from collections import namedtuple
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


def calculate_statistics(ref_charges, charges):
    deviations = abs(ref_charges - charges)
    rmsd = sqrt((1.0/deviations.size)*sum(deviations**2))
    max_deviation = max(deviations)
    average_deviation = mean(deviations)
    pearson_2 = corrcoef(ref_charges, charges)[0, 1] ** 2
    return [rmsd, max_deviation, average_deviation, pearson_2, deviations.size]


class Comparison:
    def __init__(self, ref_charges_data, charges_data, data_dir, rewriting_with_force, parameterization=False):
        self.data_dir = data_dir
        if parameterization:
            set_of_molecules_nt = namedtuple("set_of_molecules", ("all_charges", "atomic_types_charges",
                                                                  "molecules", "file"))
            self.ref_set_of_molecules = ref_charges_data
            self.set_of_molecules = set_of_molecules_nt(*charges_data)
        else:
            control_existing_files(((ref_charges_data, True, "file"),
                                    (charges_data, True, "file"),
                                    (self.data_dir, False, "directory")),
                                   rewriting_with_force)
            self.ref_set_of_molecules = SetOfMoleculesFromChargesFile(ref_charges_data)
            self.set_of_molecules = SetOfMoleculesFromChargesFile(charges_data)
            control_order_of_molecules(self.ref_set_of_molecules.names, self.set_of_molecules.names,
                                       self.ref_set_of_molecules.file, self.set_of_molecules.file)
        try:
            mkdir(data_dir)
        except FileExistsError:
            pass
        self.statistics()
        self.graphs()
        
    def graphs(self):
        plt.switch_backend('agg')
        fig = plt.figure(figsize=(11, 9))
        all_atoms_graph = fig.add_subplot(111)
        colors = cm.tab20.colors
        color_numbers = {"H~1": 0, "O~1": 2, "O~2": 4, "N~1": 6, "N~2": 8, "C~1": 10, "C~2": 12, "S~1": 14,
                         "Ca~1": 16, "S~2": 18, "P~1": 1, "P~2": 3, "N~3": 5, "C~3": 7, "Br~1": 9, "Cl~1": 9,
                         "F~1": 9, "I~1": 9, "H": 0, "C": 2, "N": 4, "O": 6, "S": 8}
        zipped_charges = list(zip(self.ref_set_of_molecules.atomic_types_charges.items(),
                                  self.set_of_molecules.atomic_types_charges.items()))
        for index, ((atomic_symbol, ref_charges), (_, charges)) in enumerate(zipped_charges):
            try:
                color_number = color_numbers[atomic_symbol]
            except KeyError:
                exit(colored("No color for {} atomic type!".format(atomic_symbol), "red"))
            all_atoms_graph.scatter(ref_charges, charges, marker=".", color=colors[color_number], label=atomic_symbol)
            atom_type_graph = plt.figure(figsize=(11, 9)).add_subplot(111, rasterized=True)
            atom_type_graph.set_title(atomic_symbol)
            atom_type_graph.set_xlabel(self.ref_set_of_molecules.file, fontsize=20)
            atom_type_graph.set_ylabel(self.set_of_molecules.file, fontsize=20)
            for (symbol, ref_chg), (_, chg) in zipped_charges:
                if symbol == atomic_symbol:
                    continue
                atom_type_graph.scatter(ref_chg, chg, marker=".", color="gainsboro")
            atom_type_graph.scatter(ref_charges, charges, marker=".", color="black")
            plt.text(plt.xlim()[1] - 0.1, plt.ylim()[0] + 0.1, "RMSD: {:.3f}\n$R^2$: {:.3f}".
                     format(self.atomic_types_data[index][1],
                            self.atomic_types_data[index][4]), ha='right', va='bottom', fontsize=28)
            plt.savefig(path.join(self.data_dir, atomic_symbol), dpi=300)
        all_atoms_graph.text(plt.xlim()[1] - 0.1, plt.ylim()[0] + 0.1, "RMSD: {:.3f}\n$R^2$: {:.3f}".
                             format(self.all_atoms_data[0], self.all_atoms_data[3]), ha='right', va='bottom', fontsize=28)
        all_atoms_graph.legend(fontsize=14, loc="upper left")
        all_atoms_graph.set_xlabel(self.ref_set_of_molecules.file, fontsize=20)
        all_atoms_graph.set_ylabel(self.set_of_molecules.file, fontsize=20)
        all_atoms_graph.set_title("Correlation graph", fontsize=20, weight="bold")
        fig.savefig(path.join(self.data_dir, "all_atoms.png"), dpi=300)

    def statistics(self):
        print("Calculating statistical data...")
        self.all_atoms_data = [round(item, 4) for item in calculate_statistics(self.ref_set_of_molecules.ref_charges,
                                                                               self.set_of_molecules.all_charges)]
        self.atomic_types_data = []
        for (atomic_type, ref_charges), (_, charges) in zip(self.ref_set_of_molecules.atomic_types_charges.items(),
                                                            self.set_of_molecules.atomic_types_charges.items()):
            self.atomic_types_data.append([atomic_type] + [round(item, 4) for item in calculate_statistics(ref_charges, charges)])
        molecules_statistical_data = []
        for ref_molecule, molecule in zip(self.ref_set_of_molecules, self.set_of_molecules.molecules):
            molecules_statistical_data.append(calculate_statistics(ref_molecule.charges, molecule.charges))
        self.molecules_data = [round(item, 4) for item in [mean([x[0] for x in molecules_statistical_data]),
                                                           mean([x[1] for x in molecules_statistical_data]),
                                                           mean([x[2] for x in molecules_statistical_data]),
                                                           mean([x[3] for x in molecules_statistical_data]),
                                                           self.ref_set_of_molecules.num_of_molecules]]
        headers = ["RMSD", "Maximum deviation", "Average deviation", "Pearson^2"]
        self.all_atoms_headers = headers + ["Number of atoms"]
        self.molecules_headers = headers + ["Number of molecules"]
        self.atomic_types_headers = ["Atomic type"] + headers + ["Number of atoms"]
        print(colored("ok\n", "green"))
        self.summary_statistics = """
Statistics for atoms:
{}

Statistics for molecules:
{}


Statistics for atomic types:
{}
""".format(tabulate([self.all_atoms_data], headers=self.all_atoms_headers),
           tabulate([self.molecules_data], headers=self.molecules_headers),
           tabulate(self.atomic_types_data, headers=self.atomic_types_headers))
        print(self.summary_statistics)

    def write_html(self, file, sdf, charges, new_parameters, ref_charges, summary_lines, parameters_lines):
        print("Writing html file...")
        atomic_types_table = "<tr><th>" + "</th><th>\n".join(self.atomic_types_headers) + "</th></tr>"
        for atomic_type in self.atomic_types_data:
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
<h2>Sdf file:</h2>
<a href = \"{}\">{}</a>

</br>
<h2>Charges file:</h2>
<a href = \"{}\">{}</a>

</br>
<h2>Reference charges file:</h2>
<a href = \"{}\">{}</a>

</br>
<h2>Parameters:</h2>
<a href = \"{}\">{}</a>
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
           sdf, sdf, charges, charges, ref_charges, ref_charges, new_parameters, new_parameters,
           "</br>\n".join([line.replace("<", "&lt;").replace(">", "&gt;") for line in parameters_lines]),
           "<tr><th>" + "</th><th>\n".join(self.all_atoms_headers) + "</th></tr>\n<tr><td>" +
           "</td><td>\n".join([str(item) for item in self.all_atoms_data]) + "</td></tr>",
           "<tr><th>" + "</th><th>\n".join(self.molecules_headers) + "</th></tr>\n<tr><td>" +
           "</td><td>\n".join([str(item) for item in self.molecules_data]) + "</td></tr>",
           atomic_types_table,
           "\n".join(["<img src=\"{}.png\" style=\"float: left; width: 800px;\">".format(data[0]) for data in
                      self.atomic_types_data]))
        with open(file, "w") as html_file:
            html_file.write(lines)
        print(colored("ok\n", "green"))
        webbrowser.open(file)
