# -*- coding: cp1250 -*-
from .control_existing import control_existing_files
from .set_of_molecules import SetOfMolecules, SetOfMoleculesFromChargesFile
from .control_order_of_molecules import control_order_of_molecules
from termcolor import colored
from numpy import sqrt, mean, max, min, sum, corrcoef, log10, array
from scipy.stats import pearsonr
from tabulate import tabulate
from sys import exit
from os import path, mkdir
from operator import itemgetter
from collections import namedtuple, Counter
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Category20
from bokeh.embed import file_html
from bokeh.resources import CDN
from operator import itemgetter
import webbrowser


def background_color(value):
    return "green" if value < 0.05 else "#4ca64c" if value < 0.1 else "#99cc99" if value < 0.15\
        else "yellow" if value < 0.2 else "orange" if value < 0.3 else "red; color: white" if value < 0.4\
        else "darkred; color: white"


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
        self.parameterization = parameterization
        if self.parameterization:
            set_of_molecules_nt = namedtuple("set_of_molecules", ("all_charges", "atomic_types_charges", "molecules", "file"))
            self.ref_set_of_molecules = ref_charges_data
            self.set_of_molecules = set_of_molecules_nt(*charges_data)
        else:
            control_existing_files(((ref_charges_data, True, "file"),
                                    (charges_data, True, "file"),
                                    (self.data_dir, False, "directory")),
                                   rewriting_with_force)
            self.ref_set_of_molecules = SetOfMoleculesFromChargesFile(ref_charges_data)
            self.set_of_molecules = SetOfMoleculesFromChargesFile(charges_data, ref=False)
            control_order_of_molecules(self.ref_set_of_molecules.names, self.set_of_molecules.names,
                                       self.ref_set_of_molecules.file, self.set_of_molecules.file)
        try:
            mkdir(data_dir)
        except FileExistsError:
            pass
        self.statistics()
        self.graphs()

    def statistics(self):
        print("Calculating statistical data...")
        self.all_atoms_data = [round(item, 4) for item in
                               calculate_statistics(self.ref_set_of_molecules.ref_charges,
                                                    self.set_of_molecules.all_charges)]
        self.atomic_types_data = []
        for (atomic_type, ref_charges), (_, charges) in zip(self.ref_set_of_molecules.atomic_types_charges.items(),
                                                            self.set_of_molecules.atomic_types_charges.items()):
            atomic_type_data = [round(item, 4) for item in calculate_statistics(ref_charges, charges)]
            self.atomic_types_data.append([atomic_type] + atomic_type_data + [round(atomic_type_data[4]/(self.all_atoms_data[4]/100), 2)])
        molecules_statistical_data = [calculate_statistics(ref_molecule.charges, molecule.charges) for ref_molecule, molecule in zip(self.ref_set_of_molecules, self.set_of_molecules.molecules)]
        molecules_num_of_atoms = [molecule[4] for molecule in molecules_statistical_data]
        self.molecules_data = [round(item, 4) for item in [mean([x[y] for x in molecules_statistical_data]) for y in range(4)] + [self.ref_set_of_molecules.num_of_molecules, min(molecules_num_of_atoms), max(molecules_num_of_atoms), mean(molecules_num_of_atoms)]]
        headers = ["RMSD", "Maximum deviation", "Average deviation", "Pearson?"]
        self.all_atoms_headers = headers + ["Number of atoms"]
        self.molecules_headers = headers + ["Number of molecules", "Smallest molecule atoms", "Largest molecule atoms", "Average number of atoms"]
        self.atomic_types_headers = ["Atomic type"] + headers + ["Number of atoms", "%"]
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
        if self.parameterization:
            self.individual_molecules = sorted([(molecule.name, num_of_atoms, ", ".join(sorted(set(molecule.atoms_representation(self.parameterization.atomic_types_pattern)))), round(molecule_data[0], 4)) for molecule, num_of_atoms, molecule_data in zip(self.ref_set_of_molecules, molecules_num_of_atoms, molecules_statistical_data)], key=itemgetter(3))
            counter_bonds = Counter()
            for molecule in self.ref_set_of_molecules:
                counter_bonds.update(molecule.bonds_representation("{}_{}".format(self.parameterization.atomic_types_pattern, self.parameterization.atomic_types_pattern)))
            num_of_bonds = sum(list(counter_bonds.values()))
            self.bonds_data = [(bond, count, round(count / (num_of_bonds / 100), 2)) for bond, count in counter_bonds.most_common()]
            self.bonds_headers = ["Type", "Number", "%"]
            self.summary_statistics += "\n\n{}\n\n".format(tabulate(self.bonds_data, headers=self.bonds_headers))
        print(colored("ok\n", "green"))
        print(self.summary_statistics)

    def graphs(self):
        print("Creating pictures...")
        # plt.switch_backend('agg')
        # if self.parameterization:
        #     course_fig = plt.figure(figsize=(11, 9)).add_subplot(111, rasterized=True)
        #     for index, (atomic_symbol, data) in enumerate(
        #             zip(["total RMSD"] + [atomic_type[0] for atomic_type in self.atomic_types_data],
        #                 zip(*self.parameterization.course))):
        #         linewidth = 4 if index == 0 else 1
        #         course_fig.plot(data, label=atomic_symbol, linewidth=linewidth)
        #     course_fig.legend(fontsize=14, loc="upper left")
        #     course_fig.set_title("Course of parameterization", fontsize=20, weight="bold")
        #     plt.ylim(top=0.5)
        #     plt.ylim(bottom=0)
        #     plt.savefig(path.join(self.data_dir, "course_of_parameterization.png"), dpi=300)
        color_numbers = {"H~1": 0, "O~1": 2, "O~2": 4, "N~1": 6, "N~2": 8, "C~1": 10, "C~2": 12, "S~1": 14,
                         "Ca~1": 16, "S~2": 18, "P~1": 1, "P~2": 3, "N~3": 5, "C~3": 7, "Br~1": 9, "Cl~1": 9,
                         "F~1": 9, "I~1": 9, "H": 0, "C": 2, "N": 4, "O": 6, "S": 8, "I": 10, "F": 12, "Br": 14, "Cl": 16, "P": 18}
        p = figure(plot_width=900,
                   plot_height=900,
                   title="Correlation graph",
                   x_axis_label="Reference charges",
                   y_axis_label="Empirical Charges")
        p.title.align = "center"
        p.title.text_font_size = "25px"
        p.xaxis.axis_label_text_font_size = "20px"
        p.yaxis.axis_label_text_font_size = "20px"
        zipped_charges = list(zip(self.ref_set_of_molecules.atomic_types_charges.items(),
                                      self.set_of_molecules.atomic_types_charges.items()))
        for (atomic_symbol, ref_charges), (_, charges) in zipped_charges:
            color = Category20[20][color_numbers[atomic_symbol]]
            p.circle(ref_charges, charges, size=6, alpha=1,
                   muted_alpha=0.07, muted_color=color, legend=atomic_symbol, fill_color=color)
        p.legend.location = "top_left"
        p.legend.click_policy = "mute"
        if self.parameterization:
            self.correlation_graph_html_source = file_html(p, CDN)
        else:
            output_file(path.join(self.data_dir, "comparison.html"))
            show(p)

    def write_html(self, file, sdf, charges, ref_charges, summary_lines, parameters_lines, parameters_file):
        print("Writing html file...")
        atomic_types_table = "<tr><th>" + "</th><th>\n".join(self.atomic_types_headers) + "</th></tr>"
        for atomic_type in self.atomic_types_data:
            atomic_types_table += "\n<tr style=\"background-color: {};\"><td>".format(background_color(atomic_type[1])) + \
                                  "</td><td>".join([str(item) for item in atomic_type]) + "</td></tr>"
        bonds_table = "<tr><th>" + "</th><th>\n".join(self.bonds_headers) + "</th></tr>"
        for bond_type in self.bonds_data:
            bonds_table += "\n<tr><td>" + "</td><td>".join([str(item) for item in bond_type]) + "</td></tr>"
        individual_molecules_table = "<tr><th>" + "</th><th>\n".join(["Name", "Number of molecules", "Present atomic types", "RMSD"]) + "</th></tr>"
        for molecule in self.individual_molecules:
            individual_molecules_table += "\n<tr><td>" + "</td><td>".join([str(item) for item in molecule]) + "</td></tr>"
        head = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="content-type" content="text/html; chrset=utf-8">
<style>
body {font-family: Arial;}

/* Style the tab */
.tab {
    overflow: hidden;
    border: 1px solid #ccc;
    background-color: #f1f1f1;
}

/* Style the buttons inside the tab */
.tab button {
    background-color: inherit;
    float: left;
    border: none;
    outline: none;
    cursor: pointer;
    padding: 14px 16px;
    transition: 0.3s;
    font-size: 17px;
}

/* Change background color of buttons on hover */
.tab button:hover {
    background-color: #ddd;
}

/* Create an active/current tablink class */
.tab button.active {
    background-color: #ccc;
}

/* Style the tab content */
.tabcontent {
    display: none;
    padding: 6px 12px;
    border: 1px solid #ccc;
    border-top: none;
}
</style>
</head>
<body>
"""
        javascript = """
<script>
document.getElementsByClassName('tablinks')[0].click()
function openTab(evt, cityName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(cityName).style.display = "block";
    evt.currentTarget.className += " active";
}
</script>"""
        lines = """
{}

<h1>Results of parameterization</h1>

<div class="tab">
  <button class="tablinks" onclick="openTab(event, 'Correlation graph')">Correlation graph</button>
  <button class="tablinks" onclick="openTab(event, 'Bonds')">Bonds</button>
  <button class="tablinks" onclick="openTab(event, 'Molecules')">Molecules</button>
  <button class="tablinks" onclick="openTab(event, 'Parameterization informations')">Parameterization information</button>
  <button class="tablinks" onclick="openTab(event, 'Parameters')">Parameters</button>
  <button class="tablinks" onclick="openTab(event, 'SDF file')">SDF file</button>
  <button class="tablinks" onclick="openTab(event, 'Empirical charges file')">Empirical charges file</button>
  <button class="tablinks" onclick="openTab(event, 'Reference charges file')">Reference charges file</button>
  <button class="tablinks" onclick="openTab(event, 'Contact & Licence')">Contact & Licence</button>
</div>

<div id="Correlation graph" class="tabcontent">
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
</div>

<div id="Bonds" class="tabcontent">
<h2>Bonds:</h2>
<table border=1>
{}
</table>
</div>

<div id="Molecules" class="tabcontent">
<h2>Molecules:</h2>
<table border=1>
{}
</table>
</div>

<div id="Parameterization informations" class="tabcontent">
<font size=\"5\">{}</font> 
</div>

<div id="Parameters" class="tabcontent">
<h2>Parameters</h2>
<a href = \"{}\">{}</a>
</br></br></br>
<font>{}</font></div>

<div id="SDF file" class="tabcontent">
<h2>SDF file</h2>
<a href = \"{}\">{}</a></div>

<div id="Empirical charges file" class="tabcontent">
<h2>Empirical charges file</h2>
<a href = \"{}\">{}</a></div>

<div id="Reference charges file" class="tabcontent">
<h2>Reference charges file</h2>
<a href = \"{}\">{}</a></div>

<div id="Contact & Licence" class="tabcontent">
<font><b>Author:</b> Ondřej Schindler</font></br> 
<font><b>Email:</b> dargen3@centrum.cz</font></br> 
<font><b>Source code:</b> <a href = \"https://github.com/dargen3/MACH\">https://github.com/dargen3/MACH</a></font></br>
<font><b>Licence:</b> MIT</font> 
</div>

{}
</body>
</html>
""".format(head,
           "<tr><th>" + "</th><th>\n".join(self.all_atoms_headers) + "</th></tr>\n<tr><td>" +
           "</td><td>\n".join([str(item) for item in self.all_atoms_data]) + "</td></tr>",
           "<tr><th>" + "</th><th>\n".join(self.molecules_headers) + "</th></tr>\n<tr><td>" +
           "</td><td>\n".join([str(item) for item in self.molecules_data]) + "</td></tr>",
           atomic_types_table, self.correlation_graph_html_source, bonds_table, individual_molecules_table,
           "</br>\n".join(["<b>" + line.replace(":", "</b>:") for line in summary_lines]),
           parameters_file, parameters_file,
           "</br>\n".join([line.replace("<", "&lt;").replace(">", "&gt;") for line in parameters_lines]),
           sdf,sdf, charges, charges, ref_charges, ref_charges,
           javascript)
        with open(file, "w") as html_file:
            html_file.write(lines)
        print(colored("ok\n", "green"))
        webbrowser.open(file)



"""



<button class="tablinks" onclick="openTab(event, 'Course of parameterization')">Course of parameterization</button>

<div id="Course of parameterization" class="tabcontent">
<img src=\"course_of_parameterization.png\" width=\"1600\">
</div>


"""