# -*- coding: cp1250 -*-
from .control_existing import control_existing_files
from .set_of_molecules import SetOfMolecules, SetOfMoleculesFromChargesFile
from .control_order_of_molecules import control_order_of_molecules
from termcolor import colored
from numpy import sqrt, mean, max, min, sum, corrcoef, array, histogram, asarray, arange
from scipy.stats import pearsonr
from sys import exit
from os import path, mkdir
from collections import namedtuple, Counter
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Category20
from bokeh.resources import INLINE
from bokeh.util.browser import view
from bokeh.embed import components
from bokeh.models import Legend, HoverTool, Range1d
from bokeh.models.widgets import Panel, Tabs
from operator import itemgetter
from shutil import copyfile
import csv


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
    def __init__(self, ref_charges_data, charges_data, data_dir, rewriting_with_force, parameterization=False, validation=None):
        self.data_dir = data_dir
        self.parameterization = parameterization
        if self.parameterization:
            set_of_molecules_nt = namedtuple("set_of_molecules", ("all_charges", "atomic_types_charges", "molecules"))
            self.ref_set_of_molecules_validation = validation[0]
            self.set_of_molecules_validation = set_of_molecules_nt(*validation[1])
            self.ref_set_of_molecules = ref_charges_data
            self.set_of_molecules = set_of_molecules_nt(*charges_data)

        else:
            output_file = "{}_{}.html".format(path.basename(charges_data).split(".")[0],
                                              path.basename(ref_charges_data).split(".")[0])
            control_existing_files(((ref_charges_data, True, "file"),
                                    (charges_data, True, "file"),
                                    (output_file, False, "file"),
                                    (self.data_dir, False, "directory")),
                                   rewriting_with_force)
            mkdir(self.data_dir)
            self.ref_set_of_molecules = SetOfMoleculesFromChargesFile(ref_charges_data)
            self.set_of_molecules = SetOfMoleculesFromChargesFile(charges_data, ref=False)
            control_order_of_molecules(self.ref_set_of_molecules.names, self.set_of_molecules.names,
                                       self.ref_set_of_molecules.file, self.set_of_molecules.file)
        self.statistics()
        self.graphs()
        if not self.parameterization:
            self.write_html_comparison(charges_data, ref_charges_data, output_file)

    def statistics(self):
        print("Calculating statistical data...")
        self.all_atoms_data = [round(item, 4) for item in
                               calculate_statistics(self.ref_set_of_molecules.ref_charges,
                                                    self.set_of_molecules.all_charges)]
        self.atomic_types_data = []
        for (atomic_type, ref_charges), (_, charges) in zip(self.ref_set_of_molecules.atomic_types_charges.items(),
                                                            self.set_of_molecules.atomic_types_charges.items()):
            atomic_type_data = [round(item, 4) for item in calculate_statistics(ref_charges, charges)]
            self.atomic_types_data.append([atomic_type] + atomic_type_data + [round(atomic_type_data[4] / (self.all_atoms_data[4] / 100), 2)])
        if self.parameterization:
            self.all_atoms_data_validation = [round(item, 4) for item in
                                              calculate_statistics(self.ref_set_of_molecules_validation.ref_charges,
                                                                   self.set_of_molecules_validation.all_charges)]

            self.atomic_types_data_validation = []
            for (atomic_type, ref_charges), (_, charges) in zip(self.ref_set_of_molecules_validation.atomic_types_charges.items(),
                                                                self.set_of_molecules_validation.atomic_types_charges.items()):
                if not len(charges):
                    continue
                atomic_type_data = [round(item, 4) for item in calculate_statistics(ref_charges, charges)]
                self.atomic_types_data_validation.append([atomic_type] + atomic_type_data + [round(atomic_type_data[4] / (self.all_atoms_data_validation[4] / 100), 2)])


        molecules_statistical_data = [calculate_statistics(ref_molecule.charges, molecule.charges) for ref_molecule, molecule in zip(self.ref_set_of_molecules, self.set_of_molecules.molecules)]



        molecules_num_of_atoms = [molecule[4] for molecule in molecules_statistical_data]
        self.molecules_data = [round(item, 4) for item in [mean([x[y] for x in molecules_statistical_data]) for y in range(4)] + [self.ref_set_of_molecules.num_of_molecules, min(molecules_num_of_atoms), max(molecules_num_of_atoms), mean(molecules_num_of_atoms)]]

        if self.parameterization:

            molecules_statistical_data_validaton = [calculate_statistics(ref_molecule.charges, molecule.charges) for ref_molecule, molecule in zip(self.ref_set_of_molecules_validation, self.set_of_molecules_validation.molecules)]
            molecules_num_of_atoms_validation = [molecule[4] for molecule in molecules_statistical_data_validaton]
            self.molecules_data_validation = [round(item, 4) for item in [mean([x[y] for x in molecules_statistical_data_validaton]) for y in range(4)] + [self.ref_set_of_molecules_validation.num_of_molecules, min(molecules_num_of_atoms_validation), max(molecules_num_of_atoms_validation), mean(molecules_num_of_atoms_validation)]]
        with open(path.join(self.data_dir, "molecules.log"), "w") as molecule_logs_file:
            molecule_logs_file.write(
                "name, num. of atoms, atomic types, rmsd, max. deviation, av. deviation, pearson**2\n")
            writer = csv.writer(molecule_logs_file)
            writer.writerows(sorted([(molecule.name, num_of_atoms, ", ".join(
                sorted(set(molecule.atoms_representation(self.parameterization.atomic_types_pattern if self.parameterization else None)))),
                                      round(molecule_data[0], 4), round(molecule_data[1], 4),
                                      round(molecule_data[2], 4), round(molecule_data[3], 4)) for
                                     molecule, num_of_atoms, molecule_data in
                                     zip(self.ref_set_of_molecules, molecules_num_of_atoms,
                                         molecules_statistical_data)], key=itemgetter(3)))
        if self.parameterization:
            counter_bonds = Counter()
            for molecule in self.ref_set_of_molecules:
                counter_bonds.update(molecule.bonds_representation("{}_{}".format(self.parameterization.atomic_types_pattern, self.parameterization.atomic_types_pattern)))
            num_of_bonds = sum(list(counter_bonds.values()))
            self.bonds_data = [(bond, count, round(count / (num_of_bonds / 100), 2)) for bond, count in counter_bonds.most_common()]
        print(colored("ok\n", "green"))

    def graphs(self):
        print("Creating graphs...")
        color_numbers = {"H~1": 0, "O~1": 2, "O~2": 4, "N~1": 6, "N~2": 8, "C~1": 10, "C~2": 12, "S~1": 14,
                         "Ca~1": 16, "S~2": 18, "P~1": 1, "P~2": 3, "N~3": 5, "C~3": 7, "Br~1": 9, "Cl~1": 11,
                         "F~1": 13, "I~1": 15, "H": 0, "C": 2, "N": 4, "O": 6, "S": 8, "I": 10, "F": 12, "Br": 14, "Cl": 16, "P": 18, "Ca": 1}
        correlation_graph_parameterization = figure(plot_width=900,
                                                    plot_height=900,
                                                    title="Correlation graph",
                                                    x_axis_label="Reference charges",
                                                    y_axis_label="Empirical charges",
                                                    output_backend="webgl")
        correlation_graph_parameterization.title.align = "center"
        correlation_graph_parameterization.title.text_font_size = "17pt"
        correlation_graph_parameterization.xaxis.axis_label_text_font_size = "25px"
        correlation_graph_parameterization.yaxis.axis_label_text_font_size = "25px"
        correlation_graph_parameterization.axis.major_label_text_font_size = '20px'

        correlation_graph_parameterization.line([-1000, 1000], [-1000, 1000])
        zipped_charges = list(zip(self.ref_set_of_molecules.atomic_types_charges.items(),
                                  self.set_of_molecules.atomic_types_charges.items()))
        for index, ((atomic_symbol, ref_charges), (_, charges)) in enumerate(zipped_charges):
            try:
                color = Category20[20][color_numbers[atomic_symbol]]
            except KeyError:
                color = Category20[20][index % 20]
            correlation_graph_parameterization.circle(ref_charges, charges, size=6, legend=atomic_symbol, fill_color=color, line_color=color)
        correlation_graph_parameterization.legend.location = "top_left"
        correlation_graph_parameterization.legend.click_policy = "hide"
        max_charge = max((max(self.ref_set_of_molecules.ref_charges), max(self.set_of_molecules.all_charges)))
        min_charge = min((min(self.ref_set_of_molecules.ref_charges), min(self.set_of_molecules.all_charges)))
        corr = (max_charge - min_charge) / 10
        min_charge -= corr
        max_charge += corr
        correlation_graph_parameterization.x_range = Range1d(min_charge, max_charge)
        correlation_graph_parameterization.y_range = Range1d(min_charge, max_charge)

        if self.parameterization:
            correlation_graph_validation = figure(plot_width=900,
                                                  plot_height=900,
                                                  title="Correlation graph - validation",
                                                  x_axis_label="Reference charges",
                                                  y_axis_label="Empirical charges",
                                                  output_backend="webgl")
            correlation_graph_validation.title.align = "center"
            correlation_graph_validation.title.text_font_size = "17pt"
            correlation_graph_validation.xaxis.axis_label_text_font_size = "25px"
            correlation_graph_validation.yaxis.axis_label_text_font_size = "25px"
            correlation_graph_validation.axis.major_label_text_font_size = '20px'

            correlation_graph_validation.line([-1000, 1000], [-1000, 1000])
            zipped_charges = list(zip(self.ref_set_of_molecules_validation.atomic_types_charges.items(),
                                      self.set_of_molecules_validation.atomic_types_charges.items()))
            for index, ((atomic_symbol, ref_charges), (_, charges)) in enumerate(zipped_charges):
                try:
                    color = Category20[20][color_numbers[atomic_symbol]]
                except KeyError:
                    color = Category20[20][index % 20]
                correlation_graph_validation.circle(ref_charges, charges, size=6, legend=atomic_symbol, fill_color=color, line_color=color)
            correlation_graph_validation.legend.location = "top_left"
            correlation_graph_validation.legend.click_policy = "hide"
            correlation_graph_validation.x_range = Range1d(min_charge, max_charge)
            correlation_graph_validation.y_range = Range1d(min_charge, max_charge)

            comparison = figure(plot_width=900,
                                plot_height=900,
                                title="Correlation graph - validation",
                                x_axis_label="Reference charges",
                                y_axis_label="Empirical charges",
                                output_backend="webgl")
            comparison.title.align = "center"
            comparison.title.text_font_size = "17pt"
            comparison.xaxis.axis_label_text_font_size = "25px"
            comparison.yaxis.axis_label_text_font_size = "25px"
            comparison.axis.major_label_text_font_size = '20px'

            comparison.line([-1000, 1000], [-1000, 1000])
            comparison.circle(self.ref_set_of_molecules.ref_charges, self.set_of_molecules.all_charges, size=6, legend="Parameterization", fill_color="black", line_color="black")
            comparison.circle(self.ref_set_of_molecules_validation.ref_charges, self.set_of_molecules_validation.all_charges, size=6, legend="Validation", fill_color="red", line_color="red")
            comparison.legend.location = "top_left"
            comparison.legend.click_policy = "hide"
            comparison.x_range = Range1d(min_charge, max_charge)
            comparison.y_range = Range1d(min_charge, max_charge)
            self.correlation_graph_html_source = Tabs(tabs=[Panel(child=correlation_graph_parameterization, title="Parameterization"), Panel(child=correlation_graph_validation, title="Validation"), Panel(child=comparison, title="Comparison")])
        else:
            self.correlation_graph_html_source = correlation_graph_parameterization
        print(colored("ok\n", "green"))

    def write_html_comparison(self, charges_file, ref_charges_file, output_file):
        print("Writing html file...")
        copyfile(charges_file, path.join(self.data_dir, path.basename(charges_file)))
        copyfile(ref_charges_file, path.join(self.data_dir, path.basename(ref_charges_file)))
        ref_charges_file = path.basename(ref_charges_file)
        charges_file = path.basename(charges_file)
        script, correlation_graph = components(self.correlation_graph_html_source)
        output_file = path.join(self.data_dir, output_file)
        with open(output_file, "w") as html_file:
            html_file.write(open("modules/html_patterns/pattern_comparison.txt").read().format(
                script, INLINE.render(),
                "</td><td>\n".join([str(item) for item in self.all_atoms_data]),
                "</td><td>\n".join([str(item) for item in self.molecules_data]),
                "".join(["\n<tr style=\"background-color: {};\"><td>".format(background_color(atomic_type[1])) + "</td><td>".join([str(item) for item in atomic_type]) + "</td></tr>" for atomic_type in self.atomic_types_data]),
                correlation_graph, charges_file, charges_file, ref_charges_file, ref_charges_file))
        print(colored("ok\n", "green"))
        view(output_file)

    def write_html_parameterization(self, output_file, sdf, charges, ref_charges, summary_lines, parameters_json):
        print("Writing html file...")
        (script, (correlation_graph)) = components((self.correlation_graph_html_source))
        with open(output_file, "w") as html_file:
            html_file.write(open("modules/html_patterns/pattern_parameterization.txt").read().format(
                script, INLINE.render(),
                "</td><td>\n".join([str(item) for item in self.all_atoms_data]),
                "</td><td>\n".join([str(item) for item in self.all_atoms_data_validation]),
                "</td><td>\n".join([str(item) for item in self.molecules_data]),
                "</td><td>\n".join([str(item) for item in self.molecules_data_validation]),
                "".join(["\n<tr style=\"background-color: {};\"><td>".format(background_color(atomic_type[1])) + "</td><td>".join([str(item) for item in atomic_type]) + "</td></tr>" for atomic_type in self.atomic_types_data]),
                "".join(["\n<tr style=\"background-color: {};\"><td>".format(background_color(atomic_type[1])) + "</td><td>".join([str(item) for item in atomic_type]) + "</td></tr>" for atomic_type in self.atomic_types_data_validation]),
                correlation_graph,
                "".join(["\n<tr><td>" + "</td><td>".join([str(item) for item in bond_type]) + "</td></tr>" for bond_type in self.bonds_data]),
                "</br>\n".join(["<b>" + line.replace(": ", "</b>: ") for line in summary_lines]),
                parameters_json, sdf, sdf, charges, charges, ref_charges, ref_charges))
        print(colored("ok\n", "green"))
        view(output_file)
