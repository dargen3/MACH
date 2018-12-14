# -*- coding: cp1250 -*-
from .control_existing import control_existing_files
from .set_of_molecules import SetOfMolecules, SetOfMoleculesFromChargesFile
from .control_order_of_molecules import control_order_of_molecules
from termcolor import colored
from numpy import sqrt, mean, max, min, sum, corrcoef, log10, array, histogram, asarray, arange
from scipy.stats import pearsonr
from tabulate import tabulate
from sys import exit
from os import path, mkdir
from operator import itemgetter
from collections import namedtuple, Counter
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Category20
from bokeh.resources import INLINE
from bokeh.util.browser import view
from bokeh.embed import components
from bokeh.models import Legend, HoverTool
from operator import itemgetter
import csv
from shutil import copyfile




def background_color(value):
    return "green" if value < 0.05 else "#4ca64c" if value < 0.1 else "#99cc99" if value < 0.15\
        else "yellow" if value < 0.2 else "orange" if value < 0.3 else "red; color: white" if value < 0.4\
        else "darkred; color: white"


def delete_unrealistic_value_from_course(minimization):
    x_coordinates = []
    y_coordinates = []
    for index, value in enumerate(minimization):
        if value < 10000000000000000000000000000000:
            y_coordinates.append(value)
            x_coordinates.append(index)
    return x_coordinates, y_coordinates


def calculate_statistics(ref_charges, charges):
    deviations = abs(ref_charges - charges)
    rmsd = sqrt((1.0/deviations.size)*sum(deviations**2))
    max_deviation = max(deviations)
    average_deviation = mean(deviations)
    pearson_2 = corrcoef(ref_charges, charges)[0, 1] ** 2
    return [rmsd, max_deviation, average_deviation, pearson_2, deviations.size]


class Comparison:
    def __init__(self, ref_charges_data, charges_data, data_dir, rewriting_with_force, parameterization=False, course=None, course_parameters=None):
        self.data_dir = data_dir
        self.parameterization = parameterization
        if self.parameterization:
            self.course = course
            self.course_parameters = course_parameters
            set_of_molecules_nt = namedtuple("set_of_molecules", ("all_charges", "atomic_types_charges", "molecules"))
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
        molecules_statistical_data = [calculate_statistics(ref_molecule.charges, molecule.charges) for ref_molecule, molecule in zip(self.ref_set_of_molecules, self.set_of_molecules.molecules)]
        molecules_num_of_atoms = [molecule[4] for molecule in molecules_statistical_data]
        self.molecules_data = [round(item, 4) for item in [mean([x[y] for x in molecules_statistical_data]) for y in range(4)] + [self.ref_set_of_molecules.num_of_molecules, min(molecules_num_of_atoms), max(molecules_num_of_atoms), mean(molecules_num_of_atoms)]]
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
        print("Creating pictures...")
        if self.parameterization:
            c = figure(plot_width=900,
                       plot_height=900,
                       title="Course of minimization",
                       x_axis_label="Step",
                       y_axis_label="RMSD",
                       y_range=(0, 1))
            c.title.align = "center"
            c.title.text_font_size = "25px"
            c.xaxis.axis_label_text_font_size = "20px"
            c.yaxis.axis_label_text_font_size = "20px"
            if self.course[0] == "local_minimization":
                color = Category20[20][0]
                c_line = c.line(*delete_unrealistic_value_from_course(self.course[-1]), color=color, hover_color=color)
                c_line.hover_glyph.line_width = 2
            elif self.course[0] == "guided_minimization":
                for index, minimization in enumerate(self.course[1]):
                    color = Category20[20][(index*2) % 20]
                    c_line = c.line(*delete_unrealistic_value_from_course(minimization), color=color,
                           legend="{}. local minimization".format(index + 1), line_color=color, hover_color=color)
                    c_line.hover_glyph.line_width = 3
                    c.legend.click_policy = "hide"
                    c.legend.location = "top_right"
            c.add_tools(HoverTool(tooltips=[('RMSD', '@y'),
                                            ('Step', '@x')]))
            self.course = c

            p = figure(plot_width=900,
                       plot_height=900,
                       title="Course of minimization",
                       x_axis_label="Step",
                       y_axis_label="Value",
                       y_range=(-0.1, 1.1))
            p.title.align = "center"
            p.title.text_font_size = "25px"
            p.xaxis.axis_label_text_font_size = "20px"
            p.yaxis.axis_label_text_font_size = "20px"
            p_legend = []
            parameters = self.course_parameters[0]
            num_of_parameters = len(parameters)
            course_of_parameters = [self.course_parameters[1][x::num_of_parameters] for x in range(num_of_parameters)]
            steps = len(course_of_parameters[0])
            for key, index in parameters.items():
                color = Category20[20][index * 2 % 20]
                p_line = p.line(arange(steps), course_of_parameters[index], name=key, color=color, hover_color=color)
                p_line.hover_glyph.line_width = 3
                p_legend.append((key, [p_line]))
            legend = Legend(items=p_legend)
            legend.click_policy = "hide"
            p.add_layout(legend, 'right')
            p.add_tools(HoverTool(tooltips=[('Parameter', '$name'),
                                            ('Value', '@y'),
                                            ('Step', '@x')]))
            self.course_parameters = p

        color_numbers = {"H~1": 0, "O~1": 2, "O~2": 4, "N~1": 6, "N~2": 8, "C~1": 10, "C~2": 12, "S~1": 14,
                         "Ca~1": 16, "S~2": 18, "P~1": 1, "P~2": 3, "N~3": 5, "C~3": 7, "Br~1": 9, "Cl~1": 11,
                         "F~1": 13, "I~1": 15, "H": 0, "C": 2, "N": 4, "O": 6, "S": 8, "I": 10, "F": 12, "Br": 14, "Cl": 16, "P": 18}
        cg = figure(plot_width=900,
                   plot_height=900,
                   title="Correlation graph",
                   x_axis_label="Reference charges",
                   y_axis_label="Empirical Charges",
                   output_backend="webgl")
        cg.title.align = "center"
        cg.title.text_font_size = "25px"
        cg.xaxis.axis_label_text_font_size = "20px"
        cg.yaxis.axis_label_text_font_size = "20px"
        zipped_charges = list(zip(self.ref_set_of_molecules.atomic_types_charges.items(),
                                      self.set_of_molecules.atomic_types_charges.items()))
        for (atomic_symbol, ref_charges), (_, charges) in zipped_charges:
            color = Category20[20][color_numbers[atomic_symbol]]
            cg.circle(ref_charges, charges, size=6, legend=atomic_symbol, fill_color=color, line_color=color)
        cg.legend.location = "top_left"
        cg.legend.click_policy = "hide"
        self.correlation_graph_html_source = cg
        print(colored("ok\n", "green"))

    def write_html_comparison(self, charges_file, ref_charges_file, output_file):
        print("Writing html file...")
        copyfile(charges_file, path.join(self.data_dir, path.basename(charges_file)))
        copyfile(ref_charges_file, path.join(self.data_dir, path.basename(ref_charges_file)))
        ref_charges_file = path.basename(ref_charges_file)
        charges_file = path.basename(charges_file)
        script, correlation_graph = components(self.correlation_graph_html_source)
        output_file = path.join(self.data_dir,output_file)
        with open(output_file, "w") as html_file:
            html_file.write(open("modules/html_patterns/pattern_comparison.txt").read().format(
            script, INLINE.render(),
           "</td><td>\n".join([str(item) for item in self.all_atoms_data]),
           "</td><td>\n".join([str(item) for item in self.molecules_data]),
           "".join(["\n<tr style=\"background-color: {};\"><td>".format(background_color(atomic_type[1])) + "</td><td>".join([str(item) for item in atomic_type]) + "</td></tr>" for atomic_type in self.atomic_types_data]),
            correlation_graph, charges_file, charges_file, ref_charges_file, ref_charges_file))
        view(output_file)

    def write_html_parameterization(self, output_file, sdf, charges, ref_charges, summary_lines, parameters_json):
        print("Writing html file...")
        (script, (correlation_graph, course, course_parameters)) = components((self.correlation_graph_html_source, self.course, self.course_parameters))
        with open(output_file, "w") as html_file:
            html_file.write(open("modules/html_patterns/pattern_parameterization.txt").read().format(
            script, INLINE.render(),
           "</td><td>\n".join([str(item) for item in self.all_atoms_data]),
           "</td><td>\n".join([str(item) for item in self.molecules_data]),
           "".join(["\n<tr style=\"background-color: {};\"><td>".format(background_color(atomic_type[1])) + "</td><td>".join([str(item) for item in atomic_type]) + "</td></tr>" for atomic_type in self.atomic_types_data]),
            correlation_graph,
           "".join(["\n<tr><td>" + "</td><td>".join([str(item) for item in bond_type]) + "</td></tr>" for bond_type in self.bonds_data]),
           course, course_parameters,
           "</br>\n".join(["<b>" + line.replace(": ", "</b>: ") for line in summary_lines]),
           parameters_json, sdf, sdf, charges, charges, ref_charges, ref_charges))
        print(colored("ok\n", "green"))
        view(output_file)

