# -*- coding: utf-8 -*-
from collections import Counter, defaultdict
from os import path, mkdir
from shutil import copyfile

from bokeh.embed import components
from bokeh.models import Legend, Range1d, Label, ColumnDataSource
from bokeh.models.widgets import Panel, Tabs
from bokeh.palettes import Reds, Greens, Blues, Greys
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.util.browser import view
from numpy import sqrt, mean, max, min, sum, corrcoef
from termcolor import colored

from .control_existing import control_existing_files
from .set_of_molecules import create_set_of_molecules_from_chg_files


def background_color(value):
    return "green" if value < 0.05 else "#4ca64c" if value < 0.1 else "#99cc99" if value < 0.15 \
        else "yellow" if value < 0.2 else "orange" if value < 0.3 else "red; color: white" if value < 0.4 \
        else "darkred; color: white"


def calculate_statistics(ref_charges, emp_charges):
    deviations = abs(ref_charges - emp_charges)
    rmsd = sqrt((1.0 / deviations.size) * sum(deviations ** 2))
    max_deviation = max(deviations)
    average_deviation = mean(deviations)
    pearson_2 = corrcoef(ref_charges, emp_charges)[0, 1] ** 2
    return [rmsd, max_deviation, average_deviation, pearson_2, deviations.size]


class Comparison:
    def comparison(self, ref_chg_file, emp_chg_file, data_dir, rewriting_with_force):
        self.data_dir = data_dir
        self.set_of_molecules_parameterization = None
        self.output_file = "{}_{}.html".format(path.basename(emp_chg_file).split(".")[0],
                                               path.basename(ref_chg_file).split(".")[0])
        control_existing_files(((ref_chg_file, True, "file"),
                                (emp_chg_file, True, "file"),
                                (self.output_file, False, "file"),
                                (self.data_dir, False, "directory")),
                               rewriting_with_force)
        mkdir(self.data_dir)
        self.set_of_molecules = create_set_of_molecules_from_chg_files(ref_chg_file, emp_chg_file)
        self.atomic_types = set([atom for molecule in self.set_of_molecules.molecules for atom in molecule.atoms_representation])
        self.statistics_comparison()
        self.graphs()
        self.write_html_comparison()

    def parameterization(self, set_of_molecules_parameterization, set_of_molecules_validation, output_file, sdf_file, emp_chg_file, ref_chg_file, summary_lines, parameters_json, atomic_types):
        self.set_of_molecules_parameterization = set_of_molecules_parameterization
        self.set_of_molecules = self.set_of_molecules_parameterization
        self.set_of_molecules_validation = set_of_molecules_validation
        self.atomic_types = atomic_types
        self.statistics_parameterization()
        self.graphs()
        self.write_html_parameterization(output_file, sdf_file, emp_chg_file, ref_chg_file, summary_lines, parameters_json)

    def statistics_comparison(self):
        print("Calculating statistical data...")
        self.all_atoms_data = [round(item, 4) for item in
                               calculate_statistics(self.set_of_molecules.ref_charges,
                                                    self.set_of_molecules.emp_charges)]
        self.atomic_types_data = []
        for (atomic_type, ref_charges), (atomic_type2, emp_charges) in zip(self.set_of_molecules.ref_atomic_types_charges.items(),
                                                                           self.set_of_molecules.emp_atomic_types_charges.items()):
            atomic_type_data = [round(item, 4) for item in calculate_statistics(ref_charges, emp_charges)]
            self.atomic_types_data.append([atomic_type] + atomic_type_data + [round(atomic_type_data[4] / (self.all_atoms_data[4] / 100), 2)])
        molecules_statistical_data = [calculate_statistics(molecule.ref_charges, molecule.emp_charges) for molecule in self.set_of_molecules.molecules]
        molecules_num_of_atoms = [molecule[4] for molecule in molecules_statistical_data]
        self.molecules_data = [round(item, 4) for item in [mean([x[y] for x in molecules_statistical_data]) for y in range(4)] + [self.set_of_molecules.num_of_molecules, min(molecules_num_of_atoms), max(molecules_num_of_atoms), mean(molecules_num_of_atoms)]]
        print(colored("ok\n", "green"))

    def statistics_parameterization(self):
        print("Calculating statistical data...")
        self.all_atoms_data = [round(item, 4) for item in
                               calculate_statistics(self.set_of_molecules_parameterization.ref_charges,
                                                    self.set_of_molecules_parameterization.emp_charges)]
        self.atomic_types_data_parameterization = []
        for atomic_symbol in self.atomic_types:
            atomic_type_data = [round(item, 4) for item in calculate_statistics(self.set_of_molecules_parameterization.ref_atomic_types_charges[atomic_symbol], self.set_of_molecules_parameterization.emp_atomic_types_charges[atomic_symbol])]
            self.atomic_types_data_parameterization.append([atomic_symbol] + atomic_type_data + [round(atomic_type_data[4] / (self.all_atoms_data[4] / 100), 2)])
        self.atoms_data_validation = [round(item, 4) for item in
                                      calculate_statistics(self.set_of_molecules_validation.ref_charges,
                                                           self.set_of_molecules_validation.emp_charges)]
        self.atomic_types_data_validation = []
        for atomic_symbol in self.atomic_types:
            # try ... except construct is necessary, because validation set dont have to contain all atomic types contained in parameterization set
            try:
                atomic_type_data = [round(item, 4) for item in calculate_statistics(self.set_of_molecules_validation.ref_atomic_types_charges[atomic_symbol], self.set_of_molecules_validation.emp_atomic_types_charges[atomic_symbol])]
                self.atomic_types_data_validation.append([atomic_symbol] + atomic_type_data + [round(atomic_type_data[4] / (self.atoms_data_validation[4] / 100), 2)])
            except KeyError:
                continue
        molecules_statistical_data_parameterization = [calculate_statistics(molecule.ref_charges, molecule.emp_charges) for molecule in self.set_of_molecules_parameterization.molecules]
        molecules_num_of_atoms_parameterization = [molecule[4] for molecule in molecules_statistical_data_parameterization]
        self.molecules_data_parameterization = [round(item, 4) for item in [mean([x[y] for x in molecules_statistical_data_parameterization]) for y in range(4)] + [self.set_of_molecules_parameterization.num_of_molecules, min(molecules_num_of_atoms_parameterization), max(molecules_num_of_atoms_parameterization), mean(molecules_num_of_atoms_parameterization)]]
        molecules_statistical_data_validaton = [calculate_statistics(molecule.ref_charges, molecule.emp_charges) for molecule in self.set_of_molecules_validation.molecules]
        molecules_num_of_atoms_validation = [molecule[4] for molecule in molecules_statistical_data_validaton]
        self.molecules_data_validation = [round(item, 4) for item in [mean([x[y] for x in molecules_statistical_data_validaton]) for y in range(4)] + [self.set_of_molecules_validation.num_of_molecules, min(molecules_num_of_atoms_validation), max(molecules_num_of_atoms_validation), mean(molecules_num_of_atoms_validation)]]
        counter_bonds_parameterization = Counter()
        for molecule in self.set_of_molecules_parameterization.molecules:
            counter_bonds_parameterization.update(molecule.bonds_representation)

        counter_bonds_validation = Counter()
        for molecule in self.set_of_molecules_validation.molecules:
            counter_bonds_validation.update(molecule.bonds_representation)
        self.bonds_data = [(bond.replace("-", "  "), count_parameterization, counter_bonds_validation[bond]) for bond, count_parameterization in counter_bonds_parameterization.most_common()]
        print(colored("ok\n", "green"))

    def graphs(self):
        print("Creating graphs...")
        type_color = {"C": Reds[256], "O": Greens[256], "H": Blues[256], "N": Greys[256]}
        colors = {}
        for element in sorted(list(set([x.split("~")[0] for x in self.atomic_types]))):
            element_atomic_types = [x for x in self.atomic_types if x.split("~")[0] == element]
            for index, element_atomic_type in enumerate(element_atomic_types, 1):
                try:
                    if element_atomic_type[0] == "S":
                        colors[element_atomic_type] = ["#FF00FF", "#FF33FF", "#FF66FF", "#FF99FF"][index - 1]
                    else:
                        colors[element_atomic_type] = type_color[element][int(200 / (len(element_atomic_types) + 1)) * index]
                except KeyError:
                    exit(colored("Error! No color defined for atomic type {}.".format(atomic_symbol), "red"))

        tooltips = [("Atomic type", "$name"),
                    ("Reference charge", "@ref_charges"),
                    ("Empirical charge", "@emp_charges"),
                    ("Molecule", "@molecules"),
                    ("Index", "@indices")]

        correlation_graph = figure(plot_width=900,
                                   plot_height=900,
                                   title="Correlation graph",
                                   x_axis_label="Reference charges",
                                   y_axis_label="Empirical charges",
                                   output_backend="webgl",
                                   tooltips=tooltips)
        correlation_graph.toolbar.active_inspect = None

        correlation_graph.title.align = "center"
        correlation_graph.title.text_font_size = "17pt"
        correlation_graph.xaxis.axis_label_text_font_size = "25px"
        correlation_graph.yaxis.axis_label_text_font_size = "25px"
        correlation_graph.axis.major_label_text_font_size = '20px'
        correlation_graph.line([-1000, 1000], [-1000, 1000])
        legends_p = [[] for _ in range(len(self.set_of_molecules.ref_atomic_types_charges.items()) // 27 + 1)]

        source_molecules = defaultdict(list)
        source_indices = defaultdict(list)
        for molecule in self.set_of_molecules.molecules:
            for index, symbol in enumerate(molecule.atoms_representation, start=1):
                source_molecules[symbol].append(molecule.name)
                source_indices[symbol].append(index)

        for index, atomic_symbol in enumerate(self.atomic_types):
            color = colors[atomic_symbol]
            oc = correlation_graph.circle("ref_charges",
                                          "emp_charges",
                                          size=6,
                                          fill_color=color,
                                          line_color=color,
                                          name=atomic_symbol,
                                          source=ColumnDataSource(data=dict(emp_charges=self.set_of_molecules.emp_atomic_types_charges[atomic_symbol],
                                                                            ref_charges=self.set_of_molecules.ref_atomic_types_charges[atomic_symbol],
                                                                            indices=source_indices[atomic_symbol],
                                                                            molecules=source_molecules[atomic_symbol])))
            legends_p[index // 27].append((atomic_symbol, [oc]))

        plot_width = 944
        rmsd_pearson_labels_x = [577, 630]
        par_val_comparison_width = 994
        print(self.atomic_types)
        if "/" not in list(self.atomic_types)[0]:
            label_width = 50
        else:
            label_width = 130
            plot_width += 42
            par_val_comparison_width += 252

        for x in range(len(self.atomic_types) // 27 + 1):
            correlation_graph.add_layout(Legend(items=legends_p[x], label_width=label_width, label_height=25, margin=0, label_text_font_size="19px"), "left")
            plot_width += label_width
        correlation_graph.legend.click_policy = "hide"
        correlation_graph.plot_width = plot_width

        rmsd_label = Label(x=rmsd_pearson_labels_x[0], y=35, x_units='screen', y_units='screen',
                           text='RMSD: {}'.format(self.all_atoms_data[0]), render_mode='css', text_font_size="25px")
        pearson_label = Label(x=rmsd_pearson_labels_x[1], y=11, x_units='screen', y_units='screen',
                              text="R²: {}".format(self.all_atoms_data[3]), render_mode='css', text_font_size="25px")
        correlation_graph.add_layout(rmsd_label)
        correlation_graph.add_layout(pearson_label)

        max_charge = max((max(self.set_of_molecules.ref_charges), max(self.set_of_molecules.emp_charges)))
        min_charge = min((min(self.set_of_molecules.ref_charges), min(self.set_of_molecules.emp_charges)))
        corr = (max_charge - min_charge) / 10
        min_charge -= corr
        max_charge += corr
        correlation_graph.x_range = Range1d(min_charge, max_charge)
        correlation_graph.y_range = Range1d(min_charge, max_charge)
        if self.set_of_molecules_parameterization:

            source_molecules_v = defaultdict(list)
            source_indices_v = defaultdict(list)
            for molecule in self.set_of_molecules_validation.molecules:
                for index, symbol in enumerate(molecule.atoms_representation, start=1):
                    source_molecules_v[symbol].append(molecule.name)
                    source_indices_v[symbol].append(index)

            correlation_graph_validation = figure(plot_width=900,
                                                  plot_height=900,
                                                  title="Correlation graph - validation",
                                                  x_axis_label="Reference charges",
                                                  y_axis_label="Empirical charges",
                                                  output_backend="webgl",
                                                  tooltips=tooltips)
            correlation_graph_validation.toolbar.active_inspect = None
            correlation_graph_validation.title.align = "center"
            correlation_graph_validation.title.text_font_size = "17pt"
            correlation_graph_validation.xaxis.axis_label_text_font_size = "25px"
            correlation_graph_validation.yaxis.axis_label_text_font_size = "25px"
            correlation_graph_validation.axis.major_label_text_font_size = '20px'
            correlation_graph_validation.line([-1000, 1000], [-1000, 1000])
            legends_v = [[] for _ in range(len(self.atomic_types) // 27 + 1)]
            missed_types = 0
            for index, atomic_symbol in enumerate(self.atomic_types):
                color = colors[atomic_symbol]
                # try ... except construct is necessary, because validation set dont have to contain all atomic types contained in parameterization set
                try:
                    ref_charges = self.set_of_molecules_validation.ref_atomic_types_charges[atomic_symbol]
                    emp_charges = self.set_of_molecules_validation.emp_atomic_types_charges[atomic_symbol]
                except KeyError:
                    missed_types += 1
                    continue

                ov = correlation_graph_validation.circle("ref_charges",
                                                         "emp_charges",
                                                         size=6,
                                                         fill_color=color,
                                                         line_color=color,
                                                         name=atomic_symbol,
                                                         source=ColumnDataSource(data=dict(emp_charges=emp_charges,
                                                                                           ref_charges=ref_charges,
                                                                                           indices=source_indices_v[atomic_symbol],
                                                                                           molecules=source_molecules_v[atomic_symbol])))
                legends_v[(index - missed_types) // 27].append((atomic_symbol, [ov]))

            for x in range(len(self.atomic_types) // 27 + 1):
                correlation_graph_validation.add_layout(Legend(items=legends_v[x], label_width=label_width, label_height=25, margin=0, label_text_font_size="19px"), "left")
            correlation_graph_validation.legend.click_policy = "hide"
            correlation_graph_validation.plot_width = plot_width

            rmsd_label = Label(x=rmsd_pearson_labels_x[0], y=35, x_units='screen', y_units='screen',
                               text='RMSD: {}'.format(self.atoms_data_validation[0]), render_mode='css', text_font_size="25px")
            pearson_label = Label(x=rmsd_pearson_labels_x[1], y=11, x_units='screen', y_units='screen',
                                  text="R²: {}".format(self.atoms_data_validation[3]), render_mode='css', text_font_size="25px")
            correlation_graph_validation.add_layout(rmsd_label)
            correlation_graph_validation.add_layout(pearson_label)

            correlation_graph_validation.x_range = Range1d(min_charge, max_charge)
            correlation_graph_validation.y_range = Range1d(min_charge, max_charge)

            par_val_comparison = figure(plot_width=par_val_comparison_width,
                                        plot_height=900,
                                        title="Correlation graph - validation",
                                        x_axis_label="Reference charges",
                                        y_axis_label="Empirical charges",
                                        output_backend="webgl")
            par_val_comparison.title.align = "center"
            par_val_comparison.title.text_font_size = "17pt"
            par_val_comparison.xaxis.axis_label_text_font_size = "25px"
            par_val_comparison.yaxis.axis_label_text_font_size = "25px"
            par_val_comparison.axis.major_label_text_font_size = '20px'
            par_val_comparison.line([-1000, 1000], [-1000, 1000])
            par_val_comparison.circle(self.set_of_molecules_parameterization.ref_charges, self.set_of_molecules_parameterization.emp_charges, size=6, legend="Parameterization", fill_color="black", line_color="black")
            par_val_comparison.circle(self.set_of_molecules_validation.ref_charges, self.set_of_molecules_validation.emp_charges, size=6, legend="Validation", fill_color="red", line_color="red")
            par_val_comparison.legend.location = "top_left"
            par_val_comparison.legend.click_policy = "hide"
            par_val_comparison.x_range = Range1d(min_charge, max_charge)
            par_val_comparison.y_range = Range1d(min_charge, max_charge)
            self.correlation_graph_html_source = Tabs(tabs=[Panel(child=correlation_graph, title="Parameterization"), Panel(child=correlation_graph_validation, title="Validation"), Panel(child=par_val_comparison, title="Comparison")])
        else:
            self.correlation_graph_html_source = correlation_graph
        print(colored("ok\n", "green"))

    def write_html_comparison(self):
        print("Writing html file...")
        ref_chg_file = path.basename(self.set_of_molecules.ref_chg_file)
        emp_chg_file = path.basename(self.set_of_molecules.emp_chg_file)
        copyfile(self.set_of_molecules.emp_chg_file, path.join(self.data_dir, ref_chg_file))
        copyfile(self.set_of_molecules.ref_chg_file, path.join(self.data_dir, emp_chg_file))
        script, correlation_graph = components(self.correlation_graph_html_source)
        output_file = path.join(self.data_dir, self.output_file)
        with open(output_file, "w") as html_file:
            html_file.write(open("modules/html_patterns/pattern_comparison.txt").read().format(
                script, INLINE.render(),
                "</td><td>\n".join([str(item) for item in self.all_atoms_data]),
                "</td><td>\n".join([str(item) for item in self.molecules_data]),
                "".join(["\n<tr style=\"background-color: {};\"><td>".format(background_color(atomic_type[1])) + "</td><td>".join([str(item) for item in atomic_type]) + "</td></tr>" for atomic_type in self.atomic_types_data]),
                correlation_graph, emp_chg_file, emp_chg_file, ref_chg_file, ref_chg_file))
        print(colored("ok\n", "green"))
        view(output_file)

    def write_html_parameterization(self, output_file, sdf_file, emp_chg_file, ref_chg_file, summary_lines, parameters_json):
        print("Writing html file...")
        (script, (correlation_graph)) = components((self.correlation_graph_html_source))
        with open(output_file, "w") as html_file:
            html_file.write(open("modules/html_patterns/pattern_parameterization.txt").read().format(
                script, INLINE.render(),
                "</td><td>\n".join([str(item) for item in self.all_atoms_data]),
                "</td><td>\n".join([str(item) for item in self.atoms_data_validation]),
                "</td><td>\n".join([str(item) for item in self.molecules_data_parameterization]),
                "</td><td>\n".join([str(item) for item in self.molecules_data_validation]),
                "".join(["\n<tr style=\"background-color: {};\"><td>".format(background_color(atomic_type[1])) + "</td><td>".join([str(item) for item in atomic_type]) + "</td></tr>" for atomic_type in self.atomic_types_data_parameterization]),
                "".join(["\n<tr style=\"background-color: {};\"><td>".format(background_color(atomic_type[1])) + "</td><td>".join([str(item) for item in atomic_type]) + "</td></tr>" for atomic_type in self.atomic_types_data_validation]),
                correlation_graph,
                "".join(["\n<tr><td>" + "</td><td>".join([str(item) for item in bond_type]) + "</td></tr>" for bond_type in self.bonds_data]),
                "</br>\n".join(["<b>" + line.replace(": ", "</b>: ") for line in summary_lines]),
                parameters_json, sdf_file, sdf_file, emp_chg_file, emp_chg_file, ref_chg_file, ref_chg_file))
        print(colored("ok\n", "green"))
        view(output_file)
