# -*- coding: utf-8 -*-
from collections import Counter, defaultdict, namedtuple
import resource
import bokeh as bk
import numpy as np
from bokeh.plotting import figure
from termcolor import colored

from .control_input import control_and_copy_input_files
from .set_of_molecules import SetOfMolecules, create_set_of_mols, add_chgs


def comparison(sdf_file: str,
               ref_chgs_file: str,
               emp_chgs_file: str,
               ats_types_pattern: str,
               data_dir: str,
               rewriting_with_force: bool):

    control_and_copy_input_files(data_dir,
                                 (sdf_file, ref_chgs_file, emp_chgs_file),
                                 rewriting_with_force)

    set_of_mols = create_set_of_mols(sdf_file, ats_types_pattern)
    add_chgs(set_of_mols, emp_chgs_file, "emp_chgs")
    add_chgs(set_of_mols, ref_chgs_file, "ref_chgs")

    print("Calculation of statistical data...")
    stats = _stats(set_of_mols, f"{data_dir}/output_files/molecules.log")
    print(colored("ok\n", "green"))

    print("Creation of correlation graph...")
    corr_graph = _corr_graph(set_of_mols,
                             _colors(set_of_mols.ats_types),
                             stats,
                             "Correlation graph",
                             _graph_ranges(np.concatenate((set_of_mols.ref_chgs,
                                                           set_of_mols.emp_chgs)))),
    print(colored("ok\n", "green"))

    print("Writing html file...")
    _write_html_comparison(set_of_mols,
                           data_dir,
                           corr_graph,
                           stats)
    print(colored("ok\n", "green"))


def comparison_par(set_of_mols_par: SetOfMolecules,
                   set_of_mols_val: SetOfMolecules,
                   params_file: str,
                   data_dir: str,
                   loc_min_courses: list,
                   par_info: list):

    print("Calculation of statistical data...")
    stats_par = _stats(set_of_mols_par, f"{data_dir}/output_files/molecules_parameterization.log")
    stats_val = _stats(set_of_mols_val, f"{data_dir}/output_files/molecules_validation.log")
    print(colored("ok\n", "green"))

    print("Creation of graphs...")
    colors = _colors(set_of_mols_par.ats_types)
    graph_ranges = _graph_ranges(np.concatenate((set_of_mols_par.ref_chgs,
                                                 set_of_mols_par.emp_chgs,
                                                 set_of_mols_val.ref_chgs,
                                                 set_of_mols_val.emp_chgs)))
    tabs = [bk.models.widgets.Panel(child=_corr_graph(set_of_mols_par,
                                                      colors,
                                                      stats_par,
                                                      "Correlation graph - parameterization",
                                                      graph_ranges),
                                    title="Parameterization"),
            bk.models.widgets.Panel(child=_corr_graph(set_of_mols_val,
                                                      colors,
                                                      stats_val,
                                                      "Correlation graph - validation",
                                                      graph_ranges),
                                    title="Validation"),
            bk.models.widgets.Panel(child=_comparison_corr_graph(set_of_mols_par,
                                                                 set_of_mols_val,
                                                                 stats_par,
                                                                 stats_val,
                                                                 graph_ranges),
                                    title="Comparison")]
    loc_min_graph = _graph_loc_min(loc_min_courses)
    print(colored("ok\n", "green"))

    print("Writing html file...")
    _write_html_par(f"{data_dir}/output_files/parameterization.html",
                    bk.models.widgets.Tabs(tabs=tabs),
                    loc_min_graph,
                    stats_par,
                    stats_val,
                    set_of_mols_par,
                    par_info,
                    params_file)
    print(colored("ok\n", "green"))


def _calculate_stats(ref_chgs: np.array,
                     emp_chgs: np.array) -> namedtuple:

    deviations = abs(ref_chgs - emp_chgs)
    rmsd = round(np.sqrt((1.0 / deviations.size) * np.sum(deviations ** 2)), 4)
    max_deviation = round(np.max(deviations), 4)
    average_deviation = round(np.mean(deviations), 4)
    pearson_2 = round(np.corrcoef(ref_chgs, emp_chgs)[0, 1] ** 2, 4)
    return namedtuple("stats", ["rmsd",
                                "max_deviation",
                                "average_deviation",
                                "pearson_2",
                                "count"])(rmsd,
                                          max_deviation,
                                          average_deviation,
                                          pearson_2,
                                          deviations.size)


def _stats(set_of_mols: SetOfMolecules,
           mols_log_file: str) -> namedtuple:

    all_ats_data = _calculate_stats(set_of_mols.ref_chgs, set_of_mols.emp_chgs)

    ats_types_data = []
    for at_type in set_of_mols.ats_types:
        at_type_data = _calculate_stats(set_of_mols.ref_ats_types_chgs[at_type],
                                        set_of_mols.emp_ats_types_chgs[at_type])
        percent = round(at_type_data[4] / (all_ats_data[4] / 100), 2)
        ats_types_data.append([at_type,
                               *at_type_data,
                               percent])

    mols_data = np.round([_calculate_stats(mol.ref_chgs, mol.emp_chgs) for mol in set_of_mols.mols], 4)
    with open(mols_log_file, "w") as mols_log_file:
        mols_log_file.write("name, atomic types, rmsd, maximum deviation,"
                            "average deviation, pearson**2, number of atoms\n")
        for mol, (rmsd, max_dev, av_dev, pearson, num_of_at) in zip(set_of_mols.mols, mols_data):
            mols_log_file.write(f"{mol.name}, {';'.join(sorted(set(mol.ats_srepr)))}, {rmsd}, "
                                f"{max_dev}, {av_dev}, {pearson}, {int(num_of_at)}\n")
    mols_num_of_at = [mol.num_of_ats for mol in set_of_mols.mols]
    averaged_mols_data = [*[round(np.mean(mols_data[:, y]), 4) for y in range(4)],
                          set_of_mols.num_of_mols,
                          np.min(mols_num_of_at),
                          np.max(mols_num_of_at),
                          int(np.mean(mols_num_of_at))]

    counter_bonds = Counter([bond for mol in set_of_mols.mols for bond in mol.bonds_srepr])
    return namedtuple("stat_data", ["all_ats",
                                    "ats_types",
                                    "mols",
                                    "bonds"])(all_ats_data,
                                              ats_types_data,
                                              averaged_mols_data,
                                              counter_bonds)


def _corr_graph(set_of_mols: SetOfMolecules,
                colors: dict,
                stats: namedtuple,
                title_of_graph: str,
                graph_ranges: tuple) -> figure:

    graph = figure(plot_width=1050,
                   plot_height=900,
                   title=title_of_graph,
                   x_axis_label="Reference charges",
                   y_axis_label="Empirical charges",
                   output_backend="webgl",
                   tooltips=[("Atomic type", "$name"),
                             ("Reference charge", "@ref_chg"),
                             ("Empirical charge", "@emp_chg"),
                             ("Molecule", "@mols"),
                             ("Index", "@indices")])
    graph.toolbar.active_inspect = None
    graph.title.align = "center"
    graph.title.text_font_size = "17pt"
    graph.xaxis.axis_label_text_font_size = "25px"
    graph.yaxis.axis_label_text_font_size = "25px"
    graph.axis.major_label_text_font_size = '20px'
    graph.x_range = bk.models.Range1d(*graph_ranges)
    graph.y_range = bk.models.Range1d(*graph_ranges)

    source_mols_names = defaultdict(list)
    source_indices = defaultdict(list)
    for mol in set_of_mols.mols:
        for index, symbol in enumerate(mol.ats_srepr, start=1):
            source_mols_names[symbol].append(mol.name)
            source_indices[symbol].append(index)

    graph.line([-1000, 1000], [-1000, 1000])
    legends = []
    for index, at_symbol in enumerate(set_of_mols.ats_types):
        color = colors[at_symbol]
        oc = graph.circle("ref_chg",
                          "emp_chg",
                          size=6,
                          fill_color=color,
                          line_color=color,
                          name=at_symbol,
                          source=bk.models.ColumnDataSource(
                              data=dict(emp_chg=set_of_mols.emp_ats_types_chgs[at_symbol],
                                        ref_chg=set_of_mols.ref_ats_types_chgs[at_symbol],
                                        indices=source_indices[at_symbol],
                                        mols=source_mols_names[at_symbol])))
        legends.append((at_symbol, [oc]))

    graph.add_layout(bk.models.Legend(items=legends,
                                      label_width=100,
                                      label_height=20,
                                      margin=0,
                                      label_text_font_size="15px",
                                      border_line_color="white"), "left")
    graph.legend.click_policy = "hide"

    graph.add_layout(bk.models.Label(x=613,
                                     y=35,
                                     x_units='screen',
                                     y_units='screen',
                                     text=f'RMSD: {stats.all_ats.rmsd}',
                                     render_mode='css',
                                     text_font_size="25px"))
    graph.add_layout(bk.models.Label(x=660,
                                     y=11,
                                     x_units='screen',
                                     y_units='screen',
                                     text=f'R²: {stats.all_ats.pearson_2}',
                                     render_mode='css',
                                     text_font_size="25px"))
    return graph


def _comparison_corr_graph(set_of_mol_par: SetOfMolecules,
                           set_of_mol_val: SetOfMolecules,
                           stats_par: namedtuple,
                           stats_val: namedtuple,
                           graph_ranges: tuple) -> figure:

    graph = figure(plot_width=1050,
                   plot_height=900,
                   title="Correlation graph - parameterization & validation",
                   x_axis_label="Reference charges",
                   y_axis_label="Empirical charges",
                   output_backend="webgl")
    graph.title.align = "center"
    graph.title.text_font_size = "17pt"
    graph.xaxis.axis_label_text_font_size = "25px"
    graph.yaxis.axis_label_text_font_size = "25px"
    graph.axis.major_label_text_font_size = '20px'
    graph.x_range = bk.models.Range1d(*graph_ranges)
    graph.y_range = bk.models.Range1d(*graph_ranges)

    graph.line([-1000, 1000], [-1000, 1000])
    graph.circle(set_of_mol_val.ref_chgs,
                 set_of_mol_val.emp_chgs,
                 size=6,
                 legend="Validation",
                 fill_color="red",
                 line_color="red")
    graph.circle(set_of_mol_par.ref_chgs,
                 set_of_mol_par.emp_chgs,
                 size=6,
                 legend="Parameterization",
                 fill_color="black",
                 line_color="black")
    graph.legend.location = "top_left"
    graph.legend.click_policy = "hide"

    graph.add_layout(bk.models.Label(x=596,
                                     y=35,
                                     x_units='screen',
                                     y_units='screen',
                                     text=f'ΔRMSD: {round(abs(stats_par.all_ats.rmsd - stats_val.all_ats.rmsd), 4)}',
                                     render_mode='css',
                                     text_font_size="25px"))
    graph.add_layout(bk.models.Label(x=643,
                                     y=11,
                                     x_units='screen',
                                     y_units='screen',
                                     text=f'ΔR²: {round(abs(stats_par.all_ats.pearson_2 - stats_val.all_ats.pearson_2), 4)}',
                                     render_mode='css',
                                     text_font_size="25px"))
    return graph


def _graph_ranges(chg: np.array) -> tuple:

    max_chg = np.max(chg)
    min_chg = np.min(chg)
    corr = (max_chg - min_chg) / 10
    return min_chg - corr, max_chg + corr


def _graph_loc_min(loc_min_courses: list) -> figure:
    graph = figure(plot_width=1050,
                   plot_height=900,
                   title="Local minimization progress from multiple points",
                   x_axis_label="Step",
                   x_axis_type="log",
                   y_axis_label="Objective value",
                   output_backend="webgl",)
    graph.title.align = "center"
    graph.title.text_font_size = "17pt"
    graph.xaxis.axis_label_text_font_size = "25px"
    graph.yaxis.axis_label_text_font_size = "25px"
    graph.axis.major_label_text_font_size = '20px'
    graph.x_range.start = 0
    graph.y_range.start = 0
    graph.y_range.end = 2
    graph.legend.location = "top_right"
    graph.legend.click_policy = "hide"
    pallete = bk.palettes.Category20[20]
    for index, course in enumerate(loc_min_courses):
        graph.line([x for x in range(len(course))],
                   course,
                   color=pallete[index % 20],
                   line_width=3,
                   legend=str(index+1))
    return graph


def _colors(at_types: list) -> dict:

    type_color = {"C": bk.palettes.Reds[256],
                  "O": bk.palettes.Greens[256],
                  "H": bk.palettes.Blues[256],
                  "N": bk.palettes.Greys[256]}
    colors = {}
    for element in sorted(set([x.split("/")[0] for x in at_types])):
        element_ats_types = [x for x in at_types if x.split("/")[0] == element]
        for index, element_ats_type in enumerate(element_ats_types, 1):
            try:
                if element_ats_type[0] == "S":
                    colors[element_ats_type] = ["#FF00FF", "#FF33FF", "#FF66FF", "#FF99FF"][index - 1]
                else:
                    colors[element_ats_type] = type_color[element][int(200 / (len(element_ats_types) + 1)) * index]
            except KeyError:
                colors[element_ats_type] = "black"
    return colors


def _background_color(value: float) -> str:
    return "green" if value < 0.05 else "#4ca64c" if value < 0.1 else "#99cc99" if value < 0.15 \
        else "yellow" if value < 0.2 else "orange" if value < 0.3 else "red; color: white" if value < 0.4 \
        else "darkred; color: white"


def _format_input_file(file: str) -> tuple:
    return f"../input_files/{file.split('/')[-1]}", f"../input_files/{file.split('/')[-1]}"


def _write_html_comparison(set_of_mols: SetOfMolecules,
                           data_dir: str,
                           correlation_graph: figure,
                           stats: namedtuple):

    script, correlation_graph_html_source = bk.embed.components(correlation_graph)
    output_file = f"{data_dir}/output_files/comparison.html"
    with open(output_file, "w") as html_file:
        html_file.write(open("modules/html_patterns/pattern_comparison.txt").read().format(
            script, bk.resources.INLINE.render(),
            "</td><td>\n".join([str(item) for item in stats.all_ats]),
            "</td><td>\n".join([str(item) for item in stats.mols]),
            "".join([f"\n<tr style=\"background-color: {_background_color(at_type[1])};\"><td>" +
                     "</td><td>".join([str(item) for item in at_type]) + "</td></tr>" for at_type in stats.ats_types]),
            correlation_graph_html_source[0],
            "".join(["\n<tr><td>" + "</td><td>".join([str(item) for item in bond_type]) + "</td></tr>"
                     for bond_type in stats.bonds.items()]),
            *_format_input_file(set_of_mols.sdf_file),
            *_format_input_file(set_of_mols.emp_chgs_file),
            *_format_input_file(set_of_mols.ref_chgs_file)))
    bk.util.browser.view(output_file)


def _write_html_par(output_html_file: str,
                    correlation_graphs: bk.models.widgets.Tabs,
                    loc_min_graph: figure,
                    stats_par: namedtuple,
                    stats_val: namedtuple,
                    set_of_mols: SetOfMolecules,
                    par_info: list,
                    params_file: str):

    par_info.insert(2, f"Memory usage: {int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000)}MB")
    script, (correlation_graphs_html_source, loc_min_graph) = bk.embed.components((correlation_graphs, loc_min_graph))
    with open(output_html_file, "w") as html_file:
        html_file.write(open("modules/html_patterns/pattern_parameterization.txt").read().format(
            script, bk.resources.INLINE.render(),
            "</td><td>\n".join([str(item) for item in stats_par.all_ats]),
            "</td><td>\n".join([str(item) for item in stats_val.all_ats]),
            "</td><td>\n".join([str(item) for item in stats_par.mols]),
            "</td><td>\n".join([str(item) for item in stats_val.mols]),
            "".join([f"\n<tr style=\"background-color: {_background_color(at_type[1])};\"><td>" +
                     "</td><td>".join([str(item) for item in at_type]) + "</td></tr>"
                     for at_type in stats_par.ats_types]),
            "".join([f"\n<tr style=\"background-color: {_background_color(at_type[1])};\"><td>" +
                     "</td><td>".join([str(item) for item in at_type]) + "</td></tr>"
                     for at_type in stats_val.ats_types]),
            correlation_graphs_html_source,
            "".join(["\n<tr><td>" + "</td><td>".join([bond_type, str(count_par), str(stats_val.bonds[bond_type])]) +
                     "</td></tr>" for (bond_type, count_par) in stats_par.bonds.items()]),
            loc_min_graph,
            "</br>\n".join(["<b>" + line.replace(": ", "</b>: ") for line in par_info]),
            *_format_input_file(set_of_mols.sdf_file),
            *_format_input_file(set_of_mols.ref_chgs_file),
            *_format_input_file(params_file)))
    bk.util.browser.view(output_html_file)
