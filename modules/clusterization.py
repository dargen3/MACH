from .set_of_molecules import create_set_of_molecules
from .control_order_of_molecules import control_order_of_molecules
from .control_existing import control_existing_files
from .input_output import add_charges_to_set_of_molecules
from scipy.stats import kde
from numpy import array, linspace
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, BoxSelectTool, Cross, Select
from bokeh.plotting import figure
from bokeh.models.widgets import PreText
from bokeh.server.server import Server
from operator import itemgetter
from collections import Counter
import functools
import operator


def bonded_atoms_statistics(indices, atoms_data):
    cluster = []
    for x in indices:
        cluster.append(atoms_data[x])
    data_text = ""
    num_of_bonded_atoms = 0
    total_charge = 0
    num_of_atoms = len(cluster)
    bonded_atoms = Counter()
    for atom in cluster:
        num_of_bonded_atoms += len(atom) -1
        total_charge += atom[0]
        bonded_atoms.update(atom[1:])
    average_charge = total_charge / num_of_atoms
    average_num_of_bonded_atoms = num_of_bonded_atoms / num_of_atoms
    data_text += """
    Number of atoms: {}
    Average charge: {}
    Average number of bonded atoms: {}
    """.format(num_of_atoms, round(average_charge, 4), average_num_of_bonded_atoms)
    for index, atom in enumerate(bonded_atoms.most_common()):
        if atom[1]/num_of_atoms < 0.01:
            continue
        if index == 0:
            data_text += "Average bonded atoms composition:  {} : {}\n".format(atom[0], round(atom[1]/num_of_atoms, 4))
        else:
            data_text += "                                       {} : {}\n".format(atom[0], round(atom[1]/num_of_atoms, 4))
    data_text += "\n\n"
    return data_text


def clusterize(charges_file, sdf_file, atomic_types_pattern):
    control_existing_files([(sdf_file, True, "file"),
                            (charges_file, True, "file")], None)
    set_of_molecules = create_set_of_molecules(sdf_file, atomic_types_pattern)
    add_charges_to_set_of_molecules(set_of_molecules, charges_file)
    atomic_types = sorted(list(set(functools.reduce(operator.iconcat, [molecule.atoms_representation for molecule in set_of_molecules.molecules], []))))


    def modify_doc(doc):
        global charges
        source = ColumnDataSource(data={"xgrid": [], "density": []})
        source_charges= ColumnDataSource(data={"charges": [], "zeros": []})
        plot = figure(title="No atomic type selected.",
                      x_axis_label="Charges",
                      y_axis_label="Kernel density estimation",
                      plot_width=900, plot_height=900)
        plot.xaxis.axis_label_text_font_size = "20px"
        plot.yaxis.axis_label_text_font_size = "20px"
        plot.add_tools(BoxSelectTool(dimensions="width"))
        plot.title.align = "center"
        plot.title.text_font_size = "25px"
        plot.line('xgrid', "density", source=source)
        renderer = plot.cross("charges", "zeros", source=source_charges, size=10)
        renderer.selection_glyph = Cross(line_color="red")
        pre = PreText(text="No atoms selected. Select atoms by Box Select tool.", width=900, height=900, style={'font-size': '150%'})

        def update_density_kernel_estimation(attr, old, new):
            text = plot.title.text
            plot.title.text = "Calculation ..."
            density = kde.gaussian_kde(charges, bw_method=new)
            data = {"xgrid": xgrid, "density": density(xgrid)}
            source.data = ColumnDataSource(data=data).data
            plot.title.text = text

        def print_cluster(attr, old, new):
            if not(len(new)):
                pre.text = "No atoms selected. Select atoms by Box Select tool."
            else:
                pre.text = bonded_atoms_statistics(new, all_data)

        def update_atomic_type(attr, old, new):
            if new == "":
                plot.title.text = "No atomic type selected."
                return
            plot.title.text = "Calculation..."
            global all_data
            all_data = []
            atomic_type = new
            source_charges.selected.indices = []
            for molecule in set_of_molecules.molecules:
                molecule_atoms_reprezentation = molecule.atoms_representation
                molecule_bonds_reprezentation = [bond[:2] for bond in molecule.bonds]
                for index, (atom, charge) in enumerate(zip(molecule_atoms_reprezentation, molecule.ref_charges)):
                    if atomic_type == atom:
                        bonded_atoms = [charge]
                        for i1, i2 in molecule_bonds_reprezentation:
                            if index in (i1, i2):
                                if index == i1:
                                    bonded_atoms.append(molecule_atoms_reprezentation[i2])
                                else:
                                    bonded_atoms.append(molecule_atoms_reprezentation[i1])
                        all_data.append(bonded_atoms)
            all_data.sort(key=itemgetter(0))
            global charges
            charges = array([x[0] for x in all_data])
            density = kde.gaussian_kde(charges, bw_method=0.1)
            global xgrid
            xgrid = linspace(charges.min(), charges.max(), 10000)
            source.data = ColumnDataSource(data={"xgrid": xgrid, "density": density(xgrid)}).data
            source_charges.data = ColumnDataSource(data={"charges": charges, "zeros": [0 for _ in range(len(charges))]}).data
            plot.title.text = "Kernel density estimation for atom {}".format(atomic_type)
            slider.value = 0.1


        slider = Slider(start=0.01, end=0.5, value=0.1, step=0.01, title="Bandwidth")
        selector = Select(title="Atomic type", value="", options=atomic_types + [""])
        slider.on_change('value', update_density_kernel_estimation)
        selector.on_change("value", update_atomic_type)
        renderer.data_source.selected.on_change('indices', print_cluster)
        doc.add_root(row(column(selector, slider, plot), pre))
        doc.title = "Cluster tool"
    server = Server({'/': modify_doc}, num_procs=1)
    server.start()
    print('Opening Bokeh application on http://localhost:5006/')
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
