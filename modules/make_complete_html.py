from glob import glob
from os import system, path
from .set_of_molecules import SetOfMolecules
from termcolor import colored


def make_complete_html():  # only for my usage
    print("Creating html...")
    with open("data/index.html", "w") as html_file:
        html_file.write("<h1>Calculator charges</h1>\n")
        html_file.write("<h2>Source code: https://github.com/dargen3/MACH</h2>\n")
        html_file.write("<a href = \"data\">All data</a>\n<br />\n<br />\n<br />\n<br />\n")
        html_files = glob("data/*/*/*.html")
        methods_data_html = {}
        sets_of_molecules = []
        for file in html_files:
            splitted_path = file.split("/")
            method = splitted_path[1]
            sets_of_molecules.append(splitted_path[-2])
            if method not in methods_data_html:
                methods_data_html[method] = []
            methods_data_html[method].append(splitted_path)
        sdf_files = glob("data/*/*/*.sdf")
        sdf_files_check = []
        for sdf_file_path in sdf_files:
            sdf_file = sdf_file_path.split("/")[-1]
            if sdf_file not in sdf_files_check and not path.isfile("data/sets_of_molecules_info/{}_info.txt".format(sdf_file[:-4])):
                SetOfMolecules(sdf_file_path).info("atom_high_bond", file="data/sets_of_molecules_info/{}_info.txt".format(sdf_file[:-4]))
                sdf_files_check.append(sdf_file)
        sets_int = []
        sets_str = []
        for setm in set(sets_of_molecules):
            try:
                sets_int.append(int(setm))
            except ValueError:
                sets_str.append(setm)
        sets_of_molecules = [str(sets) for sets in sorted([seti for seti in sets_int])] + sorted(sets_str)
        html_file.write("<table border=1>\n")
        html_file.write("<tbody>\n")
        html_file.write("<th>Method</th>\n")
        for setm in sets_of_molecules:
            html_file.write("<th>{}</th>\n".format(setm))
        html_file.write("</tr>\n")
        names = {"1956": "DTP_small", "4475": "DTP_large", "proteins": "", "8144": "CCD_gen_CHNO", "17769": "CCD_gen_all"}
        html_file.write("<th>Name_of_set</th>\n")
        for setm in sets_of_molecules:
            try:
                html_file.write("<th>{}</th>\n".format(names[setm]))
            except KeyError:
                html_file.write("<th></th>\n")
        html_file.write("</tr>\n")
        html_file.write("<tr>\n")
        html_file.write("<td>Set of molecules info</td>")
        for set_info in sets_of_molecules:
            html_file.write("<td><a href = \"data/sets_of_molecules_info/{}_info.txt\">{}_info</a>\n<br />\n</td>\n".format(set_info , set_info))
        html_file.write("</tr>\n")
        for method in sorted(methods_data_html):
            html_file.write("<tr>\n")
            html_file.write("<td>{}</td>\n".format(method))
            for setm in sets_of_molecules:
                is_there = False
                for setm_of in methods_data_html[method]:
                    if setm_of[2] == setm:
                        is_there = True
                        break
                if is_there == True:
                    apath = "data/{}/{}/{}_{}.html".format(method, setm, setm, method)
                    name = "{} {}".format(setm, method)
                    html_file.write("<td><a href = \"{}\">{}</a>\n<br />\n</td>\n".format(apath, name))
                else:
                    html_file.write("<td>no results</td>\n".format(setm))
            html_file.write("</tr>\n")
        html_file.write("<tbody>\n")
        html_file.write("</table>\n")
        html_file.write("<br /><br /><br /><br /><h3>Contact: dargen3@centrum.cz</h3>\n")
    print("Copying of data...\n\n\n")
    system("rsync -v -r data dargen3@lcc.ncbr.muni.cz:/home/dargen3/www/")
    print(colored("Data was copied sucessfully.\n\n\n", "green"))
    print("Setting permissions...")
    system("ssh dargen3@lcc.ncbr.muni.cz \" mv www/data/index.html www/index.html ; chmod -R 705 * \"")
    print(colored("Setting of permissions was sucessfull.\n\n\n", "green"))

