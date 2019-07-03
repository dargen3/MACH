from sys import exit
from termcolor import colored


def control_order_of_molecules(names1, names2, file1, file2):
    print("Control of order of molecules... ")
    if names1 == names2:
        print(colored("ok\n", "green"))
        return True
    names1 = set(names1)
    names2 = set(names2)
    if names1 == names2:
        exit(colored("Files {} and {} contain same molecules, but in different order!\n".format(file1, file2), "red"))
    else:
        intersection = names1.intersection(names2)
        difference = names1.symmetric_difference(names2)
        print(colored("""Files {} and {} contain different set of molecules!
Number of common molecules:    {}
Number of different molecules: {}""".format(file1, file2, len(intersection), len(difference)), "red"))
        if input("Do you want to print difference molecules names? yes/no: ") == "yes":
            for name in difference:
                print(name)
        exit("")
