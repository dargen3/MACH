from modules.set_of_molecules import create_set_of_molecules
from sys import argv
from collections import defaultdict
from numpy import sqrt, mean
#
# def distance(a,b):
#     return sqrt(sum(abs(a-b)**2))
#
# # creation of bonds_stat_data
# set_of_molecules = create_set_of_molecules(argv[1], "hbob")
# bonds = defaultdict(list)
# for molecule in set_of_molecules.molecules:
#     coordinates = molecule.atomic_coordinates
#     for bond_reprezentation, (a1, a2, _) in zip(molecule.bonds_representation, molecule.bonds):
#         bonds[bond_reprezentation].append(distance(coordinates[a1], coordinates[a2]))
# bonds_stat_data = {}
# for bond, bond_data in bonds.items():
#     bonds_stat_data[bond] = [min(bond_data), mean(bond_data), max(bond_data), len(bond_data)]
#
# # print(bonds_stat_data)
# from pprint import pprint ; pprint(bonds_stat_data)
#
#
# bonds_stat_data = {'H~1/N-N~1/CHHH-1': [1.0094243179295743, 1.0099715807432406, 1.0106002827787697], 'C~1/CCHN-H~1/C-1': [1.0892246976609863, 1.090008932188107, 1.0907001343781677], 'C~1/CCHN-N~1/CHHH-1': [1.4546412996357565, 1.4885212219103054, 1.5070360536632244], 'C~1/CCHN-C~1/CHHO-1': [1.4922971685163358, 1.5237246306960839, 1.5425453860686449], 'C~1/CHHO-O~1/CH-1': [1.3878882263647623, 1.414493069063822, 1.432448484517496], 'C~1/CHHO-H~1/C-1': [1.089336239496711, 1.0899657530799352, 1.0905310060171958], 'H~1/O-O~1/CH-1': [0.9595094300034547, 0.9600322257582726, 0.960510438132573], 'C~1/CCHN-C~2/CNO-1': [1.4848499466843927, 1.5273837652811952, 1.5735762435046088], 'C~2/CNO-N~1/CCH-1': [1.2461388431808638, 1.330307544214994, 1.355549416903804], 'C~2/CNO-O~2/C-2': [1.1897999597632911, 1.2345510897760619, 1.2963145277250139], 'C~1/CCHN-N~1/CCH-1': [1.4189349579088706, 1.4600198543503857, 1.522207454814862], 'H~1/N-N~1/CCH-1': [0.9999520841682177, 1.0099539814338165, 1.0106962995202136], 'C~1/CCHH-C~1/CCHN-1': [1.4895616521987598, 1.5330116632574853, 1.5743581293655209], 'C~1/CCHH-H~1/C-1': [1.0892354094561454, 1.0900011554362798, 1.0907177958045884], 'C~1/CCHH-C~1/CCHH-1': [1.4189167479273856, 1.5125613587950426, 1.5816307220186647], 'C~1/CCHH-C~2/CNO-1': [1.47103068641498, 1.5182350925592973, 1.5940440131115485], 'C~2/CNO-N~1/CHH-1': [1.2671193407921482, 1.3267764424293975, 1.3598702887691838], 'H~1/N-N~1/CHH-1': [1.009316875302286, 1.0100146257804814, 1.0107211824897424], 'C~1/CCCH-C~1/CCHN-1': [1.4921370430349539, 1.5502647900245599, 1.5850934004415391], 'C~1/CCCH-C~1/CHHH-1': [1.4209838002574755, 1.5225416279608324, 1.5661259540716628], 'C~1/CCCH-H~1/C-1': [1.0894543762309268, 1.0900534713923726, 1.090594433080144], 'C~1/CHHH-H~1/C-1': [1.0893364446834923, 1.0900124618081712, 1.0907632993007397], 'C~1/CCHN-C~1/CHHH-1': [1.4793472067926063, 1.5218523717914354, 1.5708819704237318], 'C~1/CCHN-C~1/CCHO-1': [1.4833332584870869, 1.5409870709227296, 1.5695709375420746], 'C~1/CCHO-C~1/CHHH-1': [1.4887101770557059, 1.5233042057811468, 1.587904351751063], 'C~1/CCHO-O~1/CH-1': [1.4188158641768747, 1.4337769919164638, 1.458098286749063], 'C~1/CCHO-H~1/C-1': [1.0896031495886611, 1.0900154687393098, 1.0906404843911712], 'C~1/CCCH-C~1/CCHH-1': [1.4844035898768098, 1.5321420325506905, 1.5744708737268418], 'C~1/CCHH-C~1/CHHH-1': [1.4141976689706317, 1.5140340700799955, 1.570301610994305], 'C~2/CNO-N~1/CCC-1': [1.3130184693588207, 1.3377452370322382, 1.3595822713397492], 'C~1/CHHN-N~1/CCC-1': [1.437075318799795, 1.472877761805032, 1.4836150878491658], 'C~1/CHHN-H~1/C-1': [1.089284004845059, 1.0900126182883925, 1.090644777905629], 'C~1/CCHH-C~1/CHHN-1': [1.4027975193004738, 1.5098442150238394, 1.5720086118666068], 'C~1/CCHN-N~1/CCC-1': [1.454012391537507, 1.4676671987131353, 1.4909175261509928], 'C~1/CHHS-H~1/C-1': [1.0894375876573343, 1.0899980424743854, 1.0905462072597076], 'C~1/CCHH-C~1/CHHS-1': [1.4992315986563782, 1.52452286766162, 1.5501737629874264], 'C~1/CHHS-S~1/CC-1': [1.746230177701976, 1.8038739737800429, 1.831935351369041], 'C~1/HHHS-H~1/C-1': [1.089548385410969, 1.090042278485213, 1.0904539478301467], 'C~1/HHHS-S~1/CC-1': [1.6806511088819105, 1.772387134216591, 1.8272661979563813], 'C~1/CCHH-C~2/COO-1': [1.4851748469842379, 1.5187505666728032, 1.5457592976560097], 'C~2/COO-O~1/C-1': [1.2298386618116737, 1.2524914257952555, 1.2988766367177569], 'C~2/COO-O~2/C-2': [1.2220947816754513, 1.248133952218866, 1.279132978117901], 'C~1/CHHN-N~1/CCH-1': [1.432009516746744, 1.45927596991515, 1.482818500197142], 'C~1/CHHN-C~2/CNO-1': [1.48444761801323, 1.5153500132785065, 1.5536733631167763], 'C~1/CHHN-N~1/CHHH-1': [1.4513152245511207, 1.4944142641350984, 1.5316544018227733], 'C~1/CCHH-C~2/CCC-1': [1.4579532786197342, 1.5070846945155667, 1.5498431994229502], 'C~2/CCC-C~2/CCH-2': [1.369631515337249, 1.391188237317488, 1.4062693674025415], 'C~2/CCC-C~2/CCH-1': [1.36469449404651, 1.3941305127209451, 1.415544865721163], 'C~2/CCH-C~2/CCH-1': [1.3558370084121176, 1.3882121698527414, 1.4212484446460867], 'C~2/CCH-H~1/C-1': [1.0795098115059703, 1.0873354705384073, 1.0906067641736665], 'C~2/CCH-C~2/CCH-2': [1.3596464024394908, 1.3875014013777789, 1.42578718105807], 'C~2/CCO-O~1/CH-1': [1.3498858860030665, 1.3784681377605337, 1.4218799308020644], 'C~2/CCH-C~2/CCO-2': [1.3474787636686518, 1.3809228771236532, 1.3974760842452119], 'C~2/CCH-C~2/CCO-1': [1.3555688669899506, 1.3856094430436139, 1.4071881238040986], 'C~1/CCHN-C~2/COO-1': [1.5015477221185503, 1.5200165717579324, 1.5460473902760064], 'H~1/N-N~1/CCHH-1': [1.0096690982155903, 1.0099729849126102, 1.0102768716096302], 'C~1/CHHN-N~1/CCHH-1': [1.4729281915576193, 1.4729281915576193, 1.4729281915576193], 'C~1/CCHN-N~1/CCHH-1': [1.4696523888459827, 1.4696523888459827, 1.4696523888459827], 'C~2/NNN-N~1/CCH-1': [1.3197171503963951, 1.3306010645799342, 1.3520310880246953], 'C~2/NNN-N~1/CHH-1': [1.314982620766241, 1.3301493137840696, 1.3468398205749763], 'H~1/N-N~2/CHH-1': [1.0093991188143918, 1.0099626312730656, 1.0106309986231365], 'C~2/NNN-N~2/CHH-2': [1.2954555809778885, 1.3336456273397916, 1.3884487643526549], 'C~2/CCC-C~2/CCC-1': [1.4162576120061336, 1.4350465800027503, 1.45979625321683], 'C~2/CHN-N~1/CCH-1': [1.3474530212135867, 1.3690435694607141, 1.3786781925501541], 'C~2/CCC-C~2/CHN-2': [1.3400300920934298, 1.3543759421492452, 1.3671568662753697], 'C~2/CHN-H~1/C-1': [1.0796750715233234, 1.0866676795249504, 1.0904192994210704], 'C~2/CCN-N~1/CCH-1': [1.3676743539723062, 1.3728169935149601, 1.3800783336044313], 'C~2/CCC-C~2/CCN-2': [1.4081897602690447, 1.4093317621472499, 1.410473764025455], 'C~2/CCH-C~2/CCN-1': [1.3988593556742868, 1.4000510460608948, 1.4012427364475029], 'C~1/CCHH-C~2/CCN-1': [1.4680857208794214, 1.4957083117593375, 1.5145664352758152], 'C~2/CCN-N~2/CC-1': [1.3688143506012127, 1.378013565997946, 1.3925504165062308], 'C~2/CCN-C~2/CHN-2': [1.3470785257167357, 1.3544796055059996, 1.360274302417123], 'C~2/HNN-N~2/CC-2': [1.3123537334049233, 1.326417760540474, 1.3553714057641104], 'C~2/HNN-H~1/C-1': [1.089663060738546, 1.0899407731857689, 1.0902646800318831], 'C~2/HNN-N~1/CCH-1': [1.3149159422263461, 1.3220983915963958, 1.3315403409847326], 'C~1/CCHN-C~1/CHHS-1': [1.501163041711449, 1.530483374919156, 1.5605244378301915], 'C~1/CHHS-S~1/CS-1': [1.7833689918547875, 1.810673532671689, 1.8578914125075985], 'S~1/CS-S~1/CS-1': [1.9988716466892749, 2.046144044336771, 2.090526245654346], 'C~2/CCC-C~2/CCN-1': [1.3978275215455576, 1.4041768456937305, 1.4131959820904856], 'C~2/CCH-C~2/CCN-2': [1.3549326076302721, 1.3779776542932622, 1.3999557138999772]}
# set_of_molecules = create_set_of_molecules(argv[1], "hbob")
# for molecule in set_of_molecules.molecules:
#     print(molecule.name)
#     coordinates = molecule.atomic_coordinates
#     for bond_reprezentation, (a1, a2, _) in zip(molecule.bonds_representation, molecule.bonds):
#         try:
#
#             dist = distance(coordinates[a1], coordinates[a2])
#             mini = bonds_stat_data[bond_reprezentation][0]
#             maxi = bonds_stat_data[bond_reprezentation][2]
#
#             if dist + 0.01 < mini:
#                 print(bond_reprezentation, abs(dist-mini))
#             if dist - 0.01 > maxi:
#                 print(bond_reprezentation, abs(dist-maxi))
#         except KeyError:
#             continue
#     print()
#     print()
#     print()
#     print()


set_of_molecules = create_set_of_molecules(argv[1], "plain-ba")
for molecule in set_of_molecules.molecules:
    for bond_reprezentation, (a1, a2, _) in zip(molecule.bonds_representation, molecule.bonds):
        a1 +=1
        a2 +=1
        if bond_reprezentation == "C/CCH-C/CCN-2":
            print(1, bond_reprezentation, molecule.name, a1, a2)
        if bond_reprezentation == "C/CHN-N/CCH-1":
            print(2,bond_reprezentation, molecule.name, a1, a2)
        if bond_reprezentation == "C/CCC-C/CHN-2":
            print(3,bond_reprezentation, molecule.name, a1, a2)
        if bond_reprezentation == "H/N-N/CCHH-1":
            print(4,bond_reprezentation, molecule.name, a1, a2)
        if bond_reprezentation == "C/CCN-N/CC-1":
            print(5,bond_reprezentation, molecule.name, a1, a2)
        if bond_reprezentation == "C/CCC-C/CCN-2":
            print(6,bond_reprezentation, molecule.name, a1, a2)
        if bond_reprezentation == "C/CCHN-N/CCHH-1":
            print(7,bond_reprezentation, molecule.name, a1, a2)
        if bond_reprezentation == "C/CHHN-C/CNO-1":
            print(8,bond_reprezentation, molecule.name, a1, a2)
        if bond_reprezentation == "C/CCH-C/CCN-1":
            print(9,bond_reprezentation, molecule.name, a1, a2)
        if bond_reprezentation == "C/CHHN-N/CCHH-1":
            print(10,bond_reprezentation, molecule.name, a1, a2)