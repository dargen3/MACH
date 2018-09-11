




pro mě:

vztah časové náročnosti na počet atomů a počet atomových typů

největší molekula, nejmenší, průměr.

čtení naboju z sdf není hotové ani pro v3 ani pro v2

odkaz na sdf - dat na readme!

jednotlive metody - odkazy - dat na readme!

neni doplnena jina vzdalenost než angstremy

QEq 500 - oddělat cor 

pořešit jestli je v metodách nutné num_of_atom


pořešit:

grafy jak postupovala optimalizace?

rychlejší delani obrazku

final_html - až potom má cenu všechno přepočítat 

společný neduh všech metod - cesta?

do statistik i procenta a statistiky vazeb?


oddelat poslední parametrizaci

poděkování metacentrum

scratch, užívám? je potřeba nastavovat velký? Co to je? Meta

chybějící nuly ve statistice

jaké metody

v molecules dodelat at to hlasí který atom v parametrech chybí! je to divné!

v set_of_molecules_info : vypadá škaredě hlášení! - taky write parameters """


statistic_data.append([atomic_type] + statistic(charges)
                                  + [corrcoef(list(zip(*charges)))[0, 1] ** 2, len(charges)])

warnings numpy a browser

funkce() backround color

vypisovat statistic když se dělá html?

symetrická matice - solve?
jak profilovat numba function



float 64 a float 32
./mach.py --mode parameterization_find_args --path data/EEM/500/ --optimization_method minimization --data_dir asldfjalksdf -f 



výsledky: 8000
          old      new
EEM       1.95     0.33   5.9
SFKEEM    2.48     0.51   4.9   
QEq       2.32     0.71   3.3
GM        5.22     0.08   65

RAM       480MB    260MB     1.9     



překvapivě rozepsane je nekdy rychlejsi nez numpy!!!
In [2]: from numpy import array, float32

In [3]: from numba import jit

In [4]: a = array([[x for x in range(1000)] for x in range(1000)], dtype=float32
   ...: )

In [5]: b = array([x for x in range(1000000)], dtype=float32)

In [6]: @jit(nopython=True)
   ...: def f(a):
   ...:     return a/0.5
   ...: 

In [7]: @jit(nopython=True)
   ...: def ff(a):
   ...:     for x in range(1000):
   ...:         for y in range(1000):
   ...:             a[x][y] = a[x][y]/0.5
   ...:             

In [8]: %timeit f(a)
1.87 ms ± 40.3 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [9]: %timeit ff(a)
518 µs ± 7.53 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

In [10]: %timeit f(a)
1.84 ms ± 101 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [11]: @jit(nopython=True)
    ...: def fff(a):
    ...:     for x in range(1000000):
    ...:         a[x] = a[x]/0.5
    ...:             

In [12]: %timeit f(b)
1.84 ms ± 30.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

In [13]: %timeit fff(b)
273 µs ± 12.2 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)




In [1]: from numpy import array, float32

In [2]: from numba import jit

In [3]: a = array([[x for x in range(1000)] for x in range(1000)], dtype=float32
   ...: )

In [4]: @jit(nopython=True)
   ...: def f(a):
   ...:     a[:, 600] = 1.0
   ...:     
   ...: 

In [5]: @jit(nopython=True)
   ...: def ff(a):
   ...:     for x in range(600):
   ...:         a[0][x] = 1.0
   ...:         
   ...:     
   ...: 

In [6]: %timeit f(a)
2.35 µs ± 12 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

In [7]: %timeit ff(a)
292 ns ± 1.21 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)




https://numba.pydata.org/numba-doc/dev/user/performance-tips.html



jak nastavit numpy aby počítalo s nějakou přesností

pořešit bounds!

dopsat diferential evolution
