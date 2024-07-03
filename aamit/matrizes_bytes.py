
Primeiros 100 bytes em Hexedecimal:

  0: FF D8 FF E0 00 10 4A 46 49 46
 10: 00 01 01 00 00 01 00 01 00 00
 20: FF DB 00 43 00 08 06 06 07 06
 30: 05 08 07 07 07 09 09 08 0A 0C
 40: 14 0D 0C 0B 0B 0C 19 12 13 0F
 50: 14 1D 1A 1F 1E 1D 1A 1C 1C 20
 60: 24 2E 27 20 22 2C 23 1C 1C 28
 70: 37 29 2C 30 31 34 34 34 1F 27
 80: 39 3D 38 32 3C 2E 33 34 32 FF
 90: DB 00 43 01 09 09 09 0C 0B 0C

Matriz de bytes (10x10):
    
[[255 216 255 224   0  16  74  70  73  70]
 [  0   1   1   0   0   1   0   1   0   0]
 [255 219   0  67   0   8   6   6   7   6]
 [  5   8   7   7   7   9   9   8  10  12]
 [ 20  13  12  11  11  12  25  18  19  15]
 [ 20  29  26  31  30  29  26  28  28  32]
 [ 36  46  39  32  34  44  35  28  28  40]
 [ 55  41  44  48  49  52  52  52  31  39]
 [ 57  61  56  50  60  46  51  52  50 255]
 [219   0  67   1   9   9   9  12  11  12]]



 População inicial:
 
[[1 0 0 ... 1 0 0]
 [0 1 1 ... 1 0 1]
 [0 1 0 ... 1 0 0]
 ...
 [0 0 0 ... 1 1 0]
 [1 1 0 ... 0 1 0]
 [1 1 0 ... 1 1 0]]




    Feature1  Feature2
      
Feature1     1.00     0.85
Feature2     0.85     1.00


 
    Feature1  Feature2
Feature1     NaN     0.85
Feature2     NaN      NaN




Pai 1: [1 0 0 1 0 | 1 1 0 1 0]
Pai 2: [1 0 1 0 1 | 0 0 1 0 1]

Descendente 1: [1 0 0 1 0 | 0 0 1 0 1]
Descendente 2: [1 0 1 0 1 | 1 1 0 1 0]


Descendente 1 antes da mutação: [1 0 0 1 0 0 0 1 0 1]
Gene a ser mutado: 2 (índice 2)
Descendente 1 depois da mutação: [1 0 1 1 0 0 0 1 0 1]

Descendente 2 antes da mutação: [1 0 1 0 1 1 1 0 1 0]
Gene a ser mutado: 5 (índice 5)
Descendente 2 depois da mutação: [1 0 1 0 1 0 1 0 1 0]
