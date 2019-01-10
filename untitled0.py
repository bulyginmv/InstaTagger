# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 19:39:21 2018

@author: MyComputer
"""

from collections import Counter
a=input()
a=a.split()
c=Counter()
for word in a:
    c[word] += 1
a = sorted(c.items(), key=lambda item: (-item[1],item[0]))
print(a)
