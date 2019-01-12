# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 18:51:47 2018

@author: MyComputer
"""

import numpy as np
file=open('thesaurus.txt', 'r',encoding='utf-8-sig')
file=file.read()
lines=file.split('\n')
thesaurus=[]
for line in lines:
    thesaurus.append(line.split(', '))
aug=np.array(thesaurus, dtype=object)
def tag_aug(tag,conf):
    for i in range(len(lines)):
        if tag==aug[i][0]:
            out=(f'tag: {aug[i][0]} confidence: {conf}, tag: {aug[i][1]} confidence: {conf}, tag: {aug[i][2]} confidence: {conf}')
            return out
    out=(f'tag: {tag} confidence: {conf}')
    return out


