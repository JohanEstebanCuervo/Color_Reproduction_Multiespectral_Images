# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 23:42:16 2022

@author: cuerv
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import subprocess
import platform

sistema= platform.system()


x = str(subprocess.check_output('python -m serial.tools.list_ports',shell=True),'UTF-8')

lista = []

inicio=0
i=x.find(' ',inicio,len(x))
while i>0:
    lista.append(x[inicio:i])
    inicio=x.find('\n',inicio,len(x))+1
    i=x.find(' ',inicio,len(x))

print(lista)