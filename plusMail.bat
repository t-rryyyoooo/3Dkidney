@echo off  
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0


rebuild.py C:\Users\VMLAB\Desktop\kidney\selectedMargeTraining.txt -t C:\Users\VMLAB\Desktop\kidney\selectedMargeValidation.txt -l 1e-5

mail.py C:\Users\VMLAB\Desktop\kidney\history.txt

rebuild.py C:\Users\VMLAB\Desktop\kidney\selectedMargeTraining.txt -t C:\Users\VMLAB\Desktop\kidney\selectedMargeValidation.txt -l 1e-3

mail.py C:\Users\VMLAB\Desktop\kidney\history.txt