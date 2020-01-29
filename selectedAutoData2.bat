@echo off  

SETLOCAL ENABLEDELAYEDEXPANSION



cd %~dp0



::data path
set 
data=C:\Users\VMLAB\Desktop\kidney\kits19\data\case_00

set toolbin=C:\study\bin\imageproc\bin\imageproc.exe


::file name
set 
ct=\imaging.nii.gz
set 
cthist=\ct_hist25.mha
set 
label=\segmentation.nii.gz
set 
label_mask=\mask.mha




::patch size


set psize=36x36x28


set numArr=007,010,013,015,021,025,030,031,038,041,045,050,051,061,064,087,088,090,094,098,109,110,112,113,115,116,117,122,125,129,130,132,134,138,148,150,152,153,160,162,166,174,193,199


for %%i in (%numArr%) do (

    set ctpath=%data%%%i%ct%

    set cthistpath=%data%%%i%cthist%

    set labpath=%data%%%i%label%

    set mask_path=%data%%%i%label_mask%



    getCTSize1.py !ctpath! empty.txt

)
