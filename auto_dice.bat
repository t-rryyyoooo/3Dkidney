@echo off  
SETLOCAL ENABLEDELAYEDEXPANSION
::data path
set data=E:\kits19_data_processed\case_00
set logdir=E:\Script\log_cube
md %logdir%\result

::file name
set ct=\ct.mha
set cthist25=\ct_hist25.mha
set label=\mask.mha

set weight1=%logdir%\model\model_17_0.11.hdf5


::patch size
set psize=36x36x28



set numArr=000,001,002,003,004,041,042,043,044,100,101,102,103,104,145,146,147,148,149,194,195,196,197,198

for %%i in (%numArr%) do (
    set ctpath=%data%%%i%ct%
    set cthistpath=%data%%%i%cthist25%
    set labpath=%data%%%i%label%
    set result1_path=%logdir%\result\result%%i.mha
    
    set mask_path=%orgdata%%%i%label_mask%
    set pinfo=.\info\ID%%iinfo.txt

    segmentation3DUnet.py !cthistpath! E:\Script\444428.yml %weight1% !result1_path!
)


::set numArr=01,02,03,04,05,06,07,08,09,10,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
