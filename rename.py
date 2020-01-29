def ReadSliceDataList(filename):
    datalist = []
    with open(filename) as f:
        for line in f:
            imagefile = line.lstrip(r"C:\Users\VMLAB\Desktop\kidney\kits19\data\case_").rstrip(r"\imaging.nii.gz")
            datalist.append(imagefile)
        print(datalist)

    return datalist

datalists = ReadSliceDataList(r"C:\Users\VMLAB\Desktop\kidney\empty.txt")
for datalist in datalists:
    fo = open(r"C:\Users\VMLAB\Desktop\kidney\newEmpty.txt", "a")
    fo.writelines(datalist+",")
    fo.close()