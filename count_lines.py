import os

basefolder = "/media/levi/ELEMENTS/Mestrado/ExperimentsResults  01.10/cnn_2_s_overlap_1"

patients = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09", "chb10",
            "chb11", "chb12", "chb13", "chb14", "chb15", "chb16", "chb17", "chb18", "chb19", "chb20",
            "chb21", "chb22", "chb23", "chb24"]

for p in patients:
    path = os.path.join(basefolder, p, "val.txt")
    print(p)
    with open(path, 'r') as f:
        count = sum(1 for _ in f)
        print(count)
        print()
