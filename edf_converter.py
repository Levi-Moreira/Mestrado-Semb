from dataset.CHBDataSegmentProducer import CHBDataSegmentProducer


def process(patients):
    for p in patients:
        print(p)
        producer = CHBDataSegmentProducer(p)
        producer.data_file_creation()
        del producer


# patients = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09", "chb10",
#             "chb11", "chb12", "chb13", "chb14", "chb16", "chb17", "chb18", "chb19", "chb20",
#             "chb21", "chb22", "chb23", "chb24"]
# process(patients)
patients = ["chb15"]
process(patients)
