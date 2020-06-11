from dataset.DataProducer import DataProducer

patients = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09", "chb10",
            "chb11", "chb12", "chb13", "chb14", "chb15", "chb16", "chb17", "chb18", "chb19", "chb20",
            "chb21", "chb22", "chb23", "chb24"]
# patients = ["chb01"]
for p in patients:
    print(p)
    producer = DataProducer()
    producer.data_file_creation(18, p)
    del (producer)
