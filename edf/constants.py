CHANNELS_NAMES = ['FP1-F7', # 1
                  'F7-T7', # 2 x
                  'T7-P7', # 3 x
                  'P7-O1', # 4
                  'FP1-F3', # 5
                  'F3-C3', # 6 x
                  'C3-P3', # 7 x
                  'P3-O1',# 8
                  'FP2-F4', # 9
                  'F4-C4', # 10 x
                  'C4-P4', # 11
                  'P4-O2', # 12
                  'FP2-F8', # 13
                      'F8-T8', # 14 x
                  'T8-P8', # 15
                  'P8-O2', # 16
                  'FZ-CZ', # 17
                  'CZ-PZ'] # 18


def le_channel_transformer(name):
    return "EEG {}-LE".format(name)


def ref_channel_transformer(name):
    return "EEG {}-REF".format(name)


CUT_OFF_LOWER = 4
CUT_OFF_HIGHER = 40
