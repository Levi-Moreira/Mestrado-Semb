CHANNELS_NAMES = ['FP1-F7',
                  'F7-T7',
                  'T7-P7',
                  'P7-O1',
                  'FP1-F3',
                  'F3-C3',
                  'C3-P3',
                  'P3-O1',
                  'FP2-F4',
                  'F4-C4',
                  'C4-P4',
                  'P4-O2',
                  'FP2-F8',
                  'F8-T8',
                  'T8-P8',
                  'P8-O2',
                  'FZ-CZ',
                  'CZ-PZ']


def le_channel_transformer(name):
    return "EEG {}-LE".format(name)


def ref_channel_transformer(name):
    return "EEG {}-REF".format(name)


CUT_OFF_LOWER = 4
CUT_OFF_HIGHER = 40
