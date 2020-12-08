from interface.constants import WINDOW_SIZE, POSITIVE_SHIFT_WINDOW_SAMPLE_SIZE


def get_negative_chunks_from_data(data):
    chunks = []
    window_start = 0
    while (window_start + WINDOW_SIZE) < data.shape[1]:
        chunks.append(data[:, window_start:window_start + WINDOW_SIZE])
        window_start += WINDOW_SIZE
    return chunks


def get_positive_chunks_from_data(data):
    chunks = []
    window_start = 0
    while (window_start + WINDOW_SIZE) < data.shape[1]:
        chunks.append(data[:, window_start:window_start + WINDOW_SIZE])
        window_start += WINDOW_SIZE
    return chunks
