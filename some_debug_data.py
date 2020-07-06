import os

from util import data_io

if __name__ == '__main__':
    base_path = os.environ["HOME"] + "/hpc/data/parallel_text_corpora/wmt_en_ro"
    files = [f for f in os.listdir(base_path) if f.endswith(".source") or f.endswith(".target")]
    some_data = "some_data"
    os.makedirs("%s" % some_data, exist_ok=True)
    for f in files:
        data_io.write_lines("%s/%s" % (some_data, f), data_io.read_lines(base_path + "/%s" % f, limit=1000))