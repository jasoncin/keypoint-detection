import os
import json
import zipfile


class JsonHandler(object):
    """To read, dumpy json format. """
    def default(self, o):
        if isinstance(o, np.int64): return int(o)  
        raise TypeError

    def read_json_file(self, filename):
        with open(filename, encoding='utf-8') as f:
            return json.load(f)

    def dump_to_file(self, data, filename):
        with open(filename, "w", encoding="utf-8") as fp:
            # json.dump(data, fp, default=self.default)
            json.dump(data, fp, indent=2, ensure_ascii=False)

def write_log(file_path, content, mode='a'):
    with open(file_path, mode) as opt_file:
        opt_file.write(content + "\n")

def unzip(zip_file, to_folder):
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(to_folder)
    zip_ref.close()
