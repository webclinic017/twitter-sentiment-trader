import csv
from io import StringIO


def flatten_json(y):

    out = {}

    def flatten(x, name=""):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], ".".join([name, a]))
        else:
            out[name] = x

    flatten(y)
    return out


def json2csv(data, headers):
    with StringIO() as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
        return csvfile.getvalue()
