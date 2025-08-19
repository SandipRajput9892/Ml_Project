import os
import dill

def save_object(file_path, obj):
    """
    Save Python object using dill
    """
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
        dill.dump(obj, file_obj)


def load_object(file_path):
    """
    Load Python object using dill
    """
    with open(file_path, "rb") as file_obj:
        return dill.load(file_obj)
