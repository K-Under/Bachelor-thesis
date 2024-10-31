import h5py
import sys
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Filepath')
    parser.add_argument('--file_path', type=str, default=f"/data/kunter/rixs/data/separate_datasets_128_128_onlySignal_0p95_sum16/seed_0/training_data.hdf5",help='File where dataset is read from')
    return parser.parse_args()

def opening_dataset(file_path):
    print(f"Showing insides of: {file_path}")
    with h5py.File(file_path, 'r') as hdf_file:

        print("Attributes")
        for key in hdf_file.attrs.keys():
            print("Key:", key, hdf_file.attrs[key])
            print("Type:", type(hdf_file.attrs[key]))
            try:
                print(list(hdf_file.attrs[key]))
                print("\n")
            except:
                print("listing failed\n")
                pass

        print("keys")
        for key in hdf_file.keys():
            print("Key:", key)
            print("Type:", type(hdf_file[key]))
            print("Raw:", hdf_file[key])
            try:
                print(list(hdf_file[key]["data"][0][0:50]))
                print("\n")
            except:
                print("listing failed\n")
                pass
        print(hdf_file["lc"]["data"][0].shape)
        print(hdf_file["lc"]["data"][:].shape)
        print(hdf_file["lc"]["data"][:,:,0].shape)
        
        print("File opened\n\n\n")


def open_meta(file_path):
    print("Trying to open metadata...")
    with h5py.File(file_path, 'r') as hdf_file:
        meta_group = hdf_file.get('/meta')

        if meta_group:
            # List all members of the 'meta' group
            print("Contents of '/meta' group:\n")
            for key in meta_group.keys():
                print(f"Key: {key}")
                print(f"Type: {type(meta_group[key])}")
                print(f"Raw: {meta_group[key]}")
                
                try:
                    print(list(hdf_file[key]["data"]))
                    print("\n")
                except:
                    print("listing failed\n")
                    pass

                #check for sub keys
                sub_item = meta_group[key]
                print("checking subgroups...\n")
                if isinstance(sub_item, h5py.Group):
                    print(f"Sub-group '{key}' contents:")
                    for sub_attr in sub_item.attrs:
                        print(f"Attribute: {sub_attr}, Value: {sub_item.attrs[sub_attr]}")
                elif isinstance(sub_item, h5py.Dataset):
                    print(f"Dataset '{key}':")
                    print(sub_item[()])
                    #indices = [index for index, value in enumerate(sub_item) if value == 330921]
                    #print(f"\n finding index for runNB: 330921: {indices}")
                print("\n\n")
        else:
            print("no metadata found")


if __name__ == "__main__":
    args = parse_args()
    opening_dataset(args.file_path)
    open_meta(args.file_path)
