import os
import pandas as pd

def convert_dat_to_csv(dat_filepath, output_dir):
    """
    Reads a space-delimited .dat file from RACIPE and saves it as a .csv file.

    Args:
        dat_filepath (str): The full path to the input .dat file.
        output_dir (str): The directory where the new .csv file will be saved.
    """
    # Check if the input file exists
    if not os.path.exists(dat_filepath):
        print(f"Error: Input file not found at {dat_filepath}")
        return

    # Ensure the output directory exists, create it if it doesn't
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Use pandas to read the file. delim_whitespace=True handles any amount
        # of space or tabs between columns, which is robust for .dat files.
        df = pd.read_csv(dat_filepath, delim_whitespace=True)

        # Construct the output filename
        # It will be the same as the input file, but with a .csv extension
        base_filename = os.path.basename(dat_filepath)
        csv_filename = os.path.splitext(base_filename)[0] + ".csv"
        csv_filepath = os.path.join(output_dir, csv_filename)

        # Save the DataFrame to a .csv file.
        # index=False prevents pandas from writing an extra column for the row numbers.
        df.to_csv(csv_filepath, index=False)
        print(f"Successfully converted {dat_filepath} to {csv_filepath}")

    except Exception as e:
        print(f"Could not process file {dat_filepath}. Error: {e}")

def batch_process_racipe_directory(source_dir, destination_dir):
    """
    Finds all .dat files in a source directory and converts them to .csv
    files in a destination directory.

    Args:
        source_dir (str): The directory containing the RACIPE .dat output files.
        destination_dir (str): The directory where the .csv files will be saved.
    """
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found at {source_dir}")
        return
    
    print(f"Searching for .dat files in '{source_dir}'...")
    
    found_files = 0
    # Loop through all files in the source directory
    for filename in os.listdir(source_dir):
        # Check if the file ends with .dat
        if filename.endswith(".dat") or filename.endswith(".prs"):
            found_files += 1
            dat_filepath = os.path.join(source_dir, filename)
            convert_dat_to_csv(dat_filepath, destination_dir)
            
    if found_files == 0:
        print("No .dat files found in the source directory.")
    else:
        print(f"\nBatch processing complete. Converted {found_files} files.")

if __name__ == "__main__":
    batch_process_racipe_directory("/Users/govindrnair/CSB_EnergyLandscape","/Users/govindrnair/CSB_EnergyLandscape")



