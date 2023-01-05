from imutils import paths
import argparse


def find_calib_file(serial_number, directory):
    for json_path in paths.list_files(directory):
        json_path_split = json_path.split('\\')
        json_serial = json_path_split[-1].split('_')[0]
        
        if json_path_split[-1].split('_')[0] == serial_number and json_path_split[-1].split('_')[-1] == "centred.json":
            return json_path
        
    return None

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image-directory", required=True,
    help="path to directory of images")
ap.add_argument("-j", "--json-file-directory", required=True,
    help="path to directory of json files")
# ap.add_argument("-s", "--serial", required=True,
#     help="serial number of camera")
args = ap.parse_args()

image_path_left = None
image_path_right = None

for image_path in paths.list_images(args.image_directory):
    image_path_split = image_path.split('\\')
    image_serial = image_path_split[-1].split('_')[0]
    
    if "Left" in image_path:
        image_path_left = image_path
        image_path_right = image_path.split('Left')[0] + 'Right.png'
    
    # print(image_path_split[-1].split('_')[0])
    if image_path_split[-1].split('_')[0] == args.serial:
        if  "Left" in image_path:
            image_path_left = image_path
        elif "Right" in image_path:
            image_path_right = image_path
        
if image_path_left is None or image_path_right is None:
    # throw error and exit
    print("No images found for serial number: " + args.serial)
    exit()

        
command_string = f"python.exe .\\mains\\fixed_rigel_calibration.py -c 0 -f opencv -i '{json_path_sn}' -L '{image_path_left}' -R '{image_path_right}' -v -d 0 -r 0"

with open(f'{output_file_base}_uncentred.json', 'w') as fp:
