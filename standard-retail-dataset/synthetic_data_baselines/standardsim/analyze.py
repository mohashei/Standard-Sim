"""
Contains functions to move scene files from data directory to alternate directory,
to remove scene names from image set files, and to print dataset statitstics.
"""
import os
import imageio
import numpy as np
from create_dataset_json import find_files

def move_scene_errors(data_root, error_root, errors_file):
    """ Moves files from data_root to error_root dir if scene is in errors file
    @param data_root (str): Path to data root directory; e.g. '/renders_multicam_diff_1'
    @param error_root (str): Path to directory to move problematic files to, e.g. '/scene_error_renders_multicam_diff_1'
    @param errors_file (str): Path to text file containing one scene per line, e.g. 'scene_errors.txt'
    """
    with open(errors_file, 'r') as errorsFile:
        data = errorsFile.readlines()

    data = [d[:-1] for d in data]

    all_files = os.listdir(data_root)

    for n in data:
        for f in all_files:
            if n in f:
                mv_cmd = "mv " + os.path.join(data_root, f) + " " + error_root
                os.system(mv_cmd)


def remove_scenes_from_imageset(errors_file, imagesets_dir):
    """ Removes problematic scenes from imageset files.
    @param errors_file (str): Path to text file containing one scene per line, e.g. 'scene_errors.txt' 
    @param imagesets_dir (str): Path to directory containing the image set files, .e.g '/synthetic_data_baselines/imagesets'
    """
    with open(errors_file, 'r') as errorsFile:
        errors = errorsFile.readlines()

    errors = [d[:-1] for d in errors]

    # Check each image split for scene error members
    splits = ['train', 'val', 'test']
    for split in splits:
        print(split)
        with open(os.path.join(imagesets_dir, '{}.txt'.format(split)), 'r') as nameFile:
            scenes = nameFile.readlines()

        scenes = [s[:-1] for s in scenes]
        print("num scenes ", len(scenes))
        new_scenes = []
        for s in scenes:
            if s not in errors:
                new_scenes.append(s)
            else:
                print("scene ", s, " found in split ", split)
            

        with open('/app/synthetic_data_baselines/imagesets/{}_new.txt'.format(split), 'w') as newFile:
            for elt in new_scenes:
                newFile.write(elt+'\n')

def analyze_scenes(dataroot, imagesets_dir):
    """ Print breakdown of imageset files based on number stores, cams per scene, etc.
    @param dataroot (str): Path to directory containing all data
    @param imagesets_dir (str): Path to directory containing train.txt, val.txt and test.txt files
    """
    dataset_samples = find_files(dataroot)

    print("Number dataset samples ", len(dataset_samples))

    with open(os.path.join(imagesets_dir, 'train.txt'), 'r') as trainFile:
        train_scenes = trainFile.readlines()
    train_scenes = [t[:-1] for t in train_scenes]
    with open(os.path.join(imagesets_dir, 'val.txt'), 'r') as valFile:
        val_scenes = valFile.readlines()
    val_scenes = [v[:-1] for v in val_scenes]
    with open(os.path.join(imagesets_dir, 'test.txt'), 'r') as testFile:
        test_scenes = testFile.readlines()
    test_scenes = [t[:-1] for t in test_scenes]
    all_scenes = train_scenes + val_scenes + test_scenes

    # Get unique stores
    stores = set([s.split('_')[0] for s in all_scenes])
    print("Stores ", stores)
    # Get number of unique scenes per store
    stores_to_scenes = dict()
    for store in stores:
        stores_to_scenes[store] = []
    for i in range(len(dataset_samples)):
        store, scene, camera = dataset_samples[i]['label'].split('/')[-1].split('_')[:3]
        stores_to_scenes[store].append(scene)
    print("Number unique scenes per store:")
    for store in stores:
        print(store, len(set(stores_to_scenes[store])))
    # Get number of samples per store
    print("Number of samples per store:")
    for k,v in stores_to_scenes.items():
        print(k, len(v))
    # Get number of cameras per scene
    store_scenes_to_cams = dict()
    for k,v in stores_to_scenes.items():
        for i in range(len(v)):
            store_scenes_to_cams[k+'_'+v[i]] = 0
    for i in range(len(dataset_samples)):
        store, scene, camera = dataset_samples[i]['label'].split('/')[-1].split('_')[:3]
        store_scenes_to_cams[store+'_'+scene] += 1
    print("mean number cams ", sum(store_scenes_to_cams.values()) / len(store_scenes_to_cams))
    print("Len scenes train ", len(train_scenes), " len val ", len(val_scenes), " len test ", len(test_scenes))

    print("Train")
    # Get train samples
    train_samples = [d for d in dataset_samples if '_'.join(d['label'].split('/')[-1].split('_')[:3])[:-10] in train_scenes]
    # Get unique stores
    stores = set([s.split('_')[0] for s in train_scenes])
    print("Stores ", stores)
    # Get number of unique scenes per store
    stores_to_scenes = dict()
    for store in stores:
        stores_to_scenes[store] = []
    for i in range(len(train_samples)):
        store, scene, camera = train_samples[i]['label'].split('/')[-1].split('_')[:3]
        stores_to_scenes[store].append(scene)
    print("Number unique scenes per store:")
    for store in stores:
        print(store, len(set(stores_to_scenes[store])))
    # Get number of samples per store
    print("Number of samples per store:")
    for k,v in stores_to_scenes.items():
        print(k, len(v))
    # Get number of cameras per scene
    store_scenes_to_cams = dict()
    for k,v in stores_to_scenes.items():
        for i in range(len(v)):
            store_scenes_to_cams[k+'_'+v[i]] = 0
    for i in range(len(train_samples)):
        store, scene, camera = train_samples[i]['label'].split('/')[-1].split('_')[:3]
        store_scenes_to_cams[store+'_'+scene] += 1
    print("mean number cams ", sum(store_scenes_to_cams.values()) / len(store_scenes_to_cams))
    
    
    print("Val")
    # Get val samples
    val_samples = [d for d in dataset_samples if '_'.join(d['label'].split('/')[-1].split('_')[:3])[:-10] in val_scenes]
    # Get unique stores
    stores = set([s.split('_')[0] for s in val_scenes])
    print("Stores ", stores)
    # Get number of unique scenes per store
    stores_to_scenes = dict()
    for store in stores:
        stores_to_scenes[store] = []
    for i in range(len(val_samples)):
        store, scene, camera = val_samples[i]['label'].split('/')[-1].split('_')[:3]
        stores_to_scenes[store].append(scene)
    print("Number unique scenes per store:")
    for store in stores:
        print(store, len(set(stores_to_scenes[store])))
    # Get number of samples per store
    print("Number of samples per store:")
    for k,v in stores_to_scenes.items():
        print(k, len(v))
    # Get number of cameras per scene
    store_scenes_to_cams = dict()
    for k,v in stores_to_scenes.items():
        for i in range(len(v)):
            store_scenes_to_cams[k+'_'+v[i]] = 0
    for i in range(len(val_samples)):
        store, scene, camera = val_samples[i]['label'].split('/')[-1].split('_')[:3]
        store_scenes_to_cams[store+'_'+scene] += 1
    print("mean number cams ", sum(store_scenes_to_cams.values()) / len(store_scenes_to_cams))


    print("Test")
    # Get test samples
    test_samples = [d for d in dataset_samples if '_'.join(d['label'].split('/')[-1].split('_')[:3])[:-10] in test_scenes]
    # Get unique stores
    stores = set([s.split('_')[0] for s in test_scenes])
    print("Stores ", stores)
    # Get number of unique scenes per store
    stores_to_scenes = dict()
    for store in stores:
        stores_to_scenes[store] = []
    for i in range(len(test_samples)):
        store, scene, camera = test_samples[i]['label'].split('/')[-1].split('_')[:3]
        stores_to_scenes[store].append(scene)
    print("Number unique scenes per store:")
    for store in stores:
        print(store, len(set(stores_to_scenes[store])))
    # Get number of samples per store
    print("Number of samples per store:")
    for k,v in stores_to_scenes.items():
        print(k, len(v))
    # Get number of cameras per scene
    store_scenes_to_cams = dict()
    for k,v in stores_to_scenes.items():
        for i in range(len(v)):
            store_scenes_to_cams[k+'_'+v[i]] = 0
    for i in range(len(test_samples)):
        store, scene, camera = test_samples[i]['label'].split('/')[-1].split('_')[:3]
        store_scenes_to_cams[store+'_'+scene] += 1
    print("mean number cams ", sum(store_scenes_to_cams.values()) / len(store_scenes_to_cams))

def generate_standalone_depthmaps(data_root, depth_dir):
    """ Generates depth maps with single channel from .exr files and removes inf values
    @param data_root (str): Path to dataset root dir
    @param depth_dir (str): Directory to save new depth .exr to
    """
    files = find_files(data_root)
    for f in files:
        depth1_name = f['label'].replace("-label.png", "_change-0-depth0001.exr")
        depth2_name = f['label'].replace("-label.png", "_change-1-depth0001.exr")
        depth1, depth2 = (
            imageio.imread(depth1_name, format="EXR-FI"),
            imageio.imread(depth2_name, format="EXR-FI")
        )
        try:
            depth1, depth2 = depth1[:,:,0], depth2[:,:,0]
            depth1[depth1>100] = -1
            depth1[depth1==-1] = np.max(depth1)
            depth2[depth2>100] = -1
            depth2[depth2==-1] = np.max(depth2)
            imageio.imwrite(depth1_name, depth1)
            imageio.imwrite(depth2_name, depth2)
        except:
            pass

if __name__ == "__main__":
    #move_scene_errors('/renders_multicam_diff_1', '/scene_error_renders_multicam_diff_1', 'scene_errors.txt'):
    #remove_scenes_from_imageset('scene_errors.txt' , '/synthetic_data_baselines/imagesets')
    analyze_scenes('/renders_multicam_diff_1', '/synthetic_data_baselines/imagesets')
    #generate_standalone_depthmaps('/renders_multicam_diff_1', '/renders_multicam_diff_1_depth')

