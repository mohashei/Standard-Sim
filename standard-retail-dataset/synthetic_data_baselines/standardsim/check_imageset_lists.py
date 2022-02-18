"""
Checks if images for each split exist in data root
"""
import os

if __name__ == "__main__":
    data_root = "/app/renders_multicam_diff_1"
    imagesets_dir = "/app/synthetic_data_baselines/imagesets"

    # Get scenes with errors
    with open("/app/synthetic_data_baselines/utils/scene_errors.txt", 'r') as errorFile:
        scene_errors = errorFile.readlines()
    scene_errors = [s[:-1] for s in scene_errors]
    print("Num scenes with error ", len(scene_errors))
    
    # Get number of scenes in errors dir
    errors_dir = "/app/scene_error_renders_multicam_diff_1"
    error_scenes = ["_".join(f.split('_')[:3]) for f in os.listdir(errors_dir) if "_change-0.png" in f]
    print("Num scenes in error dir ", len(error_scenes))
    
    # Get scenes in error dir that don't exist in scene errors file
    for s in error_scenes:
        if s not in scene_errors:
            print(s, " in scene errors directory but not in scene errors file")

    # Train set
    with open(os.path.join(imagesets_dir, "train.txt"), 'r') as trainFile:
        train_scenes = trainFile.readlines()
    train_scenes = [t[:-1] for t in train_scenes]
    for scene in train_scenes:
        if not os.path.isfile(os.path.join(data_root, scene+"_change-0.png")):
            print(scene, " in train set not in data root")

    # Val set
    with open(os.path.join(imagesets_dir, "val.txt"), 'r') as valFile:
        val_scenes = valFile.readlines()
    val_scenes = [v[:-1] for v in val_scenes]
    for scene in val_scenes:
        if not os.path.isfile(os.path.join(data_root, scene+"_change-0.png")):
            print(scene, " in val set not in data root")

    # Test set
    with open(os.path.join(imagesets_dir, "test.txt"), 'r') as testFile:
        test_scenes = testFile.readlines()
    test_scenes = [t[:-1] for t in test_scenes]
    for scene in test_scenes:
        if not os.path.isfile(os.path.join(data_root, scene+"_change-0.png")):
            print(scene, " in test set not in data root")