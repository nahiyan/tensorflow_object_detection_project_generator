import os
import os.path
from clint.textui import progress
import requests
import zipfile
import shutil
import fileinput
import csv
import glob


def find_string(file_path, string):
    with open(file_path) as f:
        datafile = f.readlines()

    for line in datafile:
        if string in line:
            return True

    return False


def item_count(path):
    items = os.listdir(path) # dir is your directory path
    return len(items)


def softcreate(name):
    if os.path.exists(name):
        print("File/Directory already exists: " + name)

    else:
        os.mkdir(name)
        print("Created: " + name + ".")


def download(url, path):
    if os.path.exists(path) == False:
        print("Downloading: " + url + ".")

        r = requests.get(url, stream = True)
        with open('download.data', 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size = 1024), expected_size = (total_length / 1024) + 1): 
                if chunk:
                    f.write(chunk)
                    f.flush()

        os.rename("download.data", path)

    else:
        print("Already downloaded: " + url)


def unzip(src_path, dest_path, file_name):
    if os.path.exists(os.path.join(dest_path, file_name)) == False:
        shutil.unpack_archive(src_path, dest_path)

        print("Unzipped: " + src_path)

    else:
        print("Already exists: " + file_name)


def move_dir_contents(src_path, dest_path):
    files = os.listdir(src_path)

    for f in files:
        try:
            shutil.move(os.path.join(src_path, f), dest_path)
        except:
            pass


def copy_dir_contents(src_path, dest_path):
    files = os.listdir(src_path)

    for f in files:
        try:
            shutil.copy(os.path.join(src_path, f), dest_path)
        except:
            pass


def remove_dir_contents(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            pass


def move_stuff_to_od_dir(project_name):
    od_dir_path = os.path.join(project_name, "models-master", "research", "object_detection")

    tutorial_path2 = os.path.join(project_name, "TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10-master")

    if os.path.exists(os.path.join(tutorial_path2, "images")) == True:
        remove_dir_contents(os.path.join(tutorial_path2, "images", "train"))
        remove_dir_contents(os.path.join(tutorial_path2, "images", "test"))
        remove_dir_contents(os.path.join(tutorial_path2, "training"))
        remove_dir_contents(os.path.join(tutorial_path2, "inference_graph"))

        try:
            os.remove(os.path.join(tutorial_path2, "images", "test_labels.csv"))
            os.remove(os.path.join(tutorial_path2, "images", "train_labels.csv"))
        except:
            pass

        move_dir_contents(tutorial_path2, od_dir_path)
        print("Moved all files to object detection folder.")

    else:
        print("Nothing to move to object detection folder.")


def compile_protobuf_files(project_name):
    research_path = os.path.join(project_name, "models-master", "research")

    if os.path.exists(os.path.join(research_path, "object_detection", "protos", "train_pb2.py")) == False:

        os.chdir(research_path)

        os.system("protoc --python_out=. ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "anchor_generator.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "argmax_matcher.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "bipartite_matcher.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "box_coder.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "box_predictor.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "eval.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "faster_rcnn.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "faster_rcnn_box_coder.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "grid_anchor_generator.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "hyperparams.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "image_resizer.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "input_reader.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "losses.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "matcher.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "mean_stddev_box_coder.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "model.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "optimizer.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "pipeline.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "post_processing.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "preprocessor.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "region_similarity_calculator.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "square_box_coder.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "ssd.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "ssd_anchor_generator.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "string_int_label_map.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "train.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "keypoint_box_coder.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "multiscale_anchor_generator.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "graph_rewriter.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "calibration.proto ." + os.sep + "object_detection" + os.sep + "protos" + os.sep + "flexible_grid_anchor_generator.proto")

        print("Protobuf files compiled successfully.")

        os.chdir("../../../")

    else:
        print("Protobuf files already compiled.")


def research_stuff_build_and_install(project_name):
    research_path = os.path.join(project_name, "models-master", "research")

    if os.path.exists(os.path.join(research_path, "build")) == False:
        os.chdir(research_path)
        os.system("python setup.py build")
        os.system("python setup.py install")

        print("Built and installed research stuff successfully.")

        os.chdir("../../../")

    else:
        print("Already built and installed research stuff.")


def import_dataset(project_name):
    dest_path = os.path.join(project_name, "models-master", "research", "object_detection", "images")

    dataset_test_images = os.path.join("dataset", "test")
    dataset_train_images = os.path.join("dataset", "train")

    if item_count(os.path.join(dest_path, "train")) == 0:
        copy_dir_contents(dataset_test_images, os.path.join(dest_path, "test"))
        copy_dir_contents(dataset_train_images, os.path.join(dest_path, "train"))

        print("Imported dataset.")

    else:
        print("Already imported dataset.")


def generate_tfrecord(project_name):
    od_dir_path = os.path.join(project_name, "models-master", "research", "object_detection")

    # CSV related stuff

    if os.path.exists(os.path.join(od_dir_path, "images", "train_labels.csv")) == False:

        os.chdir(od_dir_path)

        os.system("python xml_to_csv.py")

        os.chdir("../../../../")

    else:
        print("Label CSV files exists already.")

    # Process generate_tfrecord.py file

    classes = get_label_classes()

    os.chdir(od_dir_path)

    if find_string("generate_tfrecord.py", "def class_text_to_int_old(row_label):") == False:

        # Read in the file
        with open("generate_tfrecord.py", 'r') as file :
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace("def class_text_to_int(row_label):", "def class_text_to_int_old(row_label):").replace("# TO-DO replace this with label map", generate_class_text_to_int(classes))

        # Write the file out again
        with open("generate_tfrecord.py", 'w') as file:
            file.write(filedata)

        print("Modified generate_tfrecord.py.")

    else:
        print("Already modified generate_tfrecord.py.")

    if os.path.exists("train.record") == False:
        os.system("python generate_tfrecord.py --csv_input=images" + os.sep + "train_labels.csv --image_dir=images" + os.sep + "train --output_path=train.record")

        os.system("python generate_tfrecord.py --csv_input=images" + os.sep + "test_labels.csv --image_dir=images" + os.sep + "test --output_path=test.record")

    else:
        print("TFRecord files already exists.")

    os.chdir("../../../../")




def generate_class_text_to_int(items):
    code = "def class_text_to_int(row_label):\n"

    counter = 0
    for item in items:
        if counter == 0:
            code += "    if row_label == '" + item + "':\n"
            code += "        return 1\n"
        else:
            code += "    elif row_label == '" + item + "':\n"
            code += "        return " + str((counter + 1)) + "\n"


        counter += 1

    code += "    else:\n"
    code += "        None\n\n"

    return code


def get_label_classes():
    classes = []

    with open('label_classes.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for row in csv_reader:
            for col in row:
                classes.append(col)

    return classes


def get_num_classes():
    return len(get_label_classes())


def label_map_file_contents(items):
    code = ""

    counter = 1
    for item in items:
        code += "item {\n"
        code += "    id: " + str(counter) + "\n"
        code += "    name: '" + item + "'\n"
        code += "}\n\n"

        counter += 1

    return code


def make_label_map(project_name):
    training_path = os.path.join(project_name, "models-master", "research", "object_detection", "training")

    items = get_label_classes()

    os.chdir(training_path)

    if os.path.exists("labelmap.pbtxt") == False:
        with open("labelmap.pbtxt", "w") as f:
            f.write(label_map_file_contents(items))

        print("Created label map successfully.")

    else:
        print("Label map already exists.")

    os.chdir("../../../../../")


def get_model_config_name():
    items = []

    with open('model_info.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for row in csv_reader:
            for col in row:
                items.append(col)

    return items[2].strip()


def get_model_url():
    items = []

    with open('model_info.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for row in csv_reader:
            for col in row:
                items.append(col)

    return items[1].strip()


def copy_model_config_file(project_name):
    models_config_dir_path = os.path.join(project_name, "models-master", "research", "object_detection", "samples", "configs")

    model_config_file_name = get_model_config_name()

    dest_path = os.path.join(project_name, "models-master", "research", "object_detection", "training", model_config_file_name)

    if os.path.exists(dest_path) == False:
        shutil.copyfile(os.path.join(models_config_dir_path, model_config_file_name), dest_path)

        print("Copied model config file.")

    else:
        print("Model config file already copied.")


def get_model_name():
    items = []

    with open('model_info.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for row in csv_reader:
            for col in row:
                items.append(col)

    return items[0].strip()


def reconfig_model_config(project_name):
    training_path = os.path.join(project_name, "models-master", "research", "object_detection", "training")

    fine_tune_checkpoint = os.path.join(get_model_name(), "model.ckpt")
    label_map_path = "training/labelmap.pbtxt"
    train_input_path = "train.record"
    test_input_path = "test.record"
    model_config_file_name = get_model_config_name()
    num_classes = get_num_classes()

    os.chdir(training_path)

    num_examples = get_num_examples()

    # Read in the file
    with open(model_config_file_name, 'r') as file :
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace("num_classes: 90", "num_classes: " + str(num_classes)).replace("PATH_TO_BE_CONFIGURED/model.ckpt", fine_tune_checkpoint).replace("num_examples: 8000", "num_examples: " + str(num_examples)).replace("PATH_TO_BE_CONFIGURED/mscoco_train.record-?????-of-00100", train_input_path).replace("PATH_TO_BE_CONFIGURED/mscoco_val.record-?????-of-00010", test_input_path).replace("PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt", label_map_path)


    # Write the file out again
    with open(model_config_file_name, 'w') as file:
        file.write(filedata)

    print("Reconfigured config file.")

    os.chdir("../../../../../")


def get_num_examples():
    xml_files = os.path.join("..", "images", "test", "*.xml")
    all_files = os.path.join("..", "images", "test", "*")
    all_files_count = len(glob.glob(all_files))
    xml_files_count = len(glob.glob(xml_files))

    return all_files_count - xml_files_count


def run_training(project_name):
    od_path = os.path.join(project_name, "models-master", "research", "object_detection")

    config_name = get_model_config_name()

    os.chdir(od_path)

    os.system("tensorboard --logdir=training")

    os.system("python legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/" + config_name)

    os.chdir("../../../../../")