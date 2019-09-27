import os
import helper

project_name = "project"

# Create the project's directory if it doesn't exist already.

helper.softcreate(project_name)


# Download the tutorial's repository

tutorial_path = os.path.join(project_name, "tutorial.zip")

helper.download("https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/archive/master.zip", tutorial_path)

helper.unzip(tutorial_path, project_name, "TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10-master")

# Download the model's directory

model_path = os.path.join(project_name, "model.zip")

helper.download("https://github.com/tensorflow/models/archive/master.zip", model_path)

helper.unzip(model_path, project_name, "models-master")

# Download the model from model zoo

model_path2 = os.path.join(project_name, "model2.tar.gz")

helper.download(helper.get_model_url(), model_path2)

od_dir = os.path.join(project_name, "models-master", "research", "object_detection")

helper.unzip(model_path2, od_dir, helper.get_model_name())

# Move stuff to the object detection directory

helper.move_stuff_to_od_dir(project_name)

# Deal with protobuf files

helper.compile_protobuf_files(project_name)

# Research stuff build and install

helper.research_stuff_build_and_install(project_name)

# Move images to object_detection directory

helper.import_dataset(project_name)

# Generate tfrecord

helper.generate_tfrecord(project_name)

# Make the label map

helper.make_label_map(project_name)

# Copy the model configuration file

helper.copy_model_config_file(project_name)

# Reconfig model config file

helper.reconfig_model_config(project_name)

# Run the training

helper.run_training(project_name)