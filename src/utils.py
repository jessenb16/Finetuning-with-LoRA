import os

def is_kaggle():
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

def get_paths():
    if is_kaggle():
        inference_data_path = "/kaggle/input/deep-learning-spring-2025-project-2/test_unlabelled.pkl"
        output_path = '/kaggle/working/output'

    else:
        inference_data_path = "test_unlabelled.pkl"
        output_path = "output"

    return inference_data_path, output_path

