import sys
import os
SERVER_ROOT = os.path.dirname(sys.modules[__name__].__file__)
SERVER_ROOT = os.path.join(SERVER_ROOT, "..")

class Config(object):
    def __init__(self):
        #raw data root
        self.raw_data_root = os.path.join(SERVER_ROOT,"../../RawData")
        self.cropping_root = os.path.join(SERVER_ROOT, "../../Codes/mingming/test_images")

        # first-level directory
        self.data_root = os.path.join(SERVER_ROOT,"../data/")
        self.scripts_root = os.path.join(SERVER_ROOT, "../scripts/")
        self.log_root = os.path.join(SERVER_ROOT,"../logs/")
        self.model_root = os.path.join(SERVER_ROOT, "../model/")


        #extension
        self.image_ext = ".jpg"
        self.mat_ext = ".mat"
        self.pkl_ext = ".pkl"

        # second-level directory
        self.breast_cancer = "BreastCancer"
        self.pidd = "PIDD"
        self.three_dim = "ThreeDim"
        self.two_dim = "TwoDim"
        self.two_dim_iris = "TwoDimIris"
        self.mnist = "MNIST"
        self.cifar10 = "CIFAR10"
        self.mnist_1_2 = "MNIST-1-2"
        self.mnist_3_5 = "MNIST-3-5"
        self.mnist_vgg = "MNIST-vgg"
        self.mnist_1_2_vgg = "MNIST-1-2-vgg"
        self.mnist_3_5_vgg = "MNIST-3-5-vgg"
        self.dog_cat = "Dog-Cat"
        self.animals = "Animals"
        self.animals_step1 = "Animals-step1"
        self.animals_step2 = "Animals-step2"
        self.animals_step3 = "Animals-step3"
        self.animals_step4 = "Animals-step4"
        self.dog_cat_extension = "Dog-Cat-Extension"
        self.dog_cat_imagenet = "Dog-Cat-Imagenet"
        self.dog_cat_combine = "Dog-Cat-Combine"
        self.dog_cat_normal_train = "Dog-Cat-Normal-Train"
        self.svhn = "SVHN"
        self.svhn_imagenet = "SVHN-Imagenet"
        self.svhn_combine = "SVHN-Combine"
        self.lipsticks = "Lipsticks"
        self.rea = "REA"

        # third-level directory
        self.train_path_name = "train"
        self.test_path_name = "test"
        self.all_data_cache_name = "all_data_cache" + self.pkl_ext

        # filename
        self.processed_dataname = "processed_data"+ self.pkl_ext
        self.grid_dataname = "grid_data" + self.pkl_ext

        # variable
        self.class_name = "class_name"
        self.X_name ="X_name"
        self.embed_X_name = "embed_X"
        self.all_embed_X_name = "all_embed_X"
        self.y_name = "y_name"
        self.train_idx_name = "train_idx"
        self.train_redundant_idx_name = "train_redundant_idx"
        self.valid_idx_name = "valid_idx"
        self.valid_redundant_idx_name = "valid_redundant_idx"
        self.test_idx_name = "test_idx"
        self.test_redundant_idx_name = "test_redundant_idx"
        self.add_info_name = "add_info"
        self.grid_X_train_name = "grid_X_train"
        self.grid_X_test_name = "grid_X_test"


        self.train_x_name = "X_train"
        self.train_y_name = "y_train"
        self.test_x_name = "X_test"
        self.test_y_name = "y_test"

        # variables name for frontend
        self.train_instance_num_name = "TrainInstanceNum"
        self.valid_instance_num_name = "ValidInstanceNum"
        self.test_instance_num_name = "TestInstanceNum"
        self.label_names_name = "LabelNames"
        self.feature_dim_name = "FeatureDim"

        self.fine_tune_feature = "fine_tune_feature"
        self.pretrain_feature = "pretrain_feature"
        self.superpixel_feature = "superpixel_feature"


config = Config()
