from torch.utils.data import Subset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import CIFAR10, ImageFolder

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.validation import And, OneOf
from fastargs.decorators import param, section
from fastargs import get_current_config


class TCGA_Dataset(data.Dataset):
    def __init__(self, root, csv_file, mode="train", size=(512, 512)):

        self.size = size
        self.datapath = os.path.join(root, "../../../akash/gigapixel/data/")

        csv_file = f"/workspace/HIPT/2-Weakly-Supervised-Subtyping/splits/10foldcv_subtype/tcga_lung/splits_{csv_file}.csv"
        df = pd.read_csv(csv_file)
        self.class_dict = {"TCGA-LUAD": 0, "TCGA-LUSC": 1}

        files = natsorted(glob(os.path.join(root, "../../../akash/gigapixel/data/*/*.svs"))
                          )

        self.labels_dict = {}
        for filename in files:
            self.labels_dict[filename.split(
                "/")[-1][:-4]] = filename.split("/")[-2]
        # if self.training == "sketch":
        self.Train_List, self.Val_List, self.Test_List = df.train.to_list(
        ), df.val.dropna().to_list(), df.test.dropna().to_list()
        # self.data = "coordinate"

        self.train_transform = get_transform("train", self.size)

        self.val_transform = get_transform("val", self.size)

        self.test_transform = get_transform("val", self.size)

        if mode == "train":
            self.mode = "train"
            print("Total Training Sample {}".format(len(self.Train_List)))
        elif mode == "val":
            self.mode = "val"
            print("Total Validing Sample {}".format(len(self.Val_List)))
        elif mode == "test":
            self.mode = "test"
            print("Total test Sample {}".format(len(self.Test_List)))

    def __getitem__(self, item):

        if self.mode == "train":

            path = self.Train_List[item]
            image_path = os.path.join(
                self.datapath, self.labels_dict[path], path + ".svs")
            # Open the slide
            slide = openslide.open_slide(image_path)
            arr = slide.get_thumbnail(self.size)
            arr = self.train_transform(arr)
            class_name = self.labels_dict[path]

            # sample = {
            #     "image": arr,
            #     "label": self.class_dict[class_name],
            # }
            # return sample

        elif self.mode == "val":

            path = self.Val_List[item]
            image_path = os.path.join(
                self.datapath, self.labels_dict[path], path + ".svs")
            # Open the slide
            slide = openslide.open_slide(image_path)
            arr = slide.get_thumbnail(self.size)
            arr = self.val_transform(arr)
            class_name = self.labels_dict[path]

            # sample = {
            #     "image": arr,
            #     "label": self.name2num[class_name],
            # }
            # return arr, self.name2num[class_name]

        else:

            path = self.Test_List[item]
            # import pdb; pdb.set_trace()
            image_path = os.path.join(
                self.datapath, self.labels_dict[path], path + ".svs")
            # Open the slide
            slide = openslide.open_slide(image_path)
            arr = slide.get_thumbnail(self.size)
            arr = self.test_transform(arr)
            class_name = self.labels_dict[path]

            # sample = {
            #     "image": arr,
            #     "label": self.name2num[class_name],
            # }

        return arr, self.class_dict[class_name]

    def __len__(self):
        if self.mode == "train":
            return len(self.Train_List)
        elif self.mode == "val":
            return len(self.Val_List)
        elif self.mode == "test":
            return len(self.Test_List)


Section('cfg', 'arguments to give the writer').params(
    dataset=Param(And(str, OneOf(['cifar', 'imagenet'])),
                  'Which dataset to write', default='imagenet'),
    split=Param(And(str, OneOf(['train', 'val'])),
                'Train or val set', required=True),
    data_dir=Param(str, 'Where to find the PyTorch dataset', required=True),
    write_path=Param(str, 'Where to write the new dataset', required=True),
    write_mode=Param(str, 'Mode: raw, smart or jpg',
                     required=False, default='smart'),
    max_resolution=Param(int, 'Max image side length', required=True),
    num_workers=Param(int, 'Number of workers to use', default=16),
    chunk_size=Param(int, 'Chunk size for writing', default=100),
    jpeg_quality=Param(float, 'Quality of jpeg images', default=90),
    subset=Param(int, 'How many images to use (-1 for all)', default=-1),
    compress_probability=Param(float, 'compress probability', default=None)
)


@section('cfg')
@param('dataset')
@param('split')
@param('data_dir')
@param('write_path')
@param('max_resolution')
@param('num_workers')
@param('chunk_size')
@param('subset')
@param('jpeg_quality')
@param('write_mode')
@param('compress_probability')
def main(dataset, split, data_dir, write_path, max_resolution, num_workers,
         chunk_size, subset, jpeg_quality, write_mode,
         compress_probability):
    if dataset == 'cifar':
        my_dataset = CIFAR10(root=data_dir, train=(
            split == 'train'), download=True)
    elif dataset == 'imagenet':
        my_dataset = ImageFolder(root=data_dir)
    else:
        raise ValueError('Unrecognized dataset', dataset)

    if subset > 0:
        my_dataset = Subset(my_dataset, range(subset))
    writer = DatasetWriter(write_path, {
        'image': RGBImageField(write_mode=write_mode,
                               max_resolution=max_resolution,
                               compress_probability=compress_probability,
                               jpeg_quality=jpeg_quality),
        'label': IntField(),
    }, num_workers=num_workers)

    writer.from_indexed_dataset(my_dataset, chunksize=chunk_size)


if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
