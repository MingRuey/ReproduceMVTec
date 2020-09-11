from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

from MLBOX.Database.core.database import DBLoader
from tensorflow.python.eager.context import export_run_metadata
from Dataset.feature_fmt import MVTectFMT

DST_DIR = "/rawdata2/TFDataset/MVTecAD"


class TestViewMVTecAD:

    @staticmethod
    def export_example(outdir, example):
        image = example["image_content"].numpy()[0, ...].astype(np.uint8)
        image = image[..., ::-1]
        image = np.array(image)

        mask = example["mask"].numpy()[0, ...].astype(np.uint8)

        imgid = example["image_id"][0].numpy().decode()
        label = example["classes"][0].numpy()
        cate = label[0].decode()
        broken_type = label[1].decode()

        imginfo = "{}_cate-{}_type-{}.bmp".format(imgid, cate, broken_type)
        maskinfo = "{}_cate-{}_type-{}_mask.bmp".format(imgid, cate, broken_type)

        cv2.imwrite(str(outdir.joinpath(imginfo)), image)
        cv2.imwrite(str(outdir.joinpath(maskinfo)), mask)

    def test_view_train(self, tmp_path: Path):
        parser = MVTectFMT()
        loader = DBLoader()

        loader.load(DST_DIR, parser=parser)
        train = loader.train.to_tfdataset(1, 1)
        for cnt, example in enumerate(train):
            self.export_example(tmp_path, example)
            if cnt == 5:
                break

        print("MVTecAd train example export to {}".format(tmp_path))

    def test_view_Test(self, tmp_path: Path):
        parser = MVTectFMT()
        loader = DBLoader()

        loader.load(DST_DIR, parser=parser)
        test = loader.test.to_tfdataset(batch=1, epoch=1)

        good_cnt = 0
        broken_cnt = 0
        for example in test:
            label = example["classes"][0].numpy()
            broken_type = label[1].decode()
            if broken_type == "good":
                subdir = tmp_path.joinpath("good")
                subdir.mkdir(exist_ok=True)
                self.export_example(subdir, example)
                good_cnt += 1
            else:
                subdir = tmp_path.joinpath("broken")
                subdir.mkdir(exist_ok=True)
                self.export_example(subdir, example)
                broken_cnt += 1

            if (good_cnt >= 5 and broken_cnt >= 5):
                break

        print("MVTecAd test example export to {}".format(tmp_path))