# Sample usage

```
python 01_create_dataset.py /data/mldata/io/data 96
python 02_feature_extract.py /data/mldata/io/data laps 32
python 03_clf_baseline.py /data/mldata/io/data laps
python 04_ft.py /data/mldata/io/data laps vgg16 32 1 1
python 05_proj_viz.py /data/mldata/io/data vgg16 laps none
python 05_proj_viz.py /data/mldata/io/data vgg16 laps ft_vgg16_20180802_120431.h5
```
