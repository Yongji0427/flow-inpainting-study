# Data inputs

Download Oxford-IIIT Pet images and annotations from the [official dataset page](https://www.robots.ox.ac.uk/~vgg/data/pets/). Dataset images, annotations, prepared image tensors, and mask banks are not included in this repository.

`splits.csv` is the exact inherited manifest referenced by the frozen original-test protocol. It contains image IDs, image metadata/hashes, and train/validation/test assignments. Its SHA-256 is:

```
dfa10dadae4ebae46170edc7e38c3206975de8e675e9a4c4bb8be1d3ab12d51c
```

Manifest assignments are 2,924 train, 730 validation, and 3,669 test images. The restoration development pool uses 128 of the validation images before mask matching. The prepared-data IDs are documented in `GeoStep/data_manifest.json`. Final support counts after matching are different from these offered counts; see `docs/PROTOCOL.md`.

The legacy preparation path is `/data/LGFNet/manifests/splits.csv`, and the annotation reader expects `/data/lgfnet_data/oxford-iiit-pet`. These are recorded execution defaults, not automatic downloads or guaranteed portable paths. See `docs/REPRODUCE.md` for the remaining historical prerequisites.

The small set of 20 qualitative display tiles used in the report is stored
under `results/qualitative/`, with selection provenance. It does not replace
the original dataset or provide a complete test prediction archive.
