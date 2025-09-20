## Packages
This project is using mamba as it's main package manager but it should also be compatible with pip/uv as well
With mamba and make available run:
```
make setup
make activate
```

## Dataset
```
git clone git@hf.co:$DATASET_URL
```

## Files
### frame_labeling.py
The Schreirer-Slaney music-speech corpus is labeled per-file, which is a bit of problem for this project.
This script creates frame-level labels and was also created for reproducibility purposes as the database isn't publicly available.

## Don't forget  

- [ ] metacentrum citation (https://docs.metacentrum.cz/en/docs/access/terms)

## Notes
Metacentrum hugging face SSH key

Segment feature statistic
- broadcast statistics to all frames in segment for now
- sliding window for real time
