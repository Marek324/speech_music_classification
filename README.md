## Packages
This project is using mamba as it's main package manager but it should also be compatible with pip/uv as well
With mamba and make available run:
```
make setup
make activate
```

## Dataset
```
source .env
git clone git@hf.co:$DATASET_URL
```

## Don't forget  

- [ ] metacentrum citation (https://docs.metacentrum.cz/en/docs/access/terms)

## Notes

Segment feature statistic
- broadcast statistics to all frames in segment for now
- sliding window for real time
