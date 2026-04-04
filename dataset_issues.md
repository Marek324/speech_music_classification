# Dataset Issues — Sources That Missed Targets

Based on `real_dataset_times`.

---

## All issues resolved

Both previous issues have been addressed:

1. **speech_multispeaker (AMI)** — replaced with LibriMix-style synthetic augmentation
   (pairs of speech_clean clips mixed together). No longer depends on AMI; generates
   exactly the target minutes from the existing speech_clean pool.

2. **FMA genres (53% of target)** — full tier rescaled from 12000 min to 6000 min
   (100 hours). At 375 min/genre, every FMA genre has ~393+ min available (headroom).

All subclasses should now hit 100% of their targets in both mid and full tiers.
