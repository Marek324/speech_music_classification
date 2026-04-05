# Speech-Music classification dataset

> **Note**: All minute counts in this file are *targets*. Actual minutes written per subclass/split
> depend on how much data is available from each HF source. After a build, the summary table printed
> by `build.py` shows actual vs target per subclass and flags any subclass with 0 minutes in val/test.

Top-level mix: **40% speech / 40% music / 20% inactive** (same for mid and full tier totals).

## Splits

### Mid Tier
Total time: 1200 min (20 hours)
| Split | %  | Minutes |
|------ |----|---------|
| Train | 80 | 960    |
| Val   | 10 | 120    |
| Test  | 10 | 120    |

### Full Tier
Total time: 6000 min (100 hours)
| Split | %  | Minutes |
|------ |----|---------|
| Train | 84 | 5040   |
| Val   | 8  | 480    |
| Test  | 8  | 480    |

## Subclasses
| Subclass | %  | Mid | Full |
| --- | --- | --- | --- |
| **Speech** | 40*  | 480  | 2400  |
| Clean      | 40  | 192  | 960   |
| Dirty      | 20  | 96   | 480   |
| Noisy**    | 20  | 96   | 480   |
| Multispeaker** (LibriMix-style) | 7.5 | 36  | 180   |
| SoM** (Speech over Music) | 7.5 | 36 | 180 |
| MSoM** (Multi-Speaker over Music) | 5 | 24 | 120 |
| **Music** | 40* | 480 | 2400 |
| Each music genre (×7, equal)*** | — | 480/7 | 2400/7 |
| **Inactive** | 20* | 240 | 1200 |

\* % of whole dataset  
\*\* Augmented subclasses (synthetic): Multispeaker mixes pairs of speech_clean clips; SoM/MSoM mix speech with FMA music; Noisy mixes speech with DEMAND noise.  
\*\*\* Six FMA genres (Instrumental, Electronic, Pop, Rock, Hip-Hop, Folk) plus acapella (bel_canto); same target minutes per source in `sources.toml` (~68.57 min mid, ~342.86 min full).
