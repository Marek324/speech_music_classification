# Speech-Music classification dataset
## Splits
### Mini Tier
Total time: 13 minutes
| Split | % | Minutes |
| --- | --- | --- |
| Train | 70 | 9.1  |
| Val   | 15 | 1.95 |
| Test  | 15 | 1.95 |

### Mid Tier
Total time: 1200 min (20 hours)
| Split | %  | Minutes |
|------ |----|---------|
| Train | 80 | 960    |
| Val   | 10 | 120    |
| Test  | 10 | 120    |

### Full Tier
Total time: 12000 min (200 hours)
| Split | %  | Minutes |
|------ |----|---------|
| Train | 84 | 10080  |
| Val   | 8  | 960    |
| Test  | 8  | 960    |


## Subclasses
| Subclass | %  | Mini | Mid | Full |
| --- | --- | --- | --- | --- |
| **Speech** | 40*  | 6   | 480  | 4800  |
| Clean      | 40  | 5   | 192  | 1920  |
| Dirty      | 20  | —   | 96   | 960   |
| Noisy      | 20  | —   | 96   | 960   |
| Multispeaker | 7.5 | — | 36   | 360   |
| SoM\*\*\* (Speech over Music) | 7.5 | 1 | 36 | 360 |
| MSoM\*\*\* (Multi-Speaker over Music) | 5 | — | 24 | 240 |
| **Music** | 40* | 5 | 480 | 4800 |
| Each FMA genre (×6) | — | — | 72 | 750 |
| Acapella | — | — | 48 | 300 |
| Pop (mini) | — | 5 | — | — |
| **Inactive** | 20* | 2 | 240 | 2400 |

\* % in whole dataset
\*\* Music genres: Instrumental, Electronic, Pop, Rock, Acapella, Hip-Hop, Folk
\*\*\* Augmented subclasses: Speech over Music, Multi-Speaker over Music