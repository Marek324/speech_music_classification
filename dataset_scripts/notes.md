# Speech-Music classification dataset
## Splits
### Mini Tier
Total time: 20 minutes
| Split | % | Minutes |
| --- | --- | --- |
| Train | 70 | 14 |
| Val   | 15 | 3.6  |
| Test  | 15 | 3.6  |

### Mid Tier
Total time: 1200min (20hours)
| Split | % | Minutes |
| --- | --- | --- |
| Train | 80 | 3840 |
| Val   | 10 | 480  |
| Test  | 10 | 480  |

### Full Tier
Total time: 12000min (200hours)
| Split | % | Minutes |
| --- | --- | --- |
| Train | 84 | 40320 |
| Val   | 8  | 3840  |
| Test  | 8  | 3840  |


## Subclasses
| Subclass | %  | Mini | Mid | Full |
| --- | --- | --- | --- | --- |
| **Speech** | 40*  | 5.5 | 480  | 4800  |
| Clean      | 40  | 5   | 192  | 1920  |
| Dirty      | 20  | —   | 96   | 960   |
| Noisy      | 20  | —   | 96   | 960   |
| Multispeaker | 7.5 | — | 36   | 360   |
| SoM\*\*\* (Speech over Music) | 7.5 | 0.5 | 36 | 360 |
| MSoM\*\*\* (Multi-Speaker over Music) | 5 | — | 24 | 240 |
| **Music** | 40* | 5 | 400 | 4000 |
| Each genre** | — | — | 40 | 400 |
| Pop (mini) | — | 5 | — | — |
| **Inactive** | 20* | 2 | 240 | 2400 |

\* % in whole dataset
\*\* Music genres: Instrumental, Electronic, Pop, Rock, Acapella, Hip-Hop, Folk, Metal, Jazz, Country
\*\*\* Augmented subclasses: Speech over Music, Multi-Speaker over Music