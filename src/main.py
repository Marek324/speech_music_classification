# main.py
# Marek Hric

import sys

if "-h" in sys.argv or "--help" in sys.argv:
    # TODO: implement help message
    print(f"Usage: python {sys.argv[0]}")
    sys.exit(0)


from sm_lib import (
    SMDataset,
    SMDatasetBuilder
)

from classifiers import (
    SMDecisionTree
)


def main():
    d_builder: SMDatasetBuilder = SMDatasetBuilder()
    train_data: SMDataset = d_builder.build(train=True)
    classifier: SMDecisionTree = SMDecisionTree().fit(train_data) # noqa: F841
    #evaluate(classifier, load_reference(train=False))

#def create_features(ref: list[dict[str, list[float]]]) -> tuple[np.ndarray, np.ndarray]:
#    speech_list = []
#    music_list = []
#
#    extractor = SMFeatureExtractor(sr=defaults.SAMPLE_RATE)
#    seg_len = seg_len_ms * defaults.SAMPLE_RATE // 1000
#    f_len = f_len_ms * defaults.SAMPLE_RATE // 1000
#    f_hop = f_hop_ms * defaults.SAMPLE_RATE // 1000
#    seg_frames = int(np.floor((seg_len - f_len) / f_hop) + 1)
#    seg_stats = SMSegmentStatistics(seg_frames=seg_frames)
#
#    for entry in ref:
#
#        s = load_and_resample(f"{defaults.DATASET_PATH}/{entry['file']}")
#
#        if DEBUG:
#            if any(entry['file'].str.startswith(d) for d in ['train/speech', 'train/m+s']) and len(speech_list) > DEBUG_SAMP_MIN: continue # noqa: E701
#            if entry['file'].str.startswith('train/music') and len(music_list) > DEBUG_SAMP_MIN: continue # noqa: E701
##            print(f"speech_list len: {len(speech_list)}")
##            print(f"music_list len: {len(music_list)}")
#
#        frames = framing(s)
#
#        stat_win = deque(maxlen=seg_frame_count())
#
#        for frame, ref_c in zip(frames, entry['ref']):
#            feats = extractor.extract(frame)
#            stat_win.append(feats)
#            stats = feature_statistics(np.array(list(stat_win)))
#            feats = np.hstack((feats, stats))
#            match ref_c:
#                case 0: # silence
#                    continue
#                case 1: # speech
#                    speech_list.append(feats)
#                case 2: # music
#                    music_list.append(feats)
#
#
#        # reset extractor memory for new file
#        extractor.last_frame = None
#        extractor.last_mfcc = None
#
#
#    X_speech = np.vstack(speech_list)
#    X_music = np.vstack(music_list)
#
#    return X_speech, X_music
#

def evaluate(clf: SMDecisionTree, ref: list[dict[str, list[float]]]) -> float:
    pass
#    for entry in ref:
#        res = clf.predict(load_and_resample(f"{defaults.DATASET_PATH}/{entry['file']}")) # noqa: F841

    return 0.0




if __name__ == "__main__":
    main()
