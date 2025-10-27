# main.py
# Marek Hric

import sys
from tqdm import tqdm
import warnings

if "-h" in sys.argv or "--help" in sys.argv:
    # TODO: implement help message
    print(f"Usage: python {sys.argv[0]}")
    sys.exit(0)


from sm_lib import (
    SMDataset,
    SMDatasetBuilder
)

from classifiers import (
    SMClassifier,
    SMDecisionTree
)

warnings.filterwarnings('ignore', message='n_fft=.* is too large for input signal of length=.*')


def main():
    d_builder = SMDatasetBuilder()

    train_data = d_builder.build(train=True)
    classifier = SMDecisionTree()
    classifier.fit(train_data)

    test_data = d_builder.build(train=False)
    print("eval: %0.4f" % evaluate(classifier, test_data))

def evaluate(clf: SMClassifier, data: SMDataset) -> float:
    correct: int = 0 
    total: int = data.xs.shape[0]

    for frame, t in tqdm(zip(data.xs, data.targets), desc='Evaluating'):
        if clf.predict(frame) * t > 0:
            correct += 1

    return correct / total




if __name__ == "__main__":
    main()
