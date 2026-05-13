1

Hello, my name is Marek Hric, and this is a short introduction to my bachelor's thesis, Speech/Music Classification, supervised by Mr. Malenovský.

---
2

Its main goal was to design and implement an on-line speech/music classifier and benchmark it against existing solutions on a sufficiently diverse dataset.
I have slightly adjusted this goal to handle three classes: speech, music, and background noise.

---
3

The goal was achieved in three main steps.
I started by constructing the dataset.
Its final version has roughly a hundred hours with the 40/40/20 split shown on the graph.
Speech and music also break down into a number of subclasses with reasonably even distributions.

---
4

Secondly, I implemented four reference models — three traditional and one neural — adapted them to three-class classification and benchmarked them.
The temporal convolution network clearly came out on top, with the strongest traditional reference about ten macro F1 points behind.

---
5

To derive the proposed method, I started from the TCN as a baseline and came to the proposal experimentaly.

During those experiments I found that gaining a single percentage point of macro F1 required roughly twenty times more parameters.
I therefore switched the goal to matching the reference accuracy with a much smaller and cheaper model.

These experiments led me to the proposed TCN-S.
It matches the reference TCN's macro F1 score with seven times fewer parameters, and runs 5.3 times faster than real time on a single CPU thread compared to a TCN's 3.2.

Thank you for your attention and here is a short demo of it running live.
