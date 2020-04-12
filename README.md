# Puzzle-code-detection
# Task
  Detect puzzles on image with single colour background (red in this realization) and print on each puzzle it's code: count of bays
  and peninsulas.
# Realization
  For each puzzle was found its array of coordinates of contour.
  Than each array was rotated many times to get all possible placements for each puzzle.
  As different images has different shapes, so random choice for indexing in X_train was applied.
# View
  Test image
  ![Test image](https://github.com/canek13/Puzzle-code-detection/raw/master/Red_3.jpg)
  Predicted codes and detection
  ![Test image](https://github.com/canek13/Puzzle-code-detection/raw/master/experiment_red3.jpg)
