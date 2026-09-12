# Legacy tests

These tests were written against the original code and exercise behaviour that
has since been removed deliberately: fabricated datasets, a classifier that
could be told which class to return, randomly generated licence plates, an
"IMU reading" derived from the camera model's own output.

They are kept for reference, not run. The current suite is
`tests/test_road_shield.py`, which tests what the system actually does.
